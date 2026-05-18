#!/usr/bin/env python3
"""Extract LingBot-VA-compatible Wan VAE latents from a LeRobot dataset."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from einops import rearrange
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
except Exception:
    def prompt_clean(text: str) -> str:
        return text


@dataclass
class Args:
    repo_id: str = "local/insert-peg-cylinder-realmachine"
    dataset_root: Path = Path("/data/data_realworld/lerobot_export_dataset/local/insert-peg-cylinder-realmachine")
    wan22_pretrained_model_name_or_path: Path = Path("/data/lingbot-va-models/lingbot-va-base")
    obs_cam_keys: list[str] = field(
        default_factory=lambda: [
            "observation.images.cam_front",
            "observation.images.cam_side",
            "observation.images.cam_fisheye",
        ]
    )
    fps: int = 10
    height: int = 256
    width: int = 256
    max_episodes: int = 0
    device: str = "cuda"
    dtype: str = "bfloat16"
    force: bool = False


def _resolve_lerobot_dataset() -> Any:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset


def _read_episodes(dataset_root: Path) -> list[dict[str, Any]]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found: {episodes_path}")
    episodes = []
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if "action_config" not in item:
                tasks = item.get("tasks") or ["robot manipulation"]
                item["action_config"] = [
                    {
                        "start_frame": 0,
                        "end_frame": int(item["length"]),
                        "action_text": tasks[0],
                    }
                ]
            episodes.append(item)
    return episodes


def _torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _image_to_chw_uint8(image: Any) -> torch.Tensor:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return torch.from_numpy(array).permute(2, 0, 1)
    if torch.is_tensor(image):
        tensor = image.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            pass
        elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported image tensor shape: {tuple(tensor.shape)}")
        if tensor.dtype.is_floating_point and float(tensor.max()) <= 1.5:
            tensor = tensor * 255.0
        return tensor.clamp(0, 255).to(torch.uint8)
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Unsupported image array shape: {array.shape}")
    if array.shape[0] in (1, 3):
        tensor = torch.from_numpy(array)
    else:
        tensor = torch.from_numpy(array).permute(2, 0, 1)
    if tensor.dtype.is_floating_point and float(tensor.max()) <= 1.5:
        tensor = tensor * 255.0
    return tensor.clamp(0, 255).to(torch.uint8)


def _resize_video_spatial(video: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize only H/W for a video tensor shaped [B, C, T, H, W]."""
    if video.ndim != 5:
        raise ValueError(f"Expected video shape [B, C, T, H, W], got {tuple(video.shape)}")
    if int(video.shape[-2]) == int(height) and int(video.shape[-1]) == int(width):
        return video

    batch, _, frames, _, _ = video.shape
    video_4d = rearrange(video, "b c t h w -> (b t) c h w")
    video_4d = F.interpolate(video_4d, size=(height, width), mode="bilinear", align_corners=False)
    return rearrange(video_4d, "(b t) c h w -> b c t h w", b=batch, t=frames)


@torch.no_grad()
def _encode_text(tokenizer, text_encoder, text: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    text_inputs = tokenizer(
        [prompt_clean(text)],
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_device = next(text_encoder.parameters()).device
    output = text_encoder(
        text_inputs.input_ids.to(text_device),
        text_inputs.attention_mask.to(text_device),
    ).last_hidden_state[0]
    seq_len = int(text_inputs.attention_mask[0].gt(0).sum().item())
    output = output[:seq_len].to(device=device, dtype=dtype)
    if seq_len < 512:
        output = torch.cat([output, output.new_zeros(512 - seq_len, output.shape[-1])], dim=0)
    return output


def _episode_start_index(dataset, episode_index: int) -> int:
    if hasattr(dataset, "episode_data_index"):
        value = dataset.episode_data_index["from"][episode_index]
        return int(value.item() if hasattr(value, "item") else value)
    raise AttributeError("LeRobotDataset does not expose episode_data_index")


def _make_wan_frame_ids(start_frame: int, end_frame: int, frame_stride: int) -> list[int]:
    """
    Wan VAE temporal downsampling expects clip lengths of 4n+1 frames.
    Pad by repeating the last valid frame instead of changing action_config bounds.
    """
    frame_ids = list(range(start_frame, end_frame, frame_stride))
    if not frame_ids:
        frame_ids = [start_frame]

    remainder = (len(frame_ids) - 1) % 4
    if remainder != 0:
        pad_count = 4 - remainder
        frame_ids.extend([frame_ids[-1]] * pad_count)
    return frame_ids


@torch.no_grad()
def main(args: Args) -> None:
    from wan_va.modules.utils import load_text_encoder, load_tokenizer, load_vae

    dtype = _torch_dtype(args.dtype)
    device = torch.device(args.device)
    dataset_root = args.dataset_root.expanduser().resolve()

    LeRobotDataset = _resolve_lerobot_dataset()
    dataset = LeRobotDataset(repo_id=args.repo_id, root=str(dataset_root))
    episodes = _read_episodes(dataset_root)
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    model_root = args.wan22_pretrained_model_name_or_path.expanduser().resolve()
    vae = load_vae(str(model_root / "vae"), torch_dtype=dtype, torch_device=device)
    tokenizer = load_tokenizer(str(model_root / "tokenizer"))
    text_encoder = load_text_encoder(str(model_root / "text_encoder"), torch_dtype=dtype, torch_device=device)

    empty_emb = _encode_text(tokenizer, text_encoder, "", device=device, dtype=dtype).cpu()
    torch.save(empty_emb, dataset_root / "empty_emb.pt")

    info_path = dataset_root / "meta" / "info.json"
    ori_fps = args.fps
    if info_path.exists():
        ori_fps = int(json.loads(info_path.read_text(encoding="utf-8")).get("fps", args.fps))
    frame_stride = max(1, round(float(ori_fps) / float(args.fps)))

    latents_mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)

    for episode in tqdm(episodes, desc="episodes"):
        episode_index = int(episode["episode_index"])
        episode_chunk = episode_index // 1000
        ep_global_start = _episode_start_index(dataset, episode_index)

        for segment in episode["action_config"]:
            start_frame = int(segment["start_frame"])
            end_frame = int(segment["end_frame"])
            action_text = str(segment.get("action_text") or (episode.get("tasks") or ["robot manipulation"])[0])
            frame_ids = _make_wan_frame_ids(start_frame, end_frame, frame_stride)
            text_emb = _encode_text(tokenizer, text_encoder, action_text, device=device, dtype=dtype).cpu()

            for key in args.obs_cam_keys:
                out_dir = dataset_root / "latents" / f"chunk-{episode_chunk:03d}" / key
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
                if out_file.exists() and not args.force:
                    continue

                frames = []
                for frame_id in frame_ids:
                    item = dataset[ep_global_start + frame_id]
                    frame = _image_to_chw_uint8(item[key]).float()
                    frames.append(frame)
                video = torch.stack(frames, dim=1).unsqueeze(0)
                video = _resize_video_spatial(video, args.height, args.width)
                video = video.to(device=device, dtype=dtype) / 255.0 * 2.0 - 1.0

                posterior = vae.encode(video).latent_dist
                mu = posterior.mean
                mu_norm = ((mu.float() - latents_mean) * (1.0 / latents_std)).to(dtype)
                latent = rearrange(mu_norm[0].cpu(), "c f h w -> (f h w) c").to(torch.bfloat16)

                payload = {
                    "latent": latent,
                    "latent_num_frames": int(mu_norm.shape[2]),
                    "latent_height": int(mu_norm.shape[3]),
                    "latent_width": int(mu_norm.shape[4]),
                    "video_num_frames": len(frame_ids),
                    "video_height": args.height,
                    "video_width": args.width,
                    "text_emb": text_emb.to(torch.bfloat16),
                    "text": action_text,
                    "frame_ids": frame_ids,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "fps": args.fps,
                    "ori_fps": ori_fps,
                }
                torch.save(payload, out_file)


if __name__ == "__main__":
    main(tyro.cli(Args))
