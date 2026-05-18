#!/usr/bin/env python3
"""Offline validation for LingBot-VA-Tactile checkpoints.

This script evaluates a tactile transformer on an already converted LeRobot
dataset with extracted Wan latents. It computes the same denoising losses used
by tactile training, without starting the robot/server rollout loop.
"""

import argparse
import gc
import json
import os
import random
import sys
from itertools import islice
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WAN_VA_ROOT = REPO_ROOT / "wan_va"

# Keep wan_va first because its modules use original top-level imports such as
# `from configs import VA_CONFIGS`.
for path in (str(WAN_VA_ROOT), str(REPO_ROOT), str(SCRIPT_DIR)):
    while path in sys.path:
        sys.path.remove(path)
sys.path[:0] = [str(WAN_VA_ROOT), str(REPO_ROOT), str(SCRIPT_DIR)]

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from tactile_va.configs import TACTILE_CONFIGS
from tactile_va.configs.stats import apply_stats_json
from tactile_va.dataset import MultiTactileLatentLeRobotDataset
from tactile_va.modules import load_tactile_transformer
from wan_va.utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    sample_timestep_id,
)


def _resolve_transformer_path(checkpoint_path, model_path):
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if (checkpoint_path / "transformer").is_dir():
            return checkpoint_path / "transformer"
        return checkpoint_path

    model_path = Path(model_path).expanduser().resolve()
    if (model_path / "transformer").is_dir():
        return model_path / "transformer"
    return model_path


def _dtype_from_name(name):
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _move_batch_to_device(batch, device):
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


class OfflineTactileValidator:
    def __init__(self, config, checkpoint_path, device, dtype, attn_mode):
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.patch_size = config.patch_size

        transformer_path = _resolve_transformer_path(
            checkpoint_path=checkpoint_path,
            model_path=config.wan22_pretrained_model_name_or_path,
        )
        logger.info(f"Loading tactile transformer from {transformer_path}")
        self.transformer = load_tactile_transformer(
            str(transformer_path),
            torch_dtype=dtype,
            torch_device=self.device,
            attn_mode=attn_mode,
            tactile_dim=config.tactile_dim,
            tactile_shape=getattr(config, "tactile_shape", (2, 32, 58)),
            tactile_token_grid=getattr(config, "tactile_token_grid", (4, 6)),
        )
        self.transformer.eval()
        self.transformer.requires_grad_(False)

        self.train_scheduler_latent = FlowMatchScheduler(
            shift=config.snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.train_scheduler_action = FlowMatchScheduler(
            shift=config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.train_scheduler_tactile = FlowMatchScheduler(
            shift=config.tactile_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)
        self.train_scheduler_tactile.set_timesteps(1000, training=True)

    @torch.no_grad()
    def _crop_batch(self, batch):
        keep_frames = int(getattr(self.config, "train_frame_chunk_size", 0) or 0)
        if keep_frames <= 0:
            return batch

        latent_frames = batch["latents"].shape[2]
        if latent_frames <= keep_frames:
            return batch

        start = torch.randint(0, latent_frames - keep_frames + 1, (1,)).item()
        end = start + keep_frames
        tactile_per_frame = int(
            getattr(self.config, "tactile_per_frame", self.config.action_per_frame)
        )
        tactile_start = start * tactile_per_frame
        tactile_end = end * tactile_per_frame

        out = dict(batch)
        out["latents"] = batch["latents"][:, :, start:end].contiguous()
        out["actions"] = batch["actions"][:, :, start:end].contiguous()
        out["actions_mask"] = batch["actions_mask"][:, :, start:end].contiguous()
        out["tactile"] = batch["tactile"][:, :, tactile_start:tactile_end].contiguous()
        out["tactile_mask"] = batch["tactile_mask"][:, :, tactile_start:tactile_end].contiguous()
        return out

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, mask=None, action_mode=False, noisy_cond_prob=0.0):
        batch_size, _, frames, _, _ = latent.shape
        timestep_ids = sample_timestep_id(
            batch_size=frames,
            num_train_timesteps=train_scheduler.num_train_timesteps,
        )
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents = train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets = train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1

        grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,
            latent.shape[-2] // patch_h,
            latent.shape[-1] // patch_w,
            t=1 if action_mode else 0,
            f_w=1,
            f_shift=0,
            action=action_mode,
        ).to(self.device)
        grid_id = grid_id[None].repeat(batch_size, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                batch_size=frames,
                min_timestep_bd=0.5,
                max_timestep_bd=1.0,
                num_train_timesteps=train_scheduler.num_train_timesteps,
            )
            cond_noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, cond_noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if mask is not None:
            mask_float = mask.float()
            noisy_latents *= mask_float
            targets *= mask_float
            latent *= mask_float

        return {
            "timesteps": timesteps[None].repeat(batch_size, 1),
            "noisy_latents": noisy_latents,
            "targets": targets,
            "latent": latent,
            "cond_timesteps": cond_timesteps[None].repeat(batch_size, 1),
            "grid_id": grid_id,
        }

    @torch.no_grad()
    def _add_tactile_noise(self, tactile, tactile_mask):
        out = self._add_noise(
            latent=tactile,
            train_scheduler=self.train_scheduler_tactile,
            mask=tactile_mask,
            action_mode=True,
            noisy_cond_prob=getattr(self.config, "tactile_noisy_cond_prob", 0.5),
        )
        batch_size = tactile.shape[0]
        token_grid = getattr(self.config, "tactile_token_grid", (4, 6))
        out["grid_id"] = get_mesh_id(
            tactile.shape[-3],
            token_grid[0],
            token_grid[1],
            t=2,
            f_w=1,
            f_shift=0,
            action=False,
        ).to(self.device)[None].repeat(batch_size, 1, 1)
        out["tactile_mask"] = tactile_mask
        return out

    @torch.no_grad()
    def prepare_input_dict(self, batch):
        batch = self._crop_batch(batch)
        latent_dict = self._add_noise(
            latent=batch["latents"],
            train_scheduler=self.train_scheduler_latent,
            mask=None,
            action_mode=False,
            noisy_cond_prob=0.5,
        )
        action_dict = self._add_noise(
            latent=batch["actions"],
            train_scheduler=self.train_scheduler_action,
            mask=batch["actions_mask"],
            action_mode=True,
            noisy_cond_prob=0.0,
        )
        tactile_dict = self._add_tactile_noise(batch["tactile"], batch["tactile_mask"])

        latent_dict["text_emb"] = batch["text_emb"]
        action_dict["text_emb"] = batch["text_emb"]
        tactile_dict["text_emb"] = batch["text_emb"]
        action_dict["actions_mask"] = batch["actions_mask"]

        return {
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "tactile_dict": tactile_dict,
            "chunk_size": torch.randint(1, 5, (1,)).item(),
            "window_size": torch.randint(4, 65, (1,)).item(),
        }

    def _frame_loss(self, pred, target, frame_weight, mask=None):
        loss = F.mse_loss(pred.float(), target.float().detach(), reduction="none")
        loss = loss * frame_weight[:, None, :, None, None]
        if mask is None:
            mask = torch.ones_like(loss, dtype=torch.bool)
        loss = loss * mask.float()
        loss = loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        mask = mask.float().permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        return (loss.sum(dim=1) / (mask.sum(dim=1) + 1e-6)).mean()

    def _temporal_tactile_loss(self, pred, target):
        weight = getattr(self.config, "tactile_temporal_loss_weight", 0.0)
        if weight <= 0 or pred.shape[2] < 2:
            return pred.new_tensor(0.0)
        pred_dt = pred[:, :, 1:] - pred[:, :, :-1]
        target_dt = target[:, :, 1:] - target[:, :, :-1]
        return F.l1_loss(pred_dt.float(), target_dt.float().detach()) * weight

    def _contact_tactile_loss(self, pred, target):
        weight = getattr(self.config, "tactile_contact_loss_weight", 0.0)
        if weight <= 0:
            return pred.new_tensor(0.0)
        threshold = getattr(self.config, "tactile_contact_threshold", 0.05)
        target_contact = (target.float() > threshold).float()
        pred_contact_logit = pred.float() * 4.0
        return F.binary_cross_entropy_with_logits(pred_contact_logit, target_contact) * weight

    def compute_loss(self, input_dict, pred):
        latent_pred, action_pred, tactile_pred = pred
        latent_targets = input_dict["latent_dict"]["targets"]
        action_targets = input_dict["action_dict"]["targets"]
        tactile_targets = input_dict["tactile_dict"]["targets"]

        action_pred = rearrange(
            action_pred,
            "b (f n) c -> b c f n 1",
            f=action_targets.shape[-3],
        )
        latent_pred = data_seq_to_patch(
            self.patch_size,
            latent_pred,
            latent_targets.shape[-3],
            latent_targets.shape[-2],
            latent_targets.shape[-1],
            batch_size=latent_pred.shape[0],
        )

        latent_weight = self.train_scheduler_latent.training_weight(
            input_dict["latent_dict"]["timesteps"].flatten()
        ).reshape(input_dict["latent_dict"]["timesteps"].shape)
        action_weight = self.train_scheduler_action.training_weight(
            input_dict["action_dict"]["timesteps"].flatten()
        ).reshape(input_dict["action_dict"]["timesteps"].shape)
        tactile_weight = self.train_scheduler_tactile.training_weight(
            input_dict["tactile_dict"]["timesteps"].flatten()
        ).reshape(input_dict["tactile_dict"]["timesteps"].shape)

        latent_loss = self._frame_loss(latent_pred, latent_targets, latent_weight)
        action_loss = self._frame_loss(
            action_pred,
            action_targets,
            action_weight,
            input_dict["action_dict"]["actions_mask"],
        )
        tactile_mse_loss = self._frame_loss(
            tactile_pred,
            tactile_targets,
            tactile_weight,
            input_dict["tactile_dict"]["tactile_mask"],
        )
        temporal_loss = self._temporal_tactile_loss(tactile_pred, tactile_targets)
        contact_loss = self._contact_tactile_loss(tactile_pred, tactile_targets)
        tactile_total_loss = (
            tactile_mse_loss * getattr(self.config, "tactile_loss_weight", 1.0)
            + temporal_loss
            + contact_loss
        )

        return {
            "video_loss": latent_loss.detach(),
            "action_loss": action_loss.detach(),
            "tactile_mse_loss": tactile_mse_loss.detach(),
            "tactile_temporal_loss": temporal_loss.detach(),
            "tactile_contact_loss": contact_loss.detach(),
            "tactile_loss": tactile_total_loss.detach(),
            "total_loss": (latent_loss + action_loss + tactile_total_loss).detach(),
        }

    def _sigmas_for_timesteps(self, scheduler, timesteps, sample):
        timestep_flat = timesteps.flatten()
        scheduler_timesteps = scheduler.timesteps.to(device=timestep_flat.device)
        timestep_ids = torch.argmin(
            (scheduler_timesteps[:, None] - timestep_flat[None]).abs(),
            dim=0,
        )
        sigmas = scheduler.sigmas.to(device=sample.device, dtype=sample.dtype)[timestep_ids]
        return sigmas.reshape(timesteps.shape).view(timesteps.shape[0], 1, timesteps.shape[1], 1, 1)

    def _denormalize_action(self, action_norm):
        q01 = torch.tensor(
            self.config.norm_stat["q01"],
            device=action_norm.device,
            dtype=action_norm.dtype,
        ).view(1, -1, 1, 1, 1)
        q99 = torch.tensor(
            self.config.norm_stat["q99"],
            device=action_norm.device,
            dtype=action_norm.dtype,
        ).view(1, -1, 1, 1, 1)
        action_raw = (action_norm + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        return action_raw

    def _denormalize_tactile(self, tactile_norm):
        batch_size, channels, frames, height, width = tactile_norm.shape
        flat = tactile_norm.permute(0, 2, 1, 3, 4).reshape(batch_size, frames, -1)
        q01 = torch.tensor(
            self.config.tactile_norm_stat["q01"],
            device=tactile_norm.device,
            dtype=tactile_norm.dtype,
        ).view(1, 1, -1)
        q99 = torch.tensor(
            self.config.tactile_norm_stat["q99"],
            device=tactile_norm.device,
            dtype=tactile_norm.dtype,
        ).view(1, 1, -1)
        flat = (flat + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        return flat.reshape(batch_size, frames, channels, height, width).permute(0, 2, 1, 3, 4)

    def reconstruct_predictions(self, input_dict, pred):
        """Convert denoising targets into one-step clean-sample estimates."""
        latent_pred, action_pred, tactile_pred = pred
        latent_targets = input_dict["latent_dict"]["targets"]
        action_targets = input_dict["action_dict"]["targets"]

        action_pred = rearrange(
            action_pred,
            "b (f n) c -> b c f n 1",
            f=action_targets.shape[-3],
        )
        latent_pred = data_seq_to_patch(
            self.patch_size,
            latent_pred,
            latent_targets.shape[-3],
            latent_targets.shape[-2],
            latent_targets.shape[-1],
            batch_size=latent_pred.shape[0],
        )

        latent_sigma = self._sigmas_for_timesteps(
            self.train_scheduler_latent,
            input_dict["latent_dict"]["timesteps"],
            input_dict["latent_dict"]["noisy_latents"],
        )
        action_sigma = self._sigmas_for_timesteps(
            self.train_scheduler_action,
            input_dict["action_dict"]["timesteps"],
            input_dict["action_dict"]["noisy_latents"],
        )
        tactile_sigma = self._sigmas_for_timesteps(
            self.train_scheduler_tactile,
            input_dict["tactile_dict"]["timesteps"],
            input_dict["tactile_dict"]["noisy_latents"],
        )

        video_x0 = input_dict["latent_dict"]["noisy_latents"] - latent_sigma * latent_pred
        video_target_x0 = input_dict["latent_dict"]["noisy_latents"] - latent_sigma * latent_targets
        action_x0 = input_dict["action_dict"]["noisy_latents"] - action_sigma * action_pred
        action_target_x0 = input_dict["action_dict"]["noisy_latents"] - action_sigma * action_targets
        tactile_x0 = input_dict["tactile_dict"]["noisy_latents"] - tactile_sigma * tactile_pred
        tactile_target_x0 = input_dict["tactile_dict"]["noisy_latents"] - tactile_sigma * input_dict["tactile_dict"]["targets"]

        action_mask = input_dict["action_dict"]["actions_mask"]
        tactile_mask = input_dict["tactile_dict"]["tactile_mask"]
        action_x0 = action_x0 * action_mask.float()
        action_target_x0 = action_target_x0 * action_mask.float()
        tactile_x0 = tactile_x0 * tactile_mask.float()
        tactile_target_x0 = tactile_target_x0 * tactile_mask.float()

        action_raw = self._denormalize_action(action_x0)
        action_target_raw = self._denormalize_action(action_target_x0)
        used_ids = list(self.config.used_action_channel_ids)

        tactile_raw = self._denormalize_tactile(tactile_x0)
        tactile_target_raw = self._denormalize_tactile(tactile_target_x0)

        return {
            "video_pred_latent": video_x0,
            "video_target_latent": video_target_x0,
            "video_noisy_latent": input_dict["latent_dict"]["noisy_latents"],
            "action_pred_norm_full30": action_x0,
            "action_target_norm_full30": action_target_x0,
            "action_mask_full30": action_mask,
            "action_pred_raw_used": action_raw[:, used_ids, :, :, 0],
            "action_target_raw_used": action_target_raw[:, used_ids, :, :, 0],
            "tactile_pred_norm": tactile_x0,
            "tactile_target_norm": tactile_target_x0,
            "tactile_mask": tactile_mask,
            "tactile_pred_raw": tactile_raw,
            "tactile_target_raw": tactile_target_raw,
            "timesteps": {
                "video": input_dict["latent_dict"]["timesteps"],
                "action": input_dict["action_dict"]["timesteps"],
                "tactile": input_dict["tactile_dict"]["timesteps"],
            },
        }

    def dump_predictions(self, output_dir, batch_idx, input_dict, pred, losses):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions = self.reconstruct_predictions(input_dict, pred)
        cpu_payload = {}
        for key, value in predictions.items():
            if isinstance(value, dict):
                cpu_payload[key] = {
                    sub_key: sub_value.detach().cpu()
                    for sub_key, sub_value in value.items()
                }
            elif torch.is_tensor(value):
                cpu_payload[key] = value.detach().cpu()
            else:
                cpu_payload[key] = value
        cpu_payload["losses"] = {
            key: float(value.detach().cpu())
            for key, value in losses.items()
        }
        cpu_payload["meta"] = {
            "batch_idx": batch_idx,
            "used_action_channel_ids": list(self.config.used_action_channel_ids),
            "note": (
                "video_pred_latent/action_pred/tactile_pred are one-step x0 estimates "
                "from validation denoising, not full closed-loop rollout samples."
            ),
        }
        torch.save(cpu_payload, output_dir / f"prediction_batch_{batch_idx:06d}.pt")

    @torch.no_grad()
    def validate_batch(self, batch, batch_idx=None, prediction_output_dir=None):
        batch = _move_batch_to_device(batch, self.device)
        input_dict = self.prepare_input_dict(batch)
        output = self.transformer(input_dict, train_mode=True)
        losses = self.compute_loss(input_dict, output)
        if prediction_output_dir is not None:
            self.dump_predictions(
                prediction_output_dir,
                batch_idx,
                input_dict,
                output,
                losses,
            )
        return losses


def run(args):
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is not available; pass --device cpu to run on CPU.")
    if args.save_prediction_batches > 0 and not args.prediction_output_dir:
        raise ValueError("--prediction-output-dir is required when --save-prediction-batches > 0")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = apply_stats_json(TACTILE_CONFIGS[args.config_name])
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
        config.empty_emb_path = os.path.join(args.dataset_path, "empty_emb.pt")
    if args.model_path is not None:
        config.wan22_pretrained_model_name_or_path = args.model_path
    if args.stats_json_path is not None:
        config.stats_json_path = args.stats_json_path
        config = apply_stats_json(config)
    if args.train_frame_chunk_size is not None:
        config.train_frame_chunk_size = args.train_frame_chunk_size
    if args.load_worker is not None:
        config.load_worker = args.load_worker
    if args.dataset_init_worker is not None:
        config.dataset_init_worker = args.dataset_init_worker

    dtype = _dtype_from_name(args.dtype)
    validator = OfflineTactileValidator(
        config=config,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        dtype=dtype,
        attn_mode=args.attn_mode,
    )

    logger.info("Loading tactile LeRobot validation dataset...")
    dataset = MultiTactileLatentLeRobotDataset(
        config=config,
        num_init_worker=getattr(config, "dataset_init_worker", 1),
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=getattr(config, "load_worker", 0),
        pin_memory=args.device.startswith("cuda"),
    )

    max_batches = None if args.max_batches <= 0 else args.max_batches
    iter_loader = data_loader if max_batches is None else islice(data_loader, max_batches)
    progress_total = None if max_batches is None else min(max_batches, len(data_loader))
    totals = {}
    num_batches = 0
    progress = tqdm(iter_loader, total=progress_total, desc="Offline validation", dynamic_ncols=True)

    for batch_idx, batch in enumerate(progress):
        # Make validation repeatable even though timesteps/crops are sampled.
        torch.manual_seed(args.seed + batch_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + batch_idx)

        prediction_output_dir = (
            args.prediction_output_dir
            if args.save_prediction_batches > 0 and batch_idx < args.save_prediction_batches
            else None
        )
        losses = validator.validate_batch(
            batch,
            batch_idx=batch_idx,
            prediction_output_dir=prediction_output_dir,
        )
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.cpu())
        num_batches += 1

        progress.set_postfix(
            {
                "video": f"{float(losses['video_loss'].cpu()):.4f}",
                "action": f"{float(losses['action_loss'].cpu()):.4f}",
                "tactile": f"{float(losses['tactile_loss'].cpu()):.4f}",
            }
        )

        if args.gc_interval > 0 and num_batches % args.gc_interval == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    if num_batches == 0:
        raise RuntimeError("No validation batches were processed.")

    metrics = {key: value / num_batches for key, value in totals.items()}
    metrics["num_batches"] = num_batches
    metrics["num_segments"] = len(dataset)
    metrics["dataset_path"] = config.dataset_path
    metrics["checkpoint_path"] = str(
        _resolve_transformer_path(args.checkpoint_path, config.wan22_pretrained_model_name_or_path)
    )
    metrics["train_frame_chunk_size"] = int(getattr(config, "train_frame_chunk_size", 0) or 0)
    metrics["attn_mode"] = args.attn_mode

    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        logger.info(f"Saved validation metrics to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Offline validation for LingBot-VA-Tactile")
    parser.add_argument("--config-name", type=str, default="robotwin_tactile_train")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Checkpoint root or transformer dir")
    parser.add_argument("--model-path", type=str, default="/data/lingbot-va-models/lingbot-va-base")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--stats-json-path", type=str, default="lingbot-va-tactile/tactile_stats.json")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--prediction-output-dir", type=str, default=None)
    parser.add_argument(
        "--save-prediction-batches",
        type=int,
        default=0,
        help="Save one-step x0 video/action/tactile predictions for the first N batches",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=50, help="0 means full dataset")
    parser.add_argument("--train-frame-chunk-size", type=int, default=4)
    parser.add_argument("--load-worker", type=int, default=0)
    parser.add_argument("--dataset-init-worker", type=int, default=1)
    parser.add_argument("--gc-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn-mode", type=str, default="flex", choices=["flex", "torch"])
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
