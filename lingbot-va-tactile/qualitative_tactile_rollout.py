#!/usr/bin/env python3
"""Generate qualitative SkinWorld multimodal future rollout figures offline.

Unlike ``validate_tactile_offline.py``, this script does not plot independently
corrupted one-step denoising estimates. It caches an observed multimodal prefix
from one dataset segment, optionally samples predicted future video latents, and
then iteratively samples contiguous future tactile and action trajectories from
noise in deployment order.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WAN_VA_ROOT = REPO_ROOT / "wan_va"

for path in (str(WAN_VA_ROOT), str(REPO_ROOT), str(SCRIPT_DIR)):
    while path in sys.path:
        sys.path.remove(path)
sys.path[:0] = [str(WAN_VA_ROOT), str(REPO_ROOT), str(SCRIPT_DIR)]

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from tactile_va.configs import TACTILE_CONFIGS
from tactile_va.configs.stats import apply_stats_json
from tactile_va.dataset import MultiTactileLatentLeRobotDataset
from tactile_va.modules import load_tactile_transformer
from wan_va.utils import FlowMatchScheduler, data_seq_to_patch, get_mesh_id, init_logger, logger


ACTION_NAMES_8D = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]


def _resolve_transformer_path(checkpoint_path: str | None, model_path: str) -> Path:
    path = Path(checkpoint_path or model_path).expanduser().resolve()
    return path / "transformer" if (path / "transformer").is_dir() else path


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping[name]


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _select_frame_ids(num_frames: int, num_plot_frames: int) -> np.ndarray:
    if num_plot_frames <= 0 or num_plot_frames >= num_frames:
        return np.arange(num_frames, dtype=int)
    return np.linspace(0, num_frames - 1, num_plot_frames, dtype=int)


class TactileFutureRollout:
    def __init__(self, config, checkpoint_path, device, dtype, predict_video, predict_actions):
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.predict_video = predict_video
        self.predict_actions = predict_actions
        self.cache_name = "offline_tactile_rollout"
        self.patch_size = tuple(config.patch_size)
        self.video_decode_vae = None
        self.video_decode_processor = None
        self.video_decode_device = None

        transformer_path = _resolve_transformer_path(
            checkpoint_path,
            config.wan22_pretrained_model_name_or_path,
        )
        logger.info(f"Loading tactile transformer from {transformer_path}")
        self.transformer = load_tactile_transformer(
            str(transformer_path),
            torch_dtype=dtype,
            torch_device=self.device,
            attn_mode="torch",
            tactile_dim=config.tactile_dim,
            tactile_shape=tuple(config.tactile_shape),
            tactile_token_grid=tuple(config.tactile_token_grid),
        )
        self.transformer.eval()
        self.transformer.requires_grad_(False)

        self.video_scheduler = FlowMatchScheduler(
            shift=config.snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.tactile_scheduler = FlowMatchScheduler(
            shift=config.tactile_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.action_scheduler = FlowMatchScheduler(
            shift=config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )

    def _video_input(self, latents, text_emb, frame_start, timestep=0.0):
        patch_f, patch_h, patch_w = self.patch_size
        return {
            "noisy_latents": latents,
            "timesteps": torch.full(
                (latents.shape[0], latents.shape[2]),
                float(timestep),
                device=self.device,
                dtype=torch.float32,
            ),
            "grid_id": get_mesh_id(
                latents.shape[2] // patch_f,
                latents.shape[3] // patch_h,
                latents.shape[4] // patch_w,
                t=0,
                f_w=1,
                f_shift=frame_start,
                action=False,
            ).to(self.device)[None].repeat(latents.shape[0], 1, 1),
            "text_emb": text_emb,
        }

    def _action_input(self, actions, text_emb, frame_start, timestep=0.0):
        return {
            "noisy_latents": actions,
            "timesteps": torch.full(
                (actions.shape[0], actions.shape[2]),
                float(timestep),
                device=self.device,
                dtype=torch.float32,
            ),
            "grid_id": get_mesh_id(
                actions.shape[2],
                actions.shape[3],
                actions.shape[4],
                t=1,
                f_w=1,
                f_shift=frame_start,
                action=True,
            ).to(self.device)[None].repeat(actions.shape[0], 1, 1),
            "text_emb": text_emb,
        }

    def _tactile_input(self, tactile, text_emb, frame_start, timestep=0.0):
        grid_h, grid_w = tuple(self.config.tactile_token_grid)
        return {
            "noisy_latents": tactile,
            "timesteps": torch.full(
                (tactile.shape[0], tactile.shape[2]),
                float(timestep),
                device=self.device,
                dtype=torch.float32,
            ),
            "grid_id": get_mesh_id(
                tactile.shape[2],
                grid_h,
                grid_w,
                t=2,
                f_w=1,
                f_shift=frame_start,
                action=False,
            ).to(self.device)[None].repeat(tactile.shape[0], 1, 1),
            "text_emb": text_emb,
        }

    def _initialize_cache(self, latents, actions, tactile, context_frames, future_frames):
        patch_f, patch_h, patch_w = self.patch_size
        tactile_per_frame = int(self.config.tactile_per_frame)
        grid_h, grid_w = tuple(self.config.tactile_token_grid)
        context_video_tokens = (
            (context_frames // patch_f)
            * (latents.shape[3] // patch_h)
            * (latents.shape[4] // patch_w)
        )
        future_video_tokens = (
            (future_frames // patch_f)
            * (latents.shape[3] // patch_h)
            * (latents.shape[4] // patch_w)
        )
        context_action_tokens = context_frames * int(self.config.action_per_frame)
        future_action_tokens = future_frames * int(self.config.action_per_frame)
        tactile_tokens = (
            (context_frames + future_frames)
            * tactile_per_frame
            * grid_h
            * grid_w
        )
        self.transformer.clear_cache(self.cache_name)
        self.transformer.create_empty_cache(
            self.cache_name,
            attn_window=2,
            latent_token_per_chunk=(
                context_video_tokens + future_video_tokens
                if self.predict_video
                else context_video_tokens
            ),
            action_token_per_chunk=(
                context_action_tokens + future_action_tokens
                if self.predict_actions
                else context_action_tokens
            ),
            tactile_token_per_chunk=tactile_tokens,
            device=self.device,
            dtype=self.dtype,
            batch_size=latents.shape[0],
        )

    @torch.no_grad()
    def _cache_observed_history(self, batch, context_frames):
        tactile_per_frame = int(self.config.tactile_per_frame)
        text_emb = batch["text_emb"]
        context_video = batch["latents"][:, :, :context_frames].to(self.dtype)
        context_actions = batch["actions"][:, :, :context_frames].to(self.dtype)
        context_tactile = batch["tactile"][
            :, :, : context_frames * tactile_per_frame
        ].to(self.dtype)

        self.transformer(
            self._video_input(context_video, text_emb, frame_start=0),
            update_cache=2,
            cache_name=self.cache_name,
            action_mode=False,
        )
        self.transformer(
            self._action_input(context_actions, text_emb, frame_start=0),
            update_cache=2,
            cache_name=self.cache_name,
            action_mode=True,
        )
        self.transformer(
            self._tactile_input(context_tactile, text_emb, frame_start=0),
            update_cache=2,
            cache_name=self.cache_name,
            tactile_mode=True,
        )
        return context_tactile

    @torch.no_grad()
    def _sample_video_future(self, batch, context_frames, future_frames, num_steps):
        target_shape = batch["latents"][:, :, context_frames : context_frames + future_frames].shape
        latents = torch.randn(target_shape, device=self.device, dtype=self.dtype)
        text_emb = batch["text_emb"]
        self.video_scheduler.set_timesteps(num_steps)
        timesteps = torch.nn.functional.pad(self.video_scheduler.timesteps, (0, 1), value=0)
        for index, timestep in enumerate(tqdm(timesteps, desc="video rollout", leave=False)):
            last_step = index == len(timesteps) - 1
            pred = self.transformer(
                self._video_input(latents, text_emb, context_frames, timestep),
                update_cache=1 if last_step else 0,
                cache_name=self.cache_name,
                action_mode=False,
            )
            if not last_step:
                pred = data_seq_to_patch(
                    self.patch_size,
                    pred,
                    future_frames,
                    latents.shape[-2],
                    latents.shape[-1],
                    batch_size=latents.shape[0],
                )
                latents = self.video_scheduler.step(pred, timestep, latents)
        return latents

    @torch.no_grad()
    def _sample_tactile_future(self, batch, context_frames, future_frames, num_steps):
        tactile_per_frame = int(self.config.tactile_per_frame)
        future_tactile_frames = future_frames * tactile_per_frame
        shape = (
            batch["tactile"].shape[0],
            batch["tactile"].shape[1],
            future_tactile_frames,
            batch["tactile"].shape[3],
            batch["tactile"].shape[4],
        )
        tactile = torch.randn(shape, device=self.device, dtype=self.dtype)
        text_emb = batch["text_emb"]
        frame_start = context_frames * tactile_per_frame
        self.tactile_scheduler.set_timesteps(num_steps)
        timesteps = torch.nn.functional.pad(self.tactile_scheduler.timesteps, (0, 1), value=0)
        for index, timestep in enumerate(tqdm(timesteps, desc="tactile rollout", leave=False)):
            last_step = index == len(timesteps) - 1
            pred = self.transformer(
                self._tactile_input(tactile, text_emb, frame_start, timestep),
                update_cache=1 if last_step else 0,
                cache_name=self.cache_name,
                tactile_mode=True,
            )
            if not last_step:
                tactile = self.tactile_scheduler.step(pred, timestep, tactile)
        return tactile

    @torch.no_grad()
    def _sample_action_future(self, batch, context_frames, future_frames, num_steps):
        target_shape = batch["actions"][:, :, context_frames : context_frames + future_frames].shape
        actions = torch.randn(target_shape, device=self.device, dtype=self.dtype)
        text_emb = batch["text_emb"]
        self.action_scheduler.set_timesteps(num_steps)
        timesteps = torch.nn.functional.pad(self.action_scheduler.timesteps, (0, 1), value=0)
        for index, timestep in enumerate(tqdm(timesteps, desc="action rollout", leave=False)):
            last_step = index == len(timesteps) - 1
            pred = self.transformer(
                self._action_input(actions, text_emb, context_frames, timestep),
                update_cache=1 if last_step else 0,
                cache_name=self.cache_name,
                action_mode=True,
            )
            if not last_step:
                pred = rearrange(
                    pred,
                    "b (f n) c -> b c f n 1",
                    f=future_frames,
                )
                actions = self.action_scheduler.step(pred, timestep, actions)
        channel_mask = torch.zeros(
            (1, actions.shape[1], 1, 1, 1),
            device=actions.device,
            dtype=actions.dtype,
        )
        channel_mask[:, list(self.config.used_action_channel_ids)] = 1
        return actions * channel_mask

    def _denormalize_tactile(self, tactile):
        batch, channels, frames, height, width = tactile.shape
        flat = tactile.permute(0, 2, 1, 3, 4).reshape(batch, frames, -1)
        q01 = torch.tensor(
            self.config.tactile_norm_stat["q01"],
            device=tactile.device,
            dtype=tactile.dtype,
        ).view(1, 1, -1)
        q99 = torch.tensor(
            self.config.tactile_norm_stat["q99"],
            device=tactile.device,
            dtype=tactile.dtype,
        ).view(1, 1, -1)
        flat = (flat + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        return flat.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)

    def _denormalize_action(self, actions):
        q01 = torch.tensor(
            self.config.norm_stat["q01"],
            device=actions.device,
            dtype=actions.dtype,
        ).view(1, -1, 1, 1, 1)
        q99 = torch.tensor(
            self.config.norm_stat["q99"],
            device=actions.device,
            dtype=actions.dtype,
        ).view(1, -1, 1, 1, 1)
        actions = (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        return actions[:, list(self.config.used_action_channel_ids), :, :, 0]

    @torch.no_grad()
    def decode_video_sequences(self, result, decode_device):
        """Decode normalized Wan latents for qualitative video inspection."""
        if result["video_pred_latent"] is None:
            return {}
        decode_device = torch.device(decode_device)
        if self.video_decode_vae is None or self.video_decode_device != decode_device:
            from diffusers.video_processor import VideoProcessor
            from wan_va.modules.utils import load_vae

            decode_dtype = self.dtype if decode_device.type == "cuda" else torch.float32
            vae_path = Path(self.config.wan22_pretrained_model_name_or_path) / "vae"
            logger.info(f"Loading Wan VAE for visual decoding from {vae_path} on {decode_device}")
            self.video_decode_vae = load_vae(
                str(vae_path),
                torch_dtype=decode_dtype,
                torch_device=decode_device,
            )
            self.video_decode_vae.eval()
            self.video_decode_processor = VideoProcessor(vae_scale_factor=1)
            self.video_decode_device = decode_device

        context = result["video_context_latent"]
        sequences = {
            "video_pred_rgb": torch.cat([context, result["video_pred_latent"]], dim=2),
            "video_target_rgb": torch.cat([context, result["video_target_latent"]], dim=2),
        }
        decoded = {}
        vae = self.video_decode_vae
        for name, normalized_latents in sequences.items():
            normalized_latents = normalized_latents.to(device=decode_device, dtype=vae.dtype)
            mean = torch.tensor(
                vae.config.latents_mean,
                device=decode_device,
                dtype=vae.dtype,
            ).view(1, vae.config.z_dim, 1, 1, 1)
            std = torch.tensor(
                vae.config.latents_std,
                device=decode_device,
                dtype=vae.dtype,
            ).view(1, vae.config.z_dim, 1, 1, 1)
            video = vae.decode(normalized_latents * std + mean, return_dict=False)[0]
            frames = self.video_decode_processor.postprocess_video(video, output_type="np")[0]
            decoded[name] = torch.from_numpy(np.asarray(frames))
        return decoded

    @torch.no_grad()
    def predict(
        self,
        batch,
        context_frames,
        future_frames,
        video_steps,
        tactile_steps,
        action_steps,
    ):
        batch = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        batch["text_emb"] = batch["text_emb"].to(self.dtype)
        tactile_per_frame = int(self.config.tactile_per_frame)
        required_frames = context_frames + future_frames
        if batch["latents"].shape[2] < required_frames:
            raise ValueError(
                f"Segment only has {batch['latents'].shape[2]} latent frames; "
                f"need context_frames + future_frames = {required_frames}."
            )

        self._initialize_cache(
            batch["latents"],
            batch["actions"],
            batch["tactile"],
            context_frames,
            future_frames,
        )
        context_tactile = self._cache_observed_history(batch, context_frames)
        predicted_video = None
        if self.predict_video:
            predicted_video = self._sample_video_future(
                batch,
                context_frames,
                future_frames,
                video_steps,
            )
        predicted_tactile = self._sample_tactile_future(
            batch,
            context_frames,
            future_frames,
            tactile_steps,
        )
        predicted_actions = None
        if self.predict_actions:
            predicted_actions = self._sample_action_future(
                batch,
                context_frames,
                future_frames,
                action_steps,
            )
        start = context_frames * tactile_per_frame
        end = (context_frames + future_frames) * tactile_per_frame
        target_tactile = batch["tactile"][:, :, start:end]
        target_actions = batch["actions"][:, :, context_frames : context_frames + future_frames]
        target_action_mask = batch["actions_mask"][
            :, list(self.config.used_action_channel_ids), context_frames : context_frames + future_frames, :, 0
        ]
        return {
            "tactile_context_norm": context_tactile,
            "tactile_pred_norm": predicted_tactile,
            "tactile_target_norm": target_tactile,
            "tactile_context_raw": self._denormalize_tactile(context_tactile),
            "tactile_pred_raw": self._denormalize_tactile(predicted_tactile),
            "tactile_target_raw": self._denormalize_tactile(target_tactile),
            "action_pred_norm": predicted_actions,
            "action_target_norm": target_actions,
            "action_pred_raw_used": (
                self._denormalize_action(predicted_actions)
                if predicted_actions is not None
                else None
            ),
            "action_target_raw_used": self._denormalize_action(target_actions),
            "action_target_mask_used": target_action_mask,
            "video_context_latent": batch["latents"][:, :, :context_frames],
            "video_pred_latent": predicted_video,
            "video_target_latent": batch["latents"][
                :, :, context_frames : context_frames + future_frames
            ],
        }


def _write_summary_csv(pred, target, output_path):
    pred_mean = pred.mean(axis=(0, 2, 3))
    target_mean = target.mean(axis=(0, 2, 3))
    mae = np.abs(pred - target).mean(axis=(0, 2, 3))
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "target_mean_pressure", "pred_mean_pressure", "mae"])
        for frame in range(pred.shape[1]):
            writer.writerow([frame, target_mean[frame], pred_mean[frame], mae[frame]])


def _plot_summary(pred, target, output_path):
    plt = _load_pyplot()
    x = np.arange(pred.shape[1])
    pred_mean = pred.mean(axis=(0, 2, 3))
    target_mean = target.mean(axis=(0, 2, 3))
    mae = np.abs(pred - target).mean(axis=(0, 2, 3))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(x, target_mean, color="#153e6f", label="ground truth", linewidth=1.6)
    axes[0].plot(x, pred_mean, color="#d14a32", label="prediction", linewidth=1.4)
    axes[0].set_title("Mean pressure over predicted future")
    axes[0].legend(frameon=False)
    axes[1].plot(x, mae, color="#bf6b27", linewidth=1.5)
    axes[1].set_title("Frame-wise tactile MAE")
    for axis in axes:
        axis.set_xlabel("future tactile step")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmaps(context, pred, target, output_path, num_frames, title, channel=None):
    plt = _load_pyplot()
    if channel is None:
        context_map = context.mean(axis=0)
        pred_map = pred.mean(axis=0)
        target_map = target.mean(axis=0)
    else:
        context_map = context[channel]
        pred_map = pred[channel]
        target_map = target[channel]
    frame_ids = _select_frame_ids(pred_map.shape[0], num_frames)
    values = np.concatenate([target_map[frame_ids].ravel(), pred_map[frame_ids].ravel()])
    vmin, vmax = np.quantile(values, [0.01, 0.99])
    if vmax <= vmin:
        vmax = vmin + 1e-6
    errors = np.abs(pred_map[frame_ids] - target_map[frame_ids])
    err_max = max(float(np.quantile(errors, 0.99)), 1e-6)

    columns = len(frame_ids) + 1
    fig, axes = plt.subplots(3, columns, figsize=(2.05 * columns, 6.1), squeeze=False)
    context_frame = context_map[-1]
    for row, label in enumerate(["Ground Truth", "Prediction", "Absolute Error"]):
        axes[row, 0].imshow(
            context_frame if row < 2 else np.zeros_like(context_frame),
            cmap="viridis" if row < 2 else "magma",
            vmin=vmin if row < 2 else 0.0,
            vmax=vmax if row < 2 else err_max,
            aspect="auto",
        )
        axes[row, 0].set_ylabel(label)
        axes[row, 0].set_title("context" if row == 0 else "")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
    for column, frame_id in enumerate(frame_ids, start=1):
        images = [
            (target_map[frame_id], "viridis", vmin, vmax),
            (pred_map[frame_id], "viridis", vmin, vmax),
            (np.abs(pred_map[frame_id] - target_map[frame_id]), "magma", 0.0, err_max),
        ]
        for row, (image, cmap, cur_min, cur_max) in enumerate(images):
            axes[row, column].imshow(
                image,
                cmap=cmap,
                vmin=cur_min,
                vmax=cur_max,
                aspect="auto",
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(f"+{frame_id + 1}")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _flatten_actions(actions):
    channels, frames, steps = actions.shape
    return actions.reshape(channels, frames * steps)


def _plot_actions(pred, target, mask, output_path):
    plt = _load_pyplot()
    pred = _flatten_actions(pred)
    target = _flatten_actions(target)
    mask = _flatten_actions(mask.astype(bool))
    names = (
        ACTION_NAMES_8D
        if pred.shape[0] == len(ACTION_NAMES_8D)
        else [f"action_{channel}" for channel in range(pred.shape[0])]
    )
    x = np.arange(pred.shape[1])
    rows = int(np.ceil(len(names) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12.8, 2.45 * rows), squeeze=False)
    for channel, name in enumerate(names):
        axis = axes[channel // 2, channel % 2]
        valid = mask[channel]
        axis.plot(x[valid], target[channel, valid], label="ground truth", color="#153e6f", linewidth=1.55)
        axis.plot(x[valid], pred[channel, valid], label="prediction", color="#d14a32", linewidth=1.35)
        axis.set_title(name)
        axis.grid(alpha=0.25)
        axis.set_xlabel("future action step")
        if channel == 0:
            axis.legend(frameon=False)
    for channel in range(len(names), rows * 2):
        axes[channel // 2, channel % 2].axis("off")
    fig.suptitle("Predicted Future Action over Time", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_action_csv(pred, target, mask, output_path):
    pred = _flatten_actions(pred)
    target = _flatten_actions(target)
    mask = _flatten_actions(mask.astype(bool))
    names = (
        ACTION_NAMES_8D
        if pred.shape[0] == len(ACTION_NAMES_8D)
        else [f"action_{channel}" for channel in range(pred.shape[0])]
    )
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        columns = ["t"]
        for name in names:
            columns.extend([f"pred_{name}", f"target_{name}", f"abs_err_{name}", f"valid_{name}"])
        writer.writerow(columns)
        for step in range(pred.shape[1]):
            row = [step]
            for channel in range(pred.shape[0]):
                row.extend(
                    [
                        pred[channel, step],
                        target[channel, step],
                        abs(pred[channel, step] - target[channel, step]),
                        int(mask[channel, step]),
                    ]
                )
            writer.writerow(row)


def _plot_video_latents(pred, target, output_path):
    plt = _load_pyplot()
    pred_rms = np.sqrt((pred**2).mean(axis=(0, 2, 3)))
    target_rms = np.sqrt((target**2).mean(axis=(0, 2, 3)))
    mae = np.abs(pred - target).mean(axis=(0, 2, 3))
    x = np.arange(pred.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    axes[0].plot(x, target_rms, label="ground truth", color="#153e6f", linewidth=1.6)
    axes[0].plot(x, pred_rms, label="prediction", color="#d14a32", linewidth=1.45)
    axes[0].set_title("Video latent RMS")
    axes[0].legend(frameon=False)
    axes[1].plot(x, mae, color="#bf6b27", linewidth=1.5)
    axes[1].set_title("Video latent MAE")
    for axis in axes:
        axis.set_xlabel("future latent frame")
        axis.grid(alpha=0.25)
    fig.suptitle("Predicted Future Video Latents", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_decoded_video(pred, target, output_path, num_frames):
    plt = _load_pyplot()
    if float(max(pred.max(), target.max())) > 1.5:
        pred = pred / 255.0
        target = target / 255.0
    frame_ids = _select_frame_ids(pred.shape[0], num_frames)
    error = np.abs(pred.astype(np.float32) - target.astype(np.float32)).mean(axis=-1)
    err_max = max(float(np.quantile(error[frame_ids], 0.99)), 1e-6)
    fig, axes = plt.subplots(3, len(frame_ids), figsize=(2.45 * len(frame_ids), 6.7), squeeze=False)
    for column, frame_id in enumerate(frame_ids):
        axes[0, column].imshow(np.clip(target[frame_id], 0.0, 1.0))
        axes[1, column].imshow(np.clip(pred[frame_id], 0.0, 1.0))
        axes[2, column].imshow(error[frame_id], cmap="magma", vmin=0.0, vmax=err_max)
        axes[0, column].set_title(f"frame {frame_id}")
        for row in range(3):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    for row, label in enumerate(["Ground Truth", "Prediction", "RGB Error"]):
        axes[row, 0].set_ylabel(label)
    fig.suptitle("Decoded Visual Rollout (observed context prefix and predicted future)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _to_uint8_video(frames):
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        if float(np.nanmax(frames)) <= 1.5:
            frames = frames * 255.0
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frames)


def _write_video_with_ffmpeg(frames, output_path, fps):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "--save-video-mp4 requires either system ffmpeg on PATH or the "
            "imageio-ffmpeg Python package."
        )

    frames = _to_uint8_video(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB video frames with shape [T, H, W, 3], got {frames.shape}")
    height, width = frames.shape[1:3]
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    proc = subprocess.run(cmd, input=frames.tobytes(), capture_output=True, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed while writing {output_path}: {stderr}")


def _write_decoded_videos(pred, target, output_dir, fps):
    _write_video_with_ffmpeg(pred, output_dir / "video_prediction.mp4", fps)
    _write_video_with_ffmpeg(target, output_dir / "video_ground_truth.mp4", fps)


def _save_outputs(result, output_dir, segment_index, args):
    segment_dir = output_dir / f"segment_{segment_index:06d}"
    segment_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in result.items()
    }
    payload["meta"] = {
        "segment_index": segment_index,
        "context_latent_frames": args.context_latent_frames,
        "future_latent_frames": args.future_latent_frames,
        "future_tactile_frames": args.future_latent_frames * args.tactile_per_frame,
        "video_inference_steps": args.video_inference_steps,
        "tactile_inference_steps": args.tactile_inference_steps,
        "action_inference_steps": args.action_inference_steps,
        "condition_on_predicted_video": args.predict_video_before_tactile,
        "predict_actions": args.predict_actions,
        "decode_video": args.decode_video,
        "note": (
            "This is an iterative multimodal future rollout conditioned on observed "
            "history. When enabled, sampling follows deployment order: predicted "
            "video is cached before tactile, then tactile is cached before action."
        ),
    }
    # Preserve the original filename while extending the payload to all modalities.
    torch.save(payload, segment_dir / "future_tactile_rollout.pt")

    context = payload["tactile_context_raw"][0].float().numpy()
    pred = payload["tactile_pred_raw"][0].float().numpy()
    target = payload["tactile_target_raw"][0].float().numpy()
    _write_summary_csv(pred, target, segment_dir / "future_tactile_summary.csv")
    _plot_summary(pred, target, segment_dir / "future_tactile_summary.png")
    _plot_heatmaps(
        context,
        pred,
        target,
        segment_dir / "future_tactile_heatmaps_combined.png",
        args.num_plot_frames,
        "Predicted Future Touch over Time (mean of two skin sheets)",
    )
    if args.plot_channels:
        for channel in range(pred.shape[0]):
            _plot_heatmaps(
                context,
                pred,
                target,
                segment_dir / f"future_tactile_heatmaps_sheet_{channel}.png",
                args.num_plot_frames,
                f"Predicted Future Touch over Time (skin sheet {channel})",
                channel=channel,
            )
    action_mae = None
    if payload["action_pred_raw_used"] is not None:
        action_pred = payload["action_pred_raw_used"][0].float().numpy()
        action_target = payload["action_target_raw_used"][0].float().numpy()
        action_mask = payload["action_target_mask_used"][0].numpy().astype(bool)
        _plot_actions(action_pred, action_target, action_mask, segment_dir / "future_action_curves.png")
        _write_action_csv(action_pred, action_target, action_mask, segment_dir / "future_action_values.csv")
        valid = action_mask
        action_mae = float(np.abs(action_pred - action_target)[valid].mean()) if valid.any() else None
    video_latent_mae = None
    if payload["video_pred_latent"] is not None:
        video_pred = payload["video_pred_latent"][0].float().numpy()
        video_target = payload["video_target_latent"][0].float().numpy()
        _plot_video_latents(video_pred, video_target, segment_dir / "future_video_latent_summary.png")
        video_latent_mae = float(np.abs(video_pred - video_target).mean())
    if "video_pred_rgb" in payload:
        video_pred_rgb = payload["video_pred_rgb"].float().numpy()
        video_target_rgb = payload["video_target_rgb"].float().numpy()
        _plot_decoded_video(
            video_pred_rgb,
            video_target_rgb,
            segment_dir / "future_video_decoded_frames.png",
            args.num_video_plot_frames,
        )
        if args.save_video_mp4:
            _write_decoded_videos(video_pred_rgb, video_target_rgb, segment_dir, args.video_fps)
    metrics = {
        "segment_index": segment_index,
        "num_future_tactile_frames": int(pred.shape[1]),
        "tactile_raw_mae": float(np.abs(pred - target).mean()),
        "action_raw_mae": action_mae,
        "video_latent_mae": video_latent_mae,
    }
    with (segment_dir / "multimodal_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    with (segment_dir / "future_tactile_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    return segment_dir, metrics


def run(args):
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is not available; pass --device cpu to run on CPU.")
    if args.context_latent_frames <= 0 or args.future_latent_frames <= 0:
        raise ValueError("--context-latent-frames and --future-latent-frames must be positive.")
    if args.segment_index < 0 or args.num_segments <= 0:
        raise ValueError("--segment-index must be non-negative and --num-segments must be positive.")
    if min(args.video_inference_steps, args.tactile_inference_steps, args.action_inference_steps) <= 0:
        raise ValueError("All modality inference-step arguments must be positive.")
    if args.save_video_mp4 and not args.decode_video:
        raise ValueError("--save-video-mp4 requires --decode-video.")
    if args.decode_video and not args.predict_video_before_tactile:
        logger.warning("--decode-video has no generated video to decode when video prediction is disabled.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = apply_stats_json(TACTILE_CONFIGS[args.config_name])
    config.dataset_path = args.dataset_path
    config.empty_emb_path = os.path.join(args.dataset_path, "empty_emb.pt")
    config.wan22_pretrained_model_name_or_path = args.model_path
    config.stats_json_path = args.stats_json_path
    config = apply_stats_json(config)
    config.cfg_prob = 0.0
    args.tactile_per_frame = int(config.tactile_per_frame)

    dtype = _dtype_from_name(args.dtype)
    rollout = TactileFutureRollout(
        config=config,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        dtype=dtype,
        predict_video=args.predict_video_before_tactile,
        predict_actions=args.predict_actions,
    )
    logger.info("Loading tactile LeRobot dataset...")
    dataset = MultiTactileLatentLeRobotDataset(config=config, num_init_worker=args.dataset_init_worker)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.load_worker)
    selected = set(range(args.segment_index, args.segment_index + args.num_segments))
    output_dir = Path(args.output_dir).expanduser()
    summaries = []
    for index, batch in enumerate(loader):
        if index > max(selected):
            break
        if index not in selected:
            continue
        try:
            result = rollout.predict(
                batch,
                context_frames=args.context_latent_frames,
                future_frames=args.future_latent_frames,
                video_steps=args.video_inference_steps,
                tactile_steps=args.tactile_inference_steps,
                action_steps=args.action_inference_steps,
            )
        except ValueError as exc:
            logger.warning(f"Skipping segment {index}: {exc}")
            continue
        if args.decode_video:
            result.update(
                rollout.decode_video_sequences(
                    result,
                    decode_device=args.video_decode_device,
                )
            )
        segment_dir, metrics = _save_outputs(result, output_dir, index, args)
        summaries.append(metrics)
        print(f"[OK] segment={index} -> {segment_dir}")
        rollout.transformer.clear_cache(rollout.cache_name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    if not summaries:
        raise RuntimeError("No qualitative tactile rollouts were generated.")
    with (output_dir / "rollout_summary.json").open("w") as handle:
        json.dump(summaries, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate qualitative, iterative SkinWorld future tactile rollout figures."
    )
    parser.add_argument("--config-name", type=str, default="robotwin_tactile_train")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="/data/lingbot-va-models/lingbot-va-base")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--stats-json-path", type=str, default="lingbot-va-tactile/tactile_stats.json")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--segment-index", type=int, default=0)
    parser.add_argument("--num-segments", type=int, default=1)
    parser.add_argument("--context-latent-frames", type=int, default=1)
    parser.add_argument("--future-latent-frames", type=int, default=4)
    parser.add_argument("--video-inference-steps", type=int, default=5)
    parser.add_argument("--tactile-inference-steps", type=int, default=10)
    parser.add_argument("--action-inference-steps", type=int, default=10)
    parser.add_argument(
        "--predict-video-before-tactile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample and cache future video latents before tactile generation, matching server order.",
    )
    parser.add_argument(
        "--predict-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample future actions after tactile prediction and save action comparison figures.",
    )
    parser.add_argument("--num-plot-frames", type=int, default=8, help="<=0 plots every future tactile frame.")
    parser.add_argument("--plot-channels", action="store_true")
    parser.add_argument(
        "--decode-video",
        action="store_true",
        help="Decode video latents with the base Wan VAE and write RGB comparison figures.",
    )
    parser.add_argument(
        "--video-decode-device",
        type=str,
        default="cpu",
        help="Device used only for optional Wan VAE decoding, e.g. cpu or cuda:0.",
    )
    parser.add_argument("--num-video-plot-frames", type=int, default=8)
    parser.add_argument(
        "--save-video-mp4",
        action="store_true",
        help="With --decode-video, also save predicted and ground-truth MP4 videos.",
    )
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--dataset-init-worker", type=int, default=1)
    parser.add_argument("--load-worker", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
