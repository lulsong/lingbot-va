#!/usr/bin/env python3
"""Detailed one-step multimodal rollout diagnostics.

This script crops one dataset segment to two latent frames: the observed
context frame ``t`` and the target future frame ``t+1``. It then samples a
single future video latent, tactile block, and action block from noise using
the same rollout helper as ``qualitative_tactile_rollout.py``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from qualitative_tactile_rollout import (
    TactileFutureRollout,
    _dtype_from_name,
    _load_pyplot,
    _write_decoded_videos,
)
from tactile_va.configs import TACTILE_CONFIGS
from tactile_va.configs.stats import apply_stats_json
from tactile_va.dataset import MultiTactileLatentLeRobotDataset
from wan_va.utils import logger


ACTION_NAMES_8D = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "gripper"]


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value)


def _crop_single_step_batch(batch: dict, latent_index: int, tactile_per_frame: int) -> dict:
    """Keep latent frame ``t`` as context and ``t+1`` as target."""
    if latent_index < 0:
        raise ValueError("--latent-index must be non-negative.")
    total_latent_frames = int(batch["latents"].shape[2])
    if latent_index + 1 >= total_latent_frames:
        raise ValueError(
            f"latent-index={latent_index} needs t+1, but segment only has "
            f"{total_latent_frames} latent frames."
        )

    start = latent_index
    end = latent_index + 2
    tactile_start = start * tactile_per_frame
    tactile_end = end * tactile_per_frame

    cropped = {}
    for key, value in batch.items():
        if key == "latents":
            cropped[key] = value[:, :, start:end].contiguous()
        elif key in {"actions", "actions_mask"}:
            cropped[key] = value[:, :, start:end].contiguous()
        elif key in {"tactile", "tactile_mask"}:
            cropped[key] = value[:, :, tactile_start:tactile_end].contiguous()
        else:
            cropped[key] = value
    return cropped


def _latest_context_action_raw(rollout: TactileFutureRollout, batch: dict) -> torch.Tensor:
    context = batch["actions"][:, :, :1].to(rollout.device)
    return rollout._denormalize_action(context)


def _plot_video_frames(pred_rgb: np.ndarray, target_rgb: np.ndarray, output_path: Path):
    """Plot context, one-step GT/prediction, and motion/error maps."""
    plt = _load_pyplot()
    if float(max(pred_rgb.max(), target_rgb.max())) > 1.5:
        pred_rgb = pred_rgb / 255.0
        target_rgb = target_rgb / 255.0
    pred_rgb = np.clip(pred_rgb.astype(np.float32), 0.0, 1.0)
    target_rgb = np.clip(target_rgb.astype(np.float32), 0.0, 1.0)

    context = target_rgb[0]
    target = target_rgb[1]
    pred = pred_rgb[1]
    target_motion = np.abs(target - context).mean(axis=-1)
    pred_motion = np.abs(pred - context).mean(axis=-1)
    error = np.abs(pred - target).mean(axis=-1)
    scale = max(
        float(np.quantile(target_motion, 0.99)),
        float(np.quantile(pred_motion, 0.99)),
        float(np.quantile(error, 0.99)),
        1e-6,
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), squeeze=False)
    panels = [
        (context, "context t", None),
        (target, "target t+1", None),
        (pred, "prediction t+1", None),
        (target_motion, "|target-context|", "magma"),
        (pred_motion, "|prediction-context|", "magma"),
        (error, "|prediction-target|", "magma"),
    ]
    for axis, (image, title, cmap) in zip(axes.ravel(), panels):
        if cmap is None:
            axis.imshow(image)
        else:
            axis.imshow(image, cmap=cmap, vmin=0.0, vmax=scale)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Single-Step Decoded Video Diagnostic", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_video_latent_maps(context: np.ndarray, pred: np.ndarray, target: np.ndarray, output_path: Path):
    plt = _load_pyplot()
    context_map = np.sqrt((context[:, 0] ** 2).mean(axis=0))
    pred_map = np.sqrt((pred[:, 0] ** 2).mean(axis=0))
    target_map = np.sqrt((target[:, 0] ** 2).mean(axis=0))
    target_motion = np.abs(target[:, 0] - context[:, 0]).mean(axis=0)
    pred_motion = np.abs(pred[:, 0] - context[:, 0]).mean(axis=0)
    error = np.abs(pred[:, 0] - target[:, 0]).mean(axis=0)

    value_max = max(
        float(np.quantile(np.concatenate([context_map.ravel(), pred_map.ravel(), target_map.ravel()]), 0.99)),
        1e-6,
    )
    diff_max = max(
        float(np.quantile(np.concatenate([target_motion.ravel(), pred_motion.ravel(), error.ravel()]), 0.99)),
        1e-6,
    )
    panels = [
        (context_map, "context latent RMS", "viridis", 0.0, value_max),
        (target_map, "target latent RMS", "viridis", 0.0, value_max),
        (pred_map, "prediction latent RMS", "viridis", 0.0, value_max),
        (target_motion, "|target-context| latent", "magma", 0.0, diff_max),
        (pred_motion, "|prediction-context| latent", "magma", 0.0, diff_max),
        (error, "|prediction-target| latent", "magma", 0.0, diff_max),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7), squeeze=False)
    for axis, (image, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Single-Step Video Latent Diagnostic", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _video_latent_summary_maps(context: np.ndarray, pred: np.ndarray, target: np.ndarray):
    context_map = np.sqrt((context[:, 0] ** 2).mean(axis=0))
    target_map = np.sqrt((target[:, 0] ** 2).mean(axis=0))
    pred_map = np.sqrt((pred[:, 0] ** 2).mean(axis=0))
    error = np.abs(pred[:, 0] - target[:, 0]).mean(axis=0)
    return context_map, target_map, pred_map, error


def _plot_video_latent_grid(records: list[dict], output_path: Path):
    if not records:
        return
    plt = _load_pyplot()
    rows = len(records)
    fig, axes = plt.subplots(rows, 4, figsize=(9.5, max(2.0 * rows, 3.0)), squeeze=False)
    all_values = []
    all_errors = []
    for record in records:
        context_map, target_map, pred_map, error = _video_latent_summary_maps(
            record["video_context_latent"],
            record["video_pred_latent"],
            record["video_target_latent"],
        )
        all_values.extend([context_map.ravel(), target_map.ravel(), pred_map.ravel()])
        all_errors.append(error.ravel())
    value_max = max(float(np.quantile(np.concatenate(all_values), 0.99)), 1e-6)
    error_max = max(float(np.quantile(np.concatenate(all_errors), 0.99)), 1e-6)

    for row, record in enumerate(records):
        context_map, target_map, pred_map, error = _video_latent_summary_maps(
            record["video_context_latent"],
            record["video_pred_latent"],
            record["video_target_latent"],
        )
        panels = [
            (context_map, "context", "viridis", 0.0, value_max),
            (target_map, "target", "viridis", 0.0, value_max),
            (pred_map, "prediction", "viridis", 0.0, value_max),
            (error, "error", "magma", 0.0, error_max),
        ]
        for col, (image, title, cmap, vmin, vmax) in enumerate(panels):
            axes[row, col].imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(title)
            if col == 0:
                axes[row, col].set_ylabel(f"t={record['latent_index']}")
    fig.suptitle("Independent Single-Step Video Latent Grid", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _normalize_rgb_frames(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.dtype != np.float32:
        frames = frames.astype(np.float32)
    if float(np.nanmax(frames)) > 1.5:
        frames = frames / 255.0
    return np.clip(frames, 0.0, 1.0)


def _plot_decoded_video_grid(records: list[dict], output_path: Path):
    records = [record for record in records if "video_pred_rgb" in record and "video_target_rgb" in record]
    if not records:
        return
    plt = _load_pyplot()
    rows = len(records)
    fig, axes = plt.subplots(rows, 6, figsize=(15.0, max(2.15 * rows, 3.0)), squeeze=False)
    all_diffs = []
    prepared = []
    for record in records:
        pred_rgb = _normalize_rgb_frames(record["video_pred_rgb"])
        target_rgb = _normalize_rgb_frames(record["video_target_rgb"])
        context = target_rgb[0]
        target = target_rgb[1]
        pred = pred_rgb[1]
        target_motion = np.abs(target - context).mean(axis=-1)
        pred_motion = np.abs(pred - context).mean(axis=-1)
        error = np.abs(pred - target).mean(axis=-1)
        all_diffs.extend([target_motion.ravel(), pred_motion.ravel(), error.ravel()])
        prepared.append((record["latent_index"], context, target, pred, target_motion, pred_motion, error))
    diff_max = max(float(np.quantile(np.concatenate(all_diffs), 0.99)), 1e-6)

    titles = ["context", "target", "prediction", "|target-context|", "|pred-context|", "|pred-target|"]
    for row, (latent_index, context, target, pred, target_motion, pred_motion, error) in enumerate(prepared):
        panels = [
            (context, None, 0.0, 1.0),
            (target, None, 0.0, 1.0),
            (pred, None, 0.0, 1.0),
            (target_motion, "magma", 0.0, diff_max),
            (pred_motion, "magma", 0.0, diff_max),
            (error, "magma", 0.0, diff_max),
        ]
        for col, (image, cmap, vmin, vmax) in enumerate(panels):
            if cmap is None:
                axes[row, col].imshow(image)
            else:
                axes[row, col].imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(titles[col])
            if col == 0:
                axes[row, col].set_ylabel(f"t={latent_index}")
    fig.suptitle("Independent Single-Step Decoded Video Grid", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_tactile_maps(
    context: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    output_path: Path,
    channel: int | None,
):
    plt = _load_pyplot()
    if channel is None:
        context_map = context.mean(axis=0)
        pred_map = pred.mean(axis=0)
        target_map = target.mean(axis=0)
        title_suffix = "mean of tactile sheets"
    else:
        context_map = context[channel]
        pred_map = pred[channel]
        target_map = target[channel]
        title_suffix = f"tactile sheet {channel}"

    context_last = context_map[-1]
    future_steps = pred_map.shape[0]
    values = np.concatenate([context_last.ravel(), target_map.ravel(), pred_map.ravel()])
    vmin, vmax = np.quantile(values, [0.01, 0.99])
    if vmax <= vmin:
        vmax = vmin + 1e-6
    diffs = np.concatenate(
        [
            np.abs(target_map - context_last[None]).ravel(),
            np.abs(pred_map - context_last[None]).ravel(),
            np.abs(pred_map - target_map).ravel(),
        ]
    )
    diff_max = max(float(np.quantile(diffs, 0.99)), 1e-6)

    columns = future_steps + 1
    fig, axes = plt.subplots(5, columns, figsize=(2.0 * columns, 9.2), squeeze=False)
    row_labels = [
        "context",
        "target",
        "prediction",
        "|target-context|",
        "|prediction-target|",
    ]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label)
    for row in range(5):
        axes[row, 0].imshow(
            context_last if row < 3 else np.zeros_like(context_last),
            cmap="viridis" if row < 3 else "magma",
            vmin=vmin if row < 3 else 0.0,
            vmax=vmax if row < 3 else diff_max,
            aspect="auto",
        )
        axes[row, 0].set_title("t")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

    for step in range(future_steps):
        panels = [
            context_last,
            target_map[step],
            pred_map[step],
            np.abs(target_map[step] - context_last),
            np.abs(pred_map[step] - target_map[step]),
        ]
        for row, image in enumerate(panels):
            axes[row, step + 1].imshow(
                image,
                cmap="viridis" if row < 3 else "magma",
                vmin=vmin if row < 3 else 0.0,
                vmax=vmax if row < 3 else diff_max,
                aspect="auto",
            )
            axes[row, step + 1].set_title(f"t+1 step {step}")
            axes[row, step + 1].set_xticks([])
            axes[row, step + 1].set_yticks([])

    fig.suptitle(f"Single-Step Tactile Diagnostic ({title_suffix})", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _tactile_summary_maps(context: np.ndarray, pred: np.ndarray, target: np.ndarray):
    context_last = context.mean(axis=0)[-1]
    target_mean = target.mean(axis=0).mean(axis=0)
    pred_mean = pred.mean(axis=0).mean(axis=0)
    error = np.abs(pred.mean(axis=0) - target.mean(axis=0)).mean(axis=0)
    return context_last, target_mean, pred_mean, error


def _plot_tactile_grid(records: list[dict], output_path: Path):
    if not records:
        return
    plt = _load_pyplot()
    rows = len(records)
    fig, axes = plt.subplots(rows, 4, figsize=(9.5, max(2.0 * rows, 3.0)), squeeze=False)
    all_values = []
    all_errors = []
    for record in records:
        context, target, pred, error = _tactile_summary_maps(
            record["tactile_context_raw"],
            record["tactile_pred_raw"],
            record["tactile_target_raw"],
        )
        all_values.extend([context.ravel(), target.ravel(), pred.ravel()])
        all_errors.append(error.ravel())
    vmin, vmax = np.quantile(np.concatenate(all_values), [0.01, 0.99])
    if vmax <= vmin:
        vmax = vmin + 1e-6
    error_max = max(float(np.quantile(np.concatenate(all_errors), 0.99)), 1e-6)

    for row, record in enumerate(records):
        context, target, pred, error = _tactile_summary_maps(
            record["tactile_context_raw"],
            record["tactile_pred_raw"],
            record["tactile_target_raw"],
        )
        panels = [
            (context, "context", "viridis", vmin, vmax),
            (target, "target mean", "viridis", vmin, vmax),
            (pred, "prediction mean", "viridis", vmin, vmax),
            (error, "mean error", "magma", 0.0, error_max),
        ]
        for col, (image, title, cmap, cur_min, cur_max) in enumerate(panels):
            axes[row, col].imshow(image, cmap=cmap, vmin=cur_min, vmax=cur_max, aspect="auto")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(title)
            if col == 0:
                axes[row, col].set_ylabel(f"t={record['latent_index']}")
    fig.suptitle("Independent Single-Step Tactile Grid (mean sheets/substeps)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_action_single_step(
    context: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
):
    plt = _load_pyplot()
    context = context[:, 0]
    pred = pred[:, 0]
    target = target[:, 0]
    mask = mask[:, 0].astype(bool)
    names = ACTION_NAMES_8D if pred.shape[0] == len(ACTION_NAMES_8D) else [f"action_{idx}" for idx in range(pred.shape[0])]
    x_future = np.arange(pred.shape[1])

    rows = int(np.ceil(len(names) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12.8, 2.45 * rows), squeeze=False)
    for channel, name in enumerate(names):
        axis = axes[channel // 2, channel % 2]
        valid = mask[channel]
        axis.axhline(context[channel, -1], color="#555555", linestyle="--", linewidth=1.1, label="context last")
        axis.plot(x_future[valid], target[channel, valid], color="#153e6f", marker="o", label="target")
        axis.plot(x_future[valid], pred[channel, valid], color="#d14a32", marker="x", label="prediction")
        axis.set_title(name)
        axis.set_xlabel("substep inside t+1 latent")
        axis.grid(alpha=0.25)
        if channel == 0:
            axis.legend(frameon=False)
    for channel in range(len(names), rows * 2):
        axes[channel // 2, channel % 2].axis("off")
    fig.suptitle("Single-Step Action Diagnostic", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_action_grid(records: list[dict], output_path: Path):
    if not records:
        return
    plt = _load_pyplot()
    pred = np.stack([record["action_pred_raw_used"][:, 0].mean(axis=1) for record in records], axis=1)
    target = np.stack([record["action_target_raw_used"][:, 0].mean(axis=1) for record in records], axis=1)
    context = np.stack([record["action_context_raw_used"][:, 0, -1] for record in records], axis=1)
    latent_ids = np.asarray([record["latent_index"] for record in records])
    names = ACTION_NAMES_8D if pred.shape[0] == len(ACTION_NAMES_8D) else [f"action_{idx}" for idx in range(pred.shape[0])]

    rows = int(np.ceil(len(names) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(13.0, 2.45 * rows), squeeze=False)
    for channel, name in enumerate(names):
        axis = axes[channel // 2, channel % 2]
        axis.plot(latent_ids, context[channel], color="#777777", linestyle="--", linewidth=1.1, label="context last")
        axis.plot(latent_ids, target[channel], color="#153e6f", marker="o", linewidth=1.4, label="target mean")
        axis.plot(latent_ids, pred[channel], color="#d14a32", marker="x", linewidth=1.3, label="prediction mean")
        axis.set_title(name)
        axis.set_xlabel("latent index")
        axis.grid(alpha=0.25)
        if channel == 0:
            axis.legend(frameon=False)
    for channel in range(len(names), rows * 2):
        axes[channel // 2, channel % 2].axis("off")
    fig.suptitle("Independent Single-Step Action Grid (mean over 4 substeps)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_summary(metrics_list: list[dict], output_path: Path):
    if not metrics_list:
        return
    plt = _load_pyplot()
    latent_ids = np.asarray([item["latent_index"] for item in metrics_list])
    metric_names = [
        "video_latent_mae",
        "video_target_motion_mae",
        "video_pred_motion_mae",
        "tactile_raw_mae",
        "tactile_target_motion_mae",
        "tactile_pred_motion_mae",
        "action_raw_mae",
    ]
    rows = int(np.ceil(len(metric_names) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12.8, 2.5 * rows), squeeze=False)
    for idx, name in enumerate(metric_names):
        axis = axes[idx // 2, idx % 2]
        values = [item.get(name) for item in metrics_list]
        y = np.asarray([np.nan if value is None else float(value) for value in values])
        axis.plot(latent_ids, y, marker="o", linewidth=1.4)
        axis.set_title(name)
        axis.set_xlabel("latent index")
        axis.grid(alpha=0.25)
    for idx in range(len(metric_names), rows * 2):
        axes[idx // 2, idx % 2].axis("off")
    fig.suptitle("Independent Single-Step Metric Summary", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _metrics_from_payload(payload: dict) -> dict:
    video_pred = payload["video_pred_latent"][0].float()
    video_target = payload["video_target_latent"][0].float()
    video_context = payload["video_context_latent"][0].float()
    tactile_pred = payload["tactile_pred_raw"][0].float()
    tactile_target = payload["tactile_target_raw"][0].float()
    tactile_context = payload["tactile_context_raw"][0].float()
    action_pred = payload["action_pred_raw_used"][0].float()
    action_target = payload["action_target_raw_used"][0].float()
    action_mask = payload["action_target_mask_used"][0].bool()

    valid_action_error = torch.abs(action_pred - action_target)[action_mask]
    return {
        "video_latent_mae": float(torch.abs(video_pred - video_target).mean().item()),
        "video_target_motion_mae": float(torch.abs(video_target - video_context).mean().item()),
        "video_pred_motion_mae": float(torch.abs(video_pred - video_context).mean().item()),
        "tactile_raw_mae": float(torch.abs(tactile_pred - tactile_target).mean().item()),
        "tactile_target_motion_mae": float(
            torch.abs(tactile_target - tactile_context[:, -1:].expand_as(tactile_target)).mean().item()
        ),
        "tactile_pred_motion_mae": float(
            torch.abs(tactile_pred - tactile_context[:, -1:].expand_as(tactile_pred)).mean().item()
        ),
        "action_raw_mae": float(valid_action_error.mean().item()) if valid_action_error.numel() else None,
    }


def _save_single_step_outputs(payload: dict, output_dir: Path, args, config, save_video_mp4: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_dir / "single_step_multimodal_rollout.pt")

    metrics = _metrics_from_payload(payload)
    metrics.update(payload["meta"])
    with (output_dir / "single_step_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    _plot_video_latent_maps(
        _to_numpy(payload["video_context_latent"][0]),
        _to_numpy(payload["video_pred_latent"][0]),
        _to_numpy(payload["video_target_latent"][0]),
        output_dir / "video_latent_single_step.png",
    )
    _plot_tactile_maps(
        _to_numpy(payload["tactile_context_raw"][0]),
        _to_numpy(payload["tactile_pred_raw"][0]),
        _to_numpy(payload["tactile_target_raw"][0]),
        output_dir / "tactile_single_step_mean.png",
        channel=None,
    )
    if args.plot_channels:
        for channel in range(payload["tactile_pred_raw"].shape[1]):
            _plot_tactile_maps(
                _to_numpy(payload["tactile_context_raw"][0]),
                _to_numpy(payload["tactile_pred_raw"][0]),
                _to_numpy(payload["tactile_target_raw"][0]),
                output_dir / f"tactile_single_step_sheet_{channel}.png",
                channel=channel,
            )
    _plot_action_single_step(
        _to_numpy(payload["action_context_raw_used"][0]),
        _to_numpy(payload["action_pred_raw_used"][0]),
        _to_numpy(payload["action_target_raw_used"][0]),
        _to_numpy(payload["action_target_mask_used"][0]),
        output_dir / "action_single_step.png",
    )

    if args.decode_video:
        pred_rgb = _to_numpy(payload["video_pred_rgb"])
        target_rgb = _to_numpy(payload["video_target_rgb"])
        _plot_video_frames(pred_rgb, target_rgb, output_dir / "video_decoded_single_step.png")
        if save_video_mp4:
            _write_decoded_videos(pred_rgb, target_rgb, output_dir, args.video_fps)
    return metrics


def _record_from_payload(payload: dict) -> dict:
    record = {
        "latent_index": int(payload["meta"]["latent_index"]),
        "video_context_latent": _to_numpy(payload["video_context_latent"][0]),
        "video_pred_latent": _to_numpy(payload["video_pred_latent"][0]),
        "video_target_latent": _to_numpy(payload["video_target_latent"][0]),
        "tactile_context_raw": _to_numpy(payload["tactile_context_raw"][0]),
        "tactile_pred_raw": _to_numpy(payload["tactile_pred_raw"][0]),
        "tactile_target_raw": _to_numpy(payload["tactile_target_raw"][0]),
        "action_context_raw_used": _to_numpy(payload["action_context_raw_used"][0]),
        "action_pred_raw_used": _to_numpy(payload["action_pred_raw_used"][0]),
        "action_target_raw_used": _to_numpy(payload["action_target_raw_used"][0]),
    }
    if "video_pred_rgb" in payload and "video_target_rgb" in payload:
        record["video_pred_rgb"] = _to_numpy(payload["video_pred_rgb"])
        record["video_target_rgb"] = _to_numpy(payload["video_target_rgb"])
    return record


def _write_summary_outputs(records: list[dict], metrics_list: list[dict], output_dir: Path, args):
    if len(records) <= 1:
        return
    with (output_dir / "single_step_metrics_all.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_list, handle, indent=2)
    _plot_metric_summary(metrics_list, output_dir / "metrics_grid.png")
    _plot_video_latent_grid(records, output_dir / "video_latent_grid.png")
    _plot_tactile_grid(records, output_dir / "tactile_grid_mean.png")
    _plot_action_grid(records, output_dir / "action_grid.png")
    if args.decode_video:
        _plot_decoded_video_grid(records, output_dir / "video_decoded_grid.png")


def run(args):
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is not available; pass --device cpu to run on CPU.")

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

    dtype = _dtype_from_name(args.dtype)
    tactile_per_frame = int(config.tactile_per_frame)
    rollout = TactileFutureRollout(
        config=config,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        dtype=dtype,
        predict_video=True,
        predict_actions=True,
    )

    logger.info("Loading tactile LeRobot dataset...")
    dataset = MultiTactileLatentLeRobotDataset(config=config, num_init_worker=args.dataset_init_worker)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.load_worker)
    if args.segment_index >= len(dataset):
        raise ValueError(f"segment-index={args.segment_index} exceeds dataset length {len(dataset)}")

    batch = None
    for index, item in enumerate(loader):
        if index == args.segment_index:
            batch = item
            break
    if batch is None:
        raise RuntimeError(f"Could not load segment {args.segment_index}")

    num_steps = int(args.num_latent_steps)
    latent_indices = list(range(args.latent_index, args.latent_index + num_steps))
    total_latent_frames = int(batch["latents"].shape[2])
    if latent_indices[-1] + 1 >= total_latent_frames:
        raise ValueError(
            f"Requested latent range {latent_indices[0]}..{latent_indices[-1]} needs "
            f"target up to {latent_indices[-1] + 1}, but segment only has "
            f"{total_latent_frames} latent frames."
        )

    base_output_dir = Path(args.output_dir).expanduser()
    if num_steps == 1:
        output_root = base_output_dir / f"segment_{args.segment_index:06d}_latent_{args.latent_index:04d}"
    else:
        output_root = (
            base_output_dir
            / f"segment_{args.segment_index:06d}_latent_{latent_indices[0]:04d}_{latent_indices[-1]:04d}"
        )
        output_root.mkdir(parents=True, exist_ok=True)

    records = []
    metrics_list = []
    for latent_index in latent_indices:
        step_seed = int(args.seed) + int(latent_index)
        torch.manual_seed(step_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(step_seed)

        cropped = _crop_single_step_batch(batch, latent_index, tactile_per_frame)
        result = rollout.predict(
            cropped,
            context_frames=1,
            future_frames=1,
            video_steps=args.video_inference_steps,
            tactile_steps=args.tactile_inference_steps,
            action_steps=args.action_inference_steps,
        )
        cropped_device = {
            key: value.to(rollout.device) if torch.is_tensor(value) else value
            for key, value in cropped.items()
        }
        result["action_context_raw_used"] = _latest_context_action_raw(rollout, cropped_device)

        if args.decode_video:
            result.update(rollout.decode_video_sequences(result, decode_device=args.video_decode_device))

        payload = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in result.items()
        }
        payload["meta"] = {
            "segment_index": args.segment_index,
            "latent_index": latent_index,
            "context_latent_frames": 1,
            "future_latent_frames": 1,
            "future_action_steps": int(config.action_per_frame),
            "future_tactile_steps": tactile_per_frame,
            "video_inference_steps": args.video_inference_steps,
            "tactile_inference_steps": args.tactile_inference_steps,
            "action_inference_steps": args.action_inference_steps,
            "num_latent_steps": num_steps,
        }

        step_output_dir = output_root if num_steps == 1 else output_root / f"latent_{latent_index:04d}"
        metrics = _save_single_step_outputs(
            payload,
            step_output_dir,
            args,
            config,
            save_video_mp4=args.save_video_mp4,
        )
        records.append(_record_from_payload(payload))
        metrics_list.append(metrics)
        print(f"[OK] latent={latent_index} -> {step_output_dir}")

        rollout.transformer.clear_cache(rollout.cache_name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    _write_summary_outputs(records, metrics_list, output_root, args)
    print(f"[OK] single-step multimodal diagnostic -> {output_root}")
    print(json.dumps(metrics_list if num_steps > 1 else metrics_list[0], indent=2))


def main():
    parser = argparse.ArgumentParser(description="Detailed single-step multimodal LingBot-VA rollout diagnostic.")
    parser.add_argument("--config-name", type=str, default="robotwin_tactile_train")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="/data/lingbot-va-models/lingbot-va-base")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--stats-json-path", type=str, default="lingbot-va-tactile/tactile_stats.json")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--segment-index", type=int, default=0)
    parser.add_argument("--latent-index", type=int, default=0)
    parser.add_argument(
        "--num-latent-steps",
        type=int,
        default=1,
        help="Run independent t -> t+1 checks for latent-index through latent-index + N - 1.",
    )
    parser.add_argument("--video-inference-steps", type=int, default=25)
    parser.add_argument("--tactile-inference-steps", type=int, default=50)
    parser.add_argument("--action-inference-steps", type=int, default=50)
    parser.add_argument("--decode-video", action="store_true")
    parser.add_argument("--video-decode-device", type=str, default="cpu")
    parser.add_argument("--save-video-mp4", action="store_true")
    parser.add_argument("--video-fps", type=int, default=16)
    parser.add_argument("--plot-channels", action="store_true")
    parser.add_argument("--dataset-init-worker", type=int, default=1)
    parser.add_argument("--load-worker", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    if args.save_video_mp4 and not args.decode_video:
        raise ValueError("--save-video-mp4 requires --decode-video.")
    if min(args.video_inference_steps, args.tactile_inference_steps, args.action_inference_steps) <= 0:
        raise ValueError("All modality inference-step arguments must be positive.")
    if args.num_latent_steps <= 0:
        raise ValueError("--num-latent-steps must be positive.")
    run(args)


if __name__ == "__main__":
    main()
