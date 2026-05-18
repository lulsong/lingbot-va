#!/usr/bin/env python3
"""Visualize offline LingBot-VA-Tactile prediction dumps.

Input files are produced by ``validate_tactile_offline.py`` with
``--save-prediction-batches``. This script converts the saved tensors into
human-readable PNG plots and CSV files.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch


ACTION_NAMES_8D = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _prediction_files(input_path):
    input_path = Path(input_path).expanduser()
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob("prediction_batch_*.pt"))
        if not files:
            files = sorted(input_path.glob("*.pt"))
        return files
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _select_sample(array, sample_index):
    if array.ndim > 0 and array.shape[0] > sample_index:
        return array[sample_index]
    raise IndexError(f"sample_index={sample_index} is out of range for shape {array.shape}")


def _flatten_action_time(action):
    # Saved shape before sample selection: [B, C_used, F, action_per_frame].
    # Shape after sample selection: [C_used, F, action_per_frame].
    if action.ndim == 4:
        if action.shape[0] != 1:
            raise ValueError(
                f"Expected a single action sample or pre-selected sample, got {action.shape}"
            )
        action = action[0]
    if action.ndim != 3:
        raise ValueError(f"Expected action shape [C,F,N] or [B,C,F,N], got {action.shape}")
    channels, frames, steps_per_frame = action.shape
    return action.reshape(channels, frames * steps_per_frame)


def write_action_csv(pred, target, out_path):
    channels, time_steps = pred.shape
    names = ACTION_NAMES_8D if channels == len(ACTION_NAMES_8D) else [f"action_{i}" for i in range(channels)]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["t"]
        for name in names:
            header += [f"pred_{name}", f"target_{name}", f"abs_err_{name}"]
        writer.writerow(header)
        for t in range(time_steps):
            row = [t]
            for c in range(channels):
                row += [pred[c, t], target[c, t], abs(pred[c, t] - target[c, t])]
            writer.writerow(row)


def plot_action(payload, out_dir, sample_index):
    plt = _load_pyplot()
    pred = _select_sample(_to_numpy(payload["action_pred_raw_used"]), sample_index)
    target = _select_sample(_to_numpy(payload["action_target_raw_used"]), sample_index)
    pred = _flatten_action_time(pred)
    target = _flatten_action_time(target)

    channels, time_steps = pred.shape
    names = ACTION_NAMES_8D if channels == len(ACTION_NAMES_8D) else [f"action_{i}" for i in range(channels)]
    rows = int(np.ceil(channels / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(14, 2.4 * rows), squeeze=False)
    x = np.arange(time_steps)
    for c, name in enumerate(names):
        ax = axes[c // 2][c % 2]
        ax.plot(x, target[c], label="target", linewidth=1.8, color="#1f77b4")
        ax.plot(x, pred[c], label="pred", linewidth=1.5, color="#d62728", alpha=0.9)
        ax.set_title(name)
        ax.grid(True, alpha=0.25)
        if c == 0:
            ax.legend(loc="best")
    for c in range(channels, rows * 2):
        axes[c // 2][c % 2].axis("off")
    fig.suptitle("Action Prediction vs Target", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir / "action_curves.png", dpi=180)
    plt.close(fig)
    write_action_csv(pred, target, out_dir / "action_values.csv")


def write_tactile_summary_csv(pred, target, threshold, out_path):
    # Shape: [C, T, H, W].
    pred_mean = pred.mean(axis=(0, 2, 3))
    target_mean = target.mean(axis=(0, 2, 3))
    pred_max = pred.max(axis=(0, 2, 3))
    target_max = target.max(axis=(0, 2, 3))
    mae = np.abs(pred - target).mean(axis=(0, 2, 3))
    pred_contact = (pred > threshold).mean(axis=(0, 2, 3))
    target_contact = (target > threshold).mean(axis=(0, 2, 3))
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "t",
                "pred_mean",
                "target_mean",
                "pred_max",
                "target_max",
                "mae",
                "pred_contact_ratio",
                "target_contact_ratio",
            ]
        )
        for t in range(pred.shape[1]):
            writer.writerow(
                [
                    t,
                    pred_mean[t],
                    target_mean[t],
                    pred_max[t],
                    target_max[t],
                    mae[t],
                    pred_contact[t],
                    target_contact[t],
                ]
            )


def plot_tactile_summary(pred, target, threshold, out_dir):
    plt = _load_pyplot()
    time_steps = pred.shape[1]
    x = np.arange(time_steps)
    pred_mean = pred.mean(axis=(0, 2, 3))
    target_mean = target.mean(axis=(0, 2, 3))
    pred_max = pred.max(axis=(0, 2, 3))
    target_max = target.max(axis=(0, 2, 3))
    mae = np.abs(pred - target).mean(axis=(0, 2, 3))
    pred_contact = (pred > threshold).mean(axis=(0, 2, 3))
    target_contact = (target > threshold).mean(axis=(0, 2, 3))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
    axes[0][0].plot(x, target_mean, label="target", color="#1f77b4")
    axes[0][0].plot(x, pred_mean, label="pred", color="#d62728")
    axes[0][0].set_title("Mean Pressure")
    axes[0][0].legend()
    axes[0][1].plot(x, target_max, label="target", color="#1f77b4")
    axes[0][1].plot(x, pred_max, label="pred", color="#d62728")
    axes[0][1].set_title("Max Pressure")
    axes[0][1].legend()
    axes[1][0].plot(x, mae, color="#9467bd")
    axes[1][0].set_title("Mean Absolute Error")
    axes[1][1].plot(x, target_contact, label="target", color="#1f77b4")
    axes[1][1].plot(x, pred_contact, label="pred", color="#d62728")
    axes[1][1].set_title(f"Contact Ratio > {threshold:g}")
    axes[1][1].legend()
    for row in axes:
        for ax in row:
            ax.set_xlabel("tactile frame")
            ax.grid(True, alpha=0.25)
    fig.suptitle("Tactile Temporal Summary", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir / "tactile_summary.png", dpi=180)
    plt.close(fig)
    write_tactile_summary_csv(pred, target, threshold, out_dir / "tactile_summary.csv")


def _combined_tactile_map(tactile):
    # Shape [C, T, H, W] -> [T, H, W].
    return tactile.mean(axis=0)


def plot_tactile_heatmaps(pred, target, out_dir, num_frames):
    plt = _load_pyplot()
    pred_map = _combined_tactile_map(pred)
    target_map = _combined_tactile_map(target)
    error_map = np.abs(pred_map - target_map)
    time_steps = pred_map.shape[0]
    frame_ids = np.linspace(0, time_steps - 1, min(num_frames, time_steps), dtype=int)

    vmin = min(float(pred_map.min()), float(target_map.min()))
    vmax = max(float(pred_map.max()), float(target_map.max()))
    err_vmax = max(float(error_map.max()), 1e-6)

    fig, axes = plt.subplots(len(frame_ids), 3, figsize=(12, 2.7 * len(frame_ids)), squeeze=False)
    for row, frame_id in enumerate(frame_ids):
        images = [
            (target_map[frame_id], "target", vmin, vmax, "viridis"),
            (pred_map[frame_id], "pred", vmin, vmax, "viridis"),
            (error_map[frame_id], "abs error", 0.0, err_vmax, "magma"),
        ]
        for col, (image, title, cur_vmin, cur_vmax, cmap) in enumerate(images):
            ax = axes[row][col]
            im = ax.imshow(image, cmap=cmap, vmin=cur_vmin, vmax=cur_vmax, aspect="auto")
            ax.set_title(f"t={frame_id} {title}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle("Tactile Heatmaps: Target / Prediction / Error", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_dir / "tactile_heatmaps_combined.png", dpi=180)
    plt.close(fig)


def plot_tactile_channels(pred, target, out_dir, num_frames):
    plt = _load_pyplot()
    channels, time_steps, _, _ = pred.shape
    frame_ids = np.linspace(0, time_steps - 1, min(num_frames, time_steps), dtype=int)
    for channel in range(channels):
        pred_ch = pred[channel]
        target_ch = target[channel]
        error_ch = np.abs(pred_ch - target_ch)
        vmin = min(float(pred_ch.min()), float(target_ch.min()))
        vmax = max(float(pred_ch.max()), float(target_ch.max()))
        err_vmax = max(float(error_ch.max()), 1e-6)
        fig, axes = plt.subplots(len(frame_ids), 3, figsize=(12, 2.7 * len(frame_ids)), squeeze=False)
        for row, frame_id in enumerate(frame_ids):
            images = [
                (target_ch[frame_id], "target", vmin, vmax, "viridis"),
                (pred_ch[frame_id], "pred", vmin, vmax, "viridis"),
                (error_ch[frame_id], "abs error", 0.0, err_vmax, "magma"),
            ]
            for col, (image, title, cur_vmin, cur_vmax, cmap) in enumerate(images):
                ax = axes[row][col]
                im = ax.imshow(image, cmap=cmap, vmin=cur_vmin, vmax=cur_vmax, aspect="auto")
                ax.set_title(f"channel={channel} t={frame_id} {title}")
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        fig.suptitle(f"Tactile Channel {channel}", fontsize=15)
        fig.tight_layout()
        fig.savefig(out_dir / f"tactile_heatmaps_channel_{channel}.png", dpi=180)
        plt.close(fig)


def plot_tactile(payload, out_dir, sample_index, threshold, num_frames, plot_channels):
    pred = _select_sample(_to_numpy(payload["tactile_pred_raw"]), sample_index)
    target = _select_sample(_to_numpy(payload["tactile_target_raw"]), sample_index)
    if pred.ndim != 4:
        raise ValueError(f"Expected tactile shape [C,T,H,W], got {pred.shape}")
    plot_tactile_summary(pred, target, threshold, out_dir)
    plot_tactile_heatmaps(pred, target, out_dir, num_frames)
    if plot_channels:
        plot_tactile_channels(pred, target, out_dir, num_frames)


def plot_video_latent_norm(payload, out_dir, sample_index):
    if "video_pred_latent" not in payload or "video_target_latent" not in payload:
        return
    plt = _load_pyplot()
    pred = _select_sample(_to_numpy(payload["video_pred_latent"]), sample_index)
    target = _select_sample(_to_numpy(payload["video_target_latent"]), sample_index)
    # Shape [C,F,H,W] -> norm per latent frame.
    pred_norm = np.sqrt((pred**2).mean(axis=(0, 2, 3)))
    target_norm = np.sqrt((target**2).mean(axis=(0, 2, 3)))
    mae = np.abs(pred - target).mean(axis=(0, 2, 3))
    x = np.arange(pred.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x, target_norm, label="target", color="#1f77b4")
    axes[0].plot(x, pred_norm, label="pred", color="#d62728")
    axes[0].set_title("Video Latent RMS")
    axes[0].legend()
    axes[1].plot(x, mae, color="#9467bd")
    axes[1].set_title("Video Latent MAE")
    for ax in axes:
        ax.set_xlabel("latent frame")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Video Latent Summary")
    fig.tight_layout()
    fig.savefig(out_dir / "video_latent_summary.png", dpi=180)
    plt.close(fig)


def visualize_file(file_path, output_root, sample_index, threshold, num_tactile_frames, plot_channels):
    payload = torch.load(file_path, map_location="cpu", weights_only=False)
    out_dir = output_root / file_path.stem / f"sample_{sample_index:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_action(payload, out_dir, sample_index)
    plot_tactile(payload, out_dir, sample_index, threshold, num_tactile_frames, plot_channels)
    plot_video_latent_norm(payload, out_dir, sample_index)

    meta_path = out_dir / "metadata.txt"
    with open(meta_path, "w") as f:
        f.write(f"source={file_path}\n")
        if "meta" in payload:
            for key, value in payload["meta"].items():
                f.write(f"{key}={value}\n")
        if "losses" in payload:
            for key, value in payload["losses"].items():
                f.write(f"loss/{key}={value}\n")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Visualize offline tactile prediction .pt files")
    parser.add_argument("--input", type=str, required=True, help="prediction .pt file or directory")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--num-tactile-frames", type=int, default=8)
    parser.add_argument("--contact-threshold", type=float, default=0.05)
    parser.add_argument("--plot-channels", action="store_true")
    args = parser.parse_args()

    files = _prediction_files(args.input)
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No .pt prediction files found under {args.input}")

    output_root = Path(args.output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        out_dir = visualize_file(
            file_path=file_path,
            output_root=output_root,
            sample_index=args.sample_index,
            threshold=args.contact_threshold,
            num_tactile_frames=args.num_tactile_frames,
            plot_channels=args.plot_channels,
        )
        print(f"[OK] {file_path} -> {out_dir}")


if __name__ == "__main__":
    main()
