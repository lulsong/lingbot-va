#!/usr/bin/env python3
"""Compute action/tactile quantile stats from a converted LeRobot dataset.

This writes a JSON snippet that can be copied into
``tactile_va/configs/robotwin_tactile_cfg.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro


@dataclass
class Args:
    repo_id: str = "local/make-coffee-tactile"
    root: Path = Path("/data/data_realworld/lerobot_export_dataset/local/make-coffee-tactile")
    tactile_key: str = "observation.tactile"
    max_frames: int = 0
    output_json: Path = Path("lingbot-va-tactile/tactile_stats.json")
    model_action_dim: int = 30
    action_layout_name: str = "joint_gripper"
    tactile_min_range: float = 1e-4
    # "auto" maps the selected raw action layout to LingBot-VA's 30D left-arm slots.
    used_action_channel_ids: str = "auto"


def _resolve_lerobot_dataset() -> Any:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _as_flat_float32(value: Any) -> np.ndarray:
    array = _to_numpy(value)
    if array.dtype == object:
        array = np.asarray(array.tolist())
        if array.dtype == object:
            array = np.stack([np.asarray(item) for item in array])
        if array.dtype == object:
            array = np.stack([np.asarray(item.tolist()) for item in array])
    return array.reshape(1, -1).astype(np.float32)


def _local_parquet_files(root: Path) -> list[Path]:
    data_dir = root.expanduser() / "data"
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("**/*.parquet"))


def _load_from_local_parquet(args: Args) -> tuple[np.ndarray, np.ndarray] | None:
    parquet_files = _local_parquet_files(args.root)
    if not parquet_files:
        return None

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Local parquet loading requires pandas. Install pandas or use a LeRobot version "
            "that can load this dataset metadata directly."
        ) from exc

    action_values: list[np.ndarray] = []
    tactile_values: list[np.ndarray] = []
    remaining = None if args.max_frames <= 0 else args.max_frames

    for parquet_path in parquet_files:
        if remaining == 0:
            break

        df = pd.read_parquet(parquet_path, columns=["action", args.tactile_key])
        if remaining is not None:
            df = df.iloc[:remaining]
            remaining -= len(df)

        for _, row in df.iterrows():
            action_values.append(_as_flat_float32(row["action"]))
            tactile_values.append(_as_flat_float32(row[args.tactile_key]))

    if not action_values:
        raise RuntimeError(f"No frames found in local parquet dataset: {args.root}")

    return np.concatenate(action_values, axis=0), np.concatenate(tactile_values, axis=0)


def _load_from_lerobot_api(args: Args) -> tuple[np.ndarray, np.ndarray]:
    LeRobotDataset = _resolve_lerobot_dataset()
    dataset = LeRobotDataset(repo_id=args.repo_id, root=str(args.root))

    action_values: list[np.ndarray] = []
    tactile_values: list[np.ndarray] = []
    frame_count = len(dataset) if args.max_frames <= 0 else min(len(dataset), args.max_frames)

    for idx in range(frame_count):
        item = dataset[idx]
        action_values.append(_to_numpy(item["action"]).reshape(1, -1).astype(np.float32))
        tactile_values.append(_to_numpy(item[args.tactile_key]).reshape(1, -1).astype(np.float32))

    return np.concatenate(action_values, axis=0), np.concatenate(tactile_values, axis=0)


def _read_tactile_shape(root: Path, tactile_key: str) -> list[int] | None:
    info_path = root.expanduser() / "meta" / "info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    feature = info.get("features", {}).get(tactile_key)
    shape = feature.get("shape") if isinstance(feature, dict) else None
    return [int(value) for value in shape] if shape is not None else None


def _read_action_config_summary(root: Path) -> dict[str, Any]:
    episodes_path = root.expanduser() / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return {
            "episodes": 0,
            "segments": 0,
            "segment_frames_min": 0,
            "segment_frames_mean": 0.0,
            "segment_frames_max": 0,
        }

    segment_lengths: list[int] = []
    episode_count = 0
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        episode_count += 1
        item = json.loads(line)
        for segment in item.get("action_config", []):
            segment_lengths.append(int(segment["end_frame"]) - int(segment["start_frame"]))

    if not segment_lengths:
        return {
            "episodes": episode_count,
            "segments": 0,
            "segment_frames_min": 0,
            "segment_frames_mean": 0.0,
            "segment_frames_max": 0,
        }

    lengths = np.asarray(segment_lengths, dtype=np.int64)
    return {
        "episodes": int(episode_count),
        "segments": int(lengths.size),
        "segment_frames_min": int(lengths.min()),
        "segment_frames_mean": float(lengths.mean()),
        "segment_frames_max": int(lengths.max()),
    }


def _raw_action_order(raw_action_dim: int, action_layout_name: str) -> list[str]:
    eef = ["eef_x", "eef_y", "eef_z", "eef_qx", "eef_qy", "eef_qz", "eef_qw"]
    joints = [f"joint_{idx}" for idx in range(1, 8)]
    if action_layout_name == "joint_gripper" and raw_action_dim == 8:
        return joints + ["gripper"]
    if action_layout_name == "eef_gripper" and raw_action_dim == 8:
        return eef + ["gripper"]
    if action_layout_name == "eef_joint_gripper" and raw_action_dim == 15:
        return eef + joints + ["gripper"]
    return [f"action_{idx}" for idx in range(raw_action_dim)]


def _auto_channel_ids(raw_action_dim: int, action_layout_name: str) -> list[int]:
    if action_layout_name == "joint_gripper" and raw_action_dim == 8:
        return list(range(14, 21)) + [28]
    if action_layout_name == "eef_gripper" and raw_action_dim == 8:
        return list(range(7)) + [28]
    if action_layout_name == "eef_joint_gripper" and raw_action_dim == 15:
        return list(range(7)) + list(range(14, 21)) + [28]
    raise ValueError(
        "Cannot infer used_action_channel_ids automatically for "
        f"action_layout_name={action_layout_name!r}, raw_action_dim={raw_action_dim}. "
        "Pass --used-action-channel-ids explicitly or regenerate with --action-layout joint_gripper."
    )


def _parse_channel_ids(
    value: str,
    raw_action_dim: int,
    model_action_dim: int,
    action_layout_name: str,
) -> list[int]:
    if value.strip().lower() == "auto":
        channel_ids = _auto_channel_ids(raw_action_dim, action_layout_name)
    else:
        channel_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(channel_ids) != raw_action_dim:
        raise ValueError(
            f"used_action_channel_ids must contain {raw_action_dim} ids, got {len(channel_ids)}: {channel_ids}"
        )
    if len(set(channel_ids)) != len(channel_ids):
        raise ValueError(f"used_action_channel_ids contains duplicates: {channel_ids}")
    invalid = [idx for idx in channel_ids if idx < 0 or idx >= model_action_dim]
    if invalid:
        raise ValueError(
            f"used_action_channel_ids contains ids outside [0, {model_action_dim}): {invalid}"
        )
    return channel_ids


def _validate_action_layout_name(action_layout_name: str, raw_action_dim: int) -> None:
    expected_dims = {
        "joint_gripper": 8,
        "eef_gripper": 8,
        "eef_joint_gripper": 15,
    }
    if action_layout_name not in expected_dims:
        raise ValueError(
            f"Unsupported action_layout_name={action_layout_name!r}; "
            f"choose from {sorted(expected_dims)}"
        )
    expected_dim = expected_dims[action_layout_name]
    if raw_action_dim != expected_dim:
        raise ValueError(
            f"action_layout_name={action_layout_name!r} expects action_dim={expected_dim}, "
            f"got {raw_action_dim}. Do not mix old EEF/15D stats with joint_gripper training."
        )


def main(args: Args) -> None:
    loaded = _load_from_local_parquet(args)
    if loaded is None:
        loaded = _load_from_lerobot_api(args)

    actions, tactile = loaded
    frame_count = actions.shape[0]
    _validate_action_layout_name(args.action_layout_name, actions.shape[1])

    action_q01 = np.quantile(actions, 0.01, axis=0).astype(np.float32)
    action_q99 = np.quantile(actions, 0.99, axis=0).astype(np.float32)
    used_action_channel_ids = _parse_channel_ids(
        args.used_action_channel_ids,
        raw_action_dim=actions.shape[1],
        model_action_dim=args.model_action_dim,
        action_layout_name=args.action_layout_name,
    )
    padded_action_q01 = np.zeros(args.model_action_dim, dtype=np.float32)
    padded_action_q99 = np.zeros(args.model_action_dim, dtype=np.float32)
    for raw_idx, model_idx in enumerate(used_action_channel_ids):
        padded_action_q01[model_idx] = action_q01[raw_idx]
        padded_action_q99[model_idx] = action_q99[raw_idx]

    tactile_q01 = np.quantile(tactile, 0.01, axis=0).astype(np.float32)
    tactile_q99 = np.quantile(tactile, 0.99, axis=0).astype(np.float32)
    tactile_range = tactile_q99 - tactile_q01
    low_range_taxels = int((tactile_range < float(args.tactile_min_range)).sum())

    result = {
        "repo_id": args.repo_id,
        "root": str(args.root.expanduser().resolve()),
        "num_frames": int(frame_count),
        "action_dim": int(actions.shape[1]),
        "action_layout_name": args.action_layout_name,
        "model_action_dim": int(args.model_action_dim),
        "tactile_dim": int(tactile.shape[1]),
        "tactile_shape": _read_tactile_shape(args.root, args.tactile_key),
        "tactile_min_range": float(args.tactile_min_range),
        "tactile_low_range_taxels": low_range_taxels,
        "action_config_summary": _read_action_config_summary(args.root),
        "action_layout": {
            "raw_action_order": _raw_action_order(actions.shape[1], args.action_layout_name),
            "model_action_order": [
                "left_eef_7",
                "right_eef_7",
                "left_joints_7",
                "right_joints_7",
                "left_gripper",
                "right_gripper",
            ],
            "single_arm_slot": "left",
        },
        "used_action_channel_ids": used_action_channel_ids,
        "norm_stat": {
            "q01": padded_action_q01.astype(float).tolist(),
            "q99": padded_action_q99.astype(float).tolist(),
        },
        "tactile_norm_stat": {
            "q01": tactile_q01.astype(float).tolist(),
            "q99": tactile_q99.astype(float).tolist(),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"wrote {args.output_json}")
    print(f"action_dim={result['action_dim']} tactile_dim={result['tactile_dim']} frames={frame_count}")
    print(f"used_action_channel_ids={used_action_channel_ids}")
    print(f"tactile_low_range_taxels={low_range_taxels}")


if __name__ == "__main__":
    main(tyro.cli(Args))
