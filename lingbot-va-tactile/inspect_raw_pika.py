#!/usr/bin/env python3
"""Inspect raw PIKA real-machine episodes before LeRobot conversion."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from PIL import Image


@dataclass
class Args:
    input_root: Path = Path("/data/Datasets/PIKA_real_original/insert_peg_cylinder_RealMachine")
    sample_episodes: int = 5
    sample_tactile_frames_per_episode: int = 5


def _sorted_episode_dirs(root: Path) -> list[Path]:
    episodes = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("episode")]

    def key(path: Path) -> tuple[int, str]:
        suffix = path.name.replace("episode", "")
        return (int(suffix), path.name) if suffix.isdigit() else (10**9, path.name)

    return sorted(episodes, key=key)


def main(args: Args) -> None:
    root = args.input_root.expanduser().resolve()
    episodes = _sorted_episode_dirs(root)
    if not episodes:
        raise RuntimeError(f"No episode directories found under {root}")

    required = [
        "timestamp",
        "filepath_color",
        "filepath_side_rgb",
        "filepath_fisheye",
        "filepath_tactile_gripper",
        "action_delta_x",
        "action_delta_y",
        "action_delta_z",
        "action_delta_qx",
        "action_delta_qy",
        "action_delta_qz",
        "action_delta_qw",
        "gripper_target_distance_mm",
    ]

    row_counts: list[int] = []
    tactile_shapes: dict[tuple[int, ...], int] = {}
    image_shapes: dict[str, dict[tuple[int, ...], int]] = {
        "filepath_color": {},
        "filepath_side_rgb": {},
        "filepath_fisheye": {},
    }
    missing: dict[str, int] = {}
    tactile_max: list[float] = []
    tactile_nonzero: list[int] = []

    for episode in episodes:
        csv_path = episode / "teleop_log.csv"
        if not csv_path.exists():
            missing["teleop_log.csv"] = missing.get("teleop_log.csv", 0) + 1
            continue

        rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
        row_counts.append(len(rows))
        if not rows:
            continue

        for key in required:
            if key not in rows[0]:
                missing[key] = missing.get(key, 0) + 1

        for key in image_shapes:
            value = rows[0].get(key)
            if value:
                path = episode / value
                if path.exists():
                    with Image.open(path) as image:
                        shape = (image.height, image.width, len(image.getbands()))
                    image_shapes[key][shape] = image_shapes[key].get(shape, 0) + 1
                else:
                    missing[key] = missing.get(key, 0) + 1

        stride = max(1, len(rows) // max(1, args.sample_tactile_frames_per_episode))
        for row in rows[::stride][: args.sample_tactile_frames_per_episode]:
            tactile_path = episode / row["filepath_tactile_gripper"]
            if not tactile_path.exists():
                missing["filepath_tactile_gripper"] = missing.get("filepath_tactile_gripper", 0) + 1
                continue
            data = np.load(tactile_path)
            arrays = [data[key].astype(np.float32) for key in sorted(data.files)[:2]]
            tactile = np.stack(arrays, axis=0)
            tactile_shapes[tactile.shape] = tactile_shapes.get(tactile.shape, 0) + 1
            tactile_max.append(float(np.nanmax(tactile)))
            tactile_nonzero.append(int(np.count_nonzero(tactile)))

    print(f"root: {root}")
    print(f"episodes: {len(episodes)}")
    print(f"rows: min={min(row_counts)} max={max(row_counts)} avg={sum(row_counts) / len(row_counts):.1f}")
    print(f"image_shapes: {image_shapes}")
    print(f"tactile_shapes: {tactile_shapes}")
    print(f"tactile_dim: {int(np.prod(next(iter(tactile_shapes)))) if tactile_shapes else 'unknown'}")
    print(f"tactile_max: min={min(tactile_max):.6f} max={max(tactile_max):.6f}")
    print(f"tactile_nonzero: min={min(tactile_nonzero)} max={max(tactile_nonzero)}")
    print(f"missing_or_schema_issues: {missing}")


if __name__ == "__main__":
    main(tyro.cli(Args))

