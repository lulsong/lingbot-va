#!/usr/bin/env python3
"""Add LingBot-VA action_config segments to a LeRobot dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro


@dataclass
class Args:
    dataset_root: Path = Path("/data/data_realworld/lerobot_export_dataset/local/insert-peg-cylinder-realmachine")
    default_text: str = "insert the peg into the matching cylinder hole"
    segment_frames: int = 243
    min_segment_frames: int = 30
    overlap_frames: int = 0
    overwrite: bool = False


def _tasks_to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _default_action_config(
    length: int,
    tasks: list[str],
    default_text: str,
    segment_frames: int,
    min_segment_frames: int,
    overlap_frames: int,
) -> list[dict[str, Any]]:
    text = default_text or (tasks[0] if tasks else "robot manipulation")
    length = int(length)
    segment_frames = int(segment_frames)
    min_segment_frames = max(1, int(min_segment_frames))
    overlap_frames = max(0, int(overlap_frames))

    if length <= 0:
        return []
    if segment_frames <= 0 or segment_frames >= length:
        return [{"start_frame": 0, "end_frame": length, "action_text": text}]

    overlap_frames = min(overlap_frames, segment_frames - 1)
    step = max(1, segment_frames - overlap_frames)
    segments: list[dict[str, Any]] = []
    start_frame = 0
    while start_frame < length:
        end_frame = min(length, start_frame + segment_frames)
        if segments and end_frame - start_frame < min_segment_frames:
            segments[-1]["end_frame"] = length
            break
        segments.append({"start_frame": int(start_frame), "end_frame": int(end_frame), "action_text": text})
        if end_frame >= length:
            break
        start_frame += step
    return segments


def _create_jsonl_from_parquet(args: Args, dataset_root: Path) -> Path:
    parquet_paths = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"episodes.jsonl and meta/episodes parquet files not found: {dataset_root / 'meta'}")

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to create episodes.jsonl from LeRobot v3 parquet metadata") from exc

    items: list[dict[str, Any]] = []
    for parquet_path in parquet_paths:
        dataframe = pd.read_parquet(parquet_path)
        for _, row in dataframe.iterrows():
            length = int(row["length"])
            tasks = _tasks_to_list(row.get("tasks"))
            items.append(
                {
                    "episode_index": int(row["episode_index"]),
                    "tasks": tasks,
                    "length": length,
                    "action_config": _default_action_config(
                        length,
                        tasks,
                        args.default_text,
                        args.segment_frames,
                        args.min_segment_frames,
                        args.overlap_frames,
                    ),
                }
            )

    items.sort(key=lambda item: int(item["episode_index"]))
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    episodes_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8",
    )
    return episodes_path


def main(args: Args) -> None:
    dataset_root = args.dataset_root.expanduser().resolve()
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        episodes_path = _create_jsonl_from_parquet(args, dataset_root)

    updated = []
    changed = 0
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if args.overwrite or "action_config" not in item:
            length = item.get("length")
            if length is None:
                raise KeyError(f"Episode lacks length: {item}")
            tasks = item.get("tasks") or []
            item["action_config"] = _default_action_config(
                int(length),
                tasks,
                args.default_text,
                args.segment_frames,
                args.min_segment_frames,
                args.overlap_frames,
            )
            changed += 1
        updated.append(json.dumps(item, ensure_ascii=False))

    backup_path = episodes_path.with_suffix(".jsonl.bak")
    if not backup_path.exists():
        backup_path.write_text(episodes_path.read_text(encoding="utf-8"), encoding="utf-8")
    episodes_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    print(f"updated {changed} episodes in {episodes_path}")
    print(f"backup: {backup_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
