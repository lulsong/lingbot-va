#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Downsample raw PIKA tactile episodes before LeRobot conversion.

The output keeps the raw-dataset contract expected by convert_to_lerobot.py:

  input_root/
    episode1/teleop_log.csv
    episode1/<paths referenced by CSV>
    raw_action_config.json or action_config_annotations.json, if present

It creates a new raw dataset root instead of editing the original data.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro


FRAME_PATH_COLUMNS = [
    "filepath_color",
    "filepath_side_rgb",
    "filepath_fisheye",
    "filepath_tactile_gripper",
]

ACTION_CONFIG_FILENAMES = ("raw_action_config.json", "action_config_annotations.json")
REPORT_FILENAME = "downsample_report.json"


@dataclass
class Args:
    input_root: Path = Path("/data/Datasets/PIKA_real_original/make_coffee")
    output_root: Path = Path("/data/Datasets/PIKA_real_original/make_coffee_stride2")
    stride: int = 2
    output_fps: float = 30.0
    offset: int = 0
    keep_last: bool = True
    min_frames_per_episode: int = 2
    timestamp_mode: str = "compact"
    link_mode: str = "hardlink"
    overwrite: bool = False
    max_episodes: int = 0
    skip_missing_assets: bool = False
    dry_run: bool = False


def _sorted_episode_dirs(root: Path) -> list[Path]:
    episodes = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("episode")]

    def episode_key(path: Path) -> tuple[int, str]:
        suffix = path.name.replace("episode", "")
        return (int(suffix), path.name) if suffix.isdigit() else (10**9, path.name)

    return sorted(episodes, key=episode_key)


def _read_csv(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV header is missing: {csv_path}")
        return list(reader.fieldnames), list(reader)


def _write_csv(csv_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _validate_args(args: Args) -> None:
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")
    if args.output_fps <= 0:
        raise ValueError("--output-fps must be > 0")
    if args.min_frames_per_episode < 1:
        raise ValueError("--min-frames-per-episode must be >= 1")
    if args.timestamp_mode not in {"compact", "preserve"}:
        raise ValueError("--timestamp-mode must be 'compact' or 'preserve'")
    if args.link_mode not in {"hardlink", "copy", "symlink"}:
        raise ValueError("--link-mode must be one of: hardlink, copy, symlink")


def _is_relative_safe(path_text: str) -> bool:
    path = Path(path_text)
    return not path.is_absolute() and ".." not in path.parts


def _select_indices(length: int, stride: int, offset: int, keep_last: bool) -> list[int]:
    if length <= 0:
        return []
    start = min(offset, length - 1)
    selected = list(range(start, length, stride))
    if keep_last and selected[-1] != length - 1:
        selected.append(length - 1)
    return selected


def _copy_or_link(src: Path, dst: Path, link_mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"

    if link_mode == "copy":
        shutil.copy2(src, dst)
        return "copy"

    if link_mode == "symlink":
        os.symlink(src, dst)
        return "symlink"

    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy_fallback"


def _copy_selected_assets(
    src_episode: Path,
    dst_episode: Path,
    rows: list[dict[str, str]],
    link_mode: str,
    skip_missing_assets: bool,
) -> tuple[dict[str, int], list[str]]:
    method_counts: dict[str, int] = {}
    missing_assets: list[str] = []
    copied: set[str] = set()

    for row in rows:
        for column in FRAME_PATH_COLUMNS:
            value = row.get(column, "")
            if not value:
                missing_assets.append(f"{column}:<empty>")
                continue
            if not _is_relative_safe(value):
                raise ValueError(f"Unsafe or absolute asset path in {column}: {value!r}")
            if value in copied:
                continue

            src = src_episode / value
            dst = dst_episode / value
            if not src.exists():
                missing_assets.append(f"{column}:{value}")
                if skip_missing_assets:
                    continue
                raise FileNotFoundError(f"Referenced asset does not exist: {src}")

            method = _copy_or_link(src, dst, link_mode)
            method_counts[method] = method_counts.get(method, 0) + 1
            copied.add(value)

    return method_counts, missing_assets


def _remap_timestamp(row: dict[str, str], new_index: int, first_timestamp: float, args: Args) -> dict[str, str]:
    out = dict(row)
    if "timestamp" not in out or args.timestamp_mode == "preserve":
        return out
    out["timestamp"] = f"{first_timestamp + new_index / args.output_fps:.9f}"
    return out


def _selected_lookup(selected_indices: list[int]) -> dict[int, int]:
    return {old_index: new_index for new_index, old_index in enumerate(selected_indices)}


def _remap_segments_by_selection(
    segments: list[dict[str, Any]],
    selected_indices: list[int],
) -> list[dict[str, Any]]:
    if not segments:
        return []

    assignments: list[str | None] = [None] * len(selected_indices)
    sorted_segments = sorted(
        segments,
        key=lambda item: (int(item.get("start_frame", 0)), int(item.get("end_frame", 0))),
    )
    for segment in sorted_segments:
        start = int(segment.get("start_frame", 0))
        end = int(segment.get("end_frame", start))
        text = str(segment.get("action_text", "robot manipulation"))
        for new_index, old_index in enumerate(selected_indices):
            if start <= old_index < end:
                assignments[new_index] = text

    remapped: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(assignments):
        text = assignments[cursor]
        if text is None:
            cursor += 1
            continue
        end = cursor + 1
        while end < len(assignments) and assignments[end] == text:
            end += 1
        remapped.append(
            {
                "start_frame": cursor,
                "end_frame": end,
                "action_text": text,
            }
        )
        cursor = end
    return remapped


def _normalize_segments(value: Any) -> tuple[list[dict[str, Any]], str]:
    if isinstance(value, dict) and isinstance(value.get("segments"), list):
        return value["segments"], "wrapped"
    if isinstance(value, list):
        return value, "list"
    return [], "unknown"


def _wrap_segments(segments: list[dict[str, Any]], original: Any, mode: str) -> Any:
    if mode == "wrapped" and isinstance(original, dict):
        out = dict(original)
        out["segments"] = segments
        return out
    return segments


def _remap_action_config_file(
    src_path: Path,
    dst_path: Path,
    selections: dict[str, list[int]],
) -> int:
    with src_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    episodes_obj = data.get("episodes", data) if isinstance(data, dict) else data
    if not isinstance(episodes_obj, dict):
        shutil.copy2(src_path, dst_path)
        return 0

    changed = 0
    new_episodes: dict[str, Any] = {}
    for episode_name, original_value in episodes_obj.items():
        if episode_name not in selections:
            continue
        segments, mode = _normalize_segments(original_value)
        remapped = _remap_segments_by_selection(segments, selections[episode_name])
        new_episodes[episode_name] = _wrap_segments(remapped, original_value, mode)
        changed += 1

    if isinstance(data, dict) and "episodes" in data:
        out = dict(data)
        out["episodes"] = new_episodes
    else:
        out = new_episodes

    dst_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed


def _remap_annotation_items(items: list[dict[str, Any]], selected_indices: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    lookup = _selected_lookup(selected_indices)
    for item in items:
        if "start_idx" not in item or "end_idx" not in item:
            out.append(dict(item))
            continue
        start = int(item["start_idx"])
        end = int(item["end_idx"])
        new_indices = [lookup[old_idx] for old_idx in selected_indices if start <= old_idx <= end]
        if not new_indices:
            continue
        new_item = dict(item)
        new_item["start_idx"] = min(new_indices)
        new_item["end_idx"] = max(new_indices)
        out.append(new_item)
    return out


def _remap_annotations_file(
    src_path: Path,
    dst_path: Path,
    selections: dict[str, list[int]],
) -> int:
    with src_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        shutil.copy2(src_path, dst_path)
        return 0

    changed = 0
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key in selections and isinstance(value, list):
            out[key] = _remap_annotation_items(value, selections[key])
            changed += 1
        else:
            out[key] = value

    dst_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed


def _copy_root_sidecars(input_root: Path, output_root: Path, selections: dict[str, list[int]]) -> dict[str, Any]:
    copied: list[str] = []
    remapped: dict[str, int] = {}

    for path in sorted(input_root.iterdir()):
        if not path.is_file():
            continue
        dst = output_root / path.name
        if path.name in ACTION_CONFIG_FILENAMES:
            remapped[path.name] = _remap_action_config_file(path, dst, selections)
        elif path.name == "annotations.json":
            remapped[path.name] = _remap_annotations_file(path, dst, selections)
        elif path.name != REPORT_FILENAME:
            shutil.copy2(path, dst)
            copied.append(path.name)

    return {"copied": copied, "remapped": remapped}


def downsample(args: Args) -> None:
    _validate_args(args)
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if input_root == output_root:
        raise ValueError("output_root must be different from input_root")
    if input_root in output_root.parents:
        raise ValueError("output_root must not be inside input_root")

    episode_dirs = _sorted_episode_dirs(input_root)
    if args.max_episodes > 0:
        episode_dirs = episode_dirs[: args.max_episodes]
    if not episode_dirs:
        raise RuntimeError(f"No episode directories found under {input_root}")

    if output_root.exists() and not args.dry_run:
        if args.overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(f"Output root already exists: {output_root}; pass --overwrite to replace it")

    report_episodes: list[dict[str, Any]] = []
    selections: dict[str, list[int]] = {}
    total_input_rows = 0
    total_output_rows = 0
    total_skipped_episodes = 0
    link_method_counts: dict[str, int] = {}

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    for episode_dir in episode_dirs:
        csv_path = episode_dir / "teleop_log.csv"
        if not csv_path.exists():
            total_skipped_episodes += 1
            report_episodes.append({"episode": episode_dir.name, "status": "skipped", "reason": "missing teleop_log.csv"})
            continue

        fieldnames, rows = _read_csv(csv_path)
        total_input_rows += len(rows)
        selected_indices = _select_indices(len(rows), args.stride, args.offset, args.keep_last)
        if len(selected_indices) < args.min_frames_per_episode:
            total_skipped_episodes += 1
            report_episodes.append(
                {
                    "episode": episode_dir.name,
                    "status": "skipped",
                    "reason": f"selected frames < min_frames_per_episode ({len(selected_indices)} < {args.min_frames_per_episode})",
                    "input_rows": len(rows),
                }
            )
            continue

        selected_rows: list[dict[str, str]] = []
        first_timestamp = 0.0
        if rows and rows[selected_indices[0]].get("timestamp") not in (None, ""):
            try:
                first_timestamp = float(rows[selected_indices[0]]["timestamp"])
            except ValueError:
                first_timestamp = 0.0

        for new_index, old_index in enumerate(selected_indices):
            selected_rows.append(_remap_timestamp(rows[old_index], new_index, first_timestamp, args))

        selections[episode_dir.name] = selected_indices
        total_output_rows += len(selected_rows)

        episode_report: dict[str, Any] = {
            "episode": episode_dir.name,
            "status": "ok",
            "input_rows": len(rows),
            "output_rows": len(selected_rows),
            "selected_first_old_index": selected_indices[0],
            "selected_last_old_index": selected_indices[-1],
        }

        if not args.dry_run:
            dst_episode = output_root / episode_dir.name
            _write_csv(dst_episode / "teleop_log.csv", fieldnames, selected_rows)
            method_counts, missing_assets = _copy_selected_assets(
                episode_dir,
                dst_episode,
                selected_rows,
                args.link_mode,
                args.skip_missing_assets,
            )
            for method, count in method_counts.items():
                link_method_counts[method] = link_method_counts.get(method, 0) + count
            if missing_assets:
                episode_report["missing_assets"] = missing_assets

        report_episodes.append(episode_report)

    sidecars: dict[str, Any] = {}
    if not args.dry_run:
        sidecars = _copy_root_sidecars(input_root, output_root, selections)

    report = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "stride": args.stride,
        "offset": args.offset,
        "keep_last": args.keep_last,
        "output_fps": args.output_fps,
        "timestamp_mode": args.timestamp_mode,
        "link_mode": args.link_mode,
        "episodes_seen": len(episode_dirs),
        "episodes_written": len(selections),
        "episodes_skipped": total_skipped_episodes,
        "input_rows": total_input_rows,
        "output_rows": total_output_rows,
        "effective_row_ratio": (total_output_rows / total_input_rows) if total_input_rows else 0.0,
        "asset_methods": link_method_counts,
        "sidecars": sidecars,
        "episodes": report_episodes,
    }

    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    report_path = output_root / REPORT_FILENAME
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=" * 68)
    print("raw dataset downsample complete")
    print(f"input root       : {input_root}")
    print(f"output root      : {output_root}")
    print(f"episodes written : {len(selections)}")
    print(f"frames           : {total_input_rows} -> {total_output_rows}")
    print(f"timestamp mode   : {args.timestamp_mode} @ {args.output_fps:g} fps")
    print(f"report           : {report_path}")
    print("=" * 68)

    if not selections:
        raise RuntimeError("No episodes were written; check input data and min_frames_per_episode")


def main() -> None:
    downsample(tyro.cli(Args))


if __name__ == "__main__":
    main()
