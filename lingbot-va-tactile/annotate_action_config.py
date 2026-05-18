#!/usr/bin/env python3
"""Annotate LingBot-VA action_config segments for LeRobot datasets.

The script supports a practical human-in-the-loop workflow:

1. Export the current action_config into CSV.
2. Optionally generate automatic candidate segments from action/tactile signals.
3. Edit the CSV manually in a spreadsheet/text editor.
4. Import the CSV back into meta/episodes.jsonl.

Only meta/episodes.jsonl is modified, and a backup is created before writing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


CSV_COLUMNS = [
    "episode_index",
    "length",
    "segment_id",
    "start_frame",
    "end_frame",
    "action_text",
    "proposal",
    "notes",
]


PEG_INSERT_TEXTS = {
    "approach": "move the grasped peg toward the cylinder hole",
    "contact_align": "use tactile feedback to align the peg with the hole",
    "insert": "insert the peg into the cylinder hole",
    "stabilize": "stabilize the inserted peg and stop motion",
}


@dataclass
class SignalBundle:
    action_speed: np.ndarray
    gripper_delta: np.ndarray
    tactile_energy: np.ndarray | None
    tactile_delta: np.ndarray | None
    activity: np.ndarray


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def _write_jsonl(path: Path, items: list[dict[str, Any]], create_backup: bool = True) -> Path | None:
    backup_path = None
    if create_backup and path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(f"{path.name}.bak_{timestamp}")
        shutil.copy2(path, backup_path)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return backup_path


def _episodes_path(dataset_root: Path) -> Path:
    path = dataset_root / "meta" / "episodes.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found: {path}")
    return path


def _default_segments(length: int, text: str, segment_frames: int, min_segment_frames: int) -> list[dict[str, Any]]:
    if length <= 0:
        return []
    if segment_frames <= 0 or segment_frames >= length:
        return [{"start_frame": 0, "end_frame": length, "action_text": text, "proposal": "single"}]

    segments = []
    start = 0
    while start < length:
        end = min(length, start + segment_frames)
        if segments and end - start < min_segment_frames:
            segments[-1]["end_frame"] = length
            break
        segments.append(
            {
                "start_frame": int(start),
                "end_frame": int(end),
                "action_text": text,
                "proposal": "fixed",
            }
        )
        if end >= length:
            break
        start = end
    return segments


def export_csv(dataset_root: Path, output_csv: Path, default_text: str, segment_frames: int, min_segment_frames: int) -> None:
    episodes = _read_jsonl(_episodes_path(dataset_root))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            length = int(episode["length"])
            action_config = episode.get("action_config")
            if not action_config:
                text = default_text or (episode.get("tasks") or ["robot manipulation"])[0]
                action_config = _default_segments(length, text, segment_frames, min_segment_frames)
            for segment_id, segment in enumerate(action_config):
                writer.writerow(
                    {
                        "episode_index": episode_index,
                        "length": length,
                        "segment_id": segment_id,
                        "start_frame": int(segment["start_frame"]),
                        "end_frame": int(segment["end_frame"]),
                        "action_text": segment.get("action_text", default_text),
                        "proposal": segment.get("proposal", "existing"),
                        "notes": "",
                    }
                )
    print(f"[OK] exported annotation CSV: {output_csv}")


def _find_episode_parquet(dataset_root: Path, episode_index: int) -> Path:
    candidate = dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    if candidate.exists():
        return candidate
    matches = sorted((dataset_root / "data").glob(f"chunk-*/episode_{episode_index:06d}.parquet"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Parquet for episode {episode_index} not found under {dataset_root / 'data'}")


def _flatten_numeric(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object:
        parts = [_flatten_numeric(item) for item in array.tolist()]
        if not parts:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    return array.astype(np.float32, copy=False).reshape(-1)


def _stack_array_column(series: Any) -> np.ndarray:
    arrays = []
    for value in series:
        arrays.append(_flatten_numeric(value))
    if not arrays:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(arrays, axis=0)


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size == 0:
        return x.astype(np.float32)
    window = max(1, int(window))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def _robust_norm(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    q10, q90 = np.quantile(x, [0.10, 0.90])
    denom = float(q90 - q10)
    if denom < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - q10) / denom, 0.0, 1.0).astype(np.float32)


def _first_true_run(mask: np.ndarray, min_run: int) -> int | None:
    count = 0
    for idx, value in enumerate(mask):
        count = count + 1 if value else 0
        if count >= min_run:
            return idx - min_run + 1
    return None


def _last_true_run(mask: np.ndarray, min_run: int) -> int | None:
    count = 0
    for rev_idx, value in enumerate(mask[::-1]):
        count = count + 1 if value else 0
        if count >= min_run:
            return len(mask) - (rev_idx - min_run + 1) - 1
    return None


def _load_signals(dataset_root: Path, episode_index: int, smoothing_window: int) -> SignalBundle:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for auto-csv signal extraction") from exc

    parquet_path = _find_episode_parquet(dataset_root, episode_index)
    columns = ["action"]
    try:
        dataframe = pd.read_parquet(parquet_path, columns=columns + ["observation.tactile"])
        has_tactile = True
    except Exception:
        dataframe = pd.read_parquet(parquet_path, columns=columns)
        has_tactile = False

    action = _stack_array_column(dataframe["action"])
    if action.shape[1] < 8:
        raise ValueError(f"Expected action dimension >= 8 for episode {episode_index}, got {action.shape}")

    pos = action[:, :3]
    gripper = action[:, 7]
    pos_delta = np.diff(pos, axis=0, prepend=pos[:1])
    action_speed = _smooth(np.linalg.norm(pos_delta, axis=1), smoothing_window)
    gripper_delta = _smooth(np.abs(np.diff(gripper, prepend=gripper[:1])), smoothing_window)

    tactile_energy = None
    tactile_delta = None
    if has_tactile:
        tactile = _stack_array_column(dataframe["observation.tactile"])
        if tactile.size:
            tactile_energy = _smooth(np.percentile(tactile, 95, axis=1), smoothing_window)
            tactile_delta = _smooth(np.abs(np.diff(tactile_energy, prepend=tactile_energy[:1])), smoothing_window)

    parts = [_robust_norm(action_speed), _robust_norm(gripper_delta)]
    if tactile_energy is not None:
        parts.append(_robust_norm(tactile_energy))
    if tactile_delta is not None:
        parts.append(_robust_norm(tactile_delta))
    activity = np.clip(np.mean(np.stack(parts, axis=0), axis=0), 0.0, 1.0)
    activity = _smooth(activity, smoothing_window)

    return SignalBundle(
        action_speed=action_speed,
        gripper_delta=gripper_delta,
        tactile_energy=tactile_energy,
        tactile_delta=tactile_delta,
        activity=activity,
    )


def _valid_boundary(boundary: int, boundaries: list[int], length: int, min_segment_frames: int) -> bool:
    if boundary <= 0 or boundary >= length:
        return False
    all_boundaries = sorted(boundaries + [boundary])
    return all(
        all_boundaries[i + 1] - all_boundaries[i] >= min_segment_frames
        for i in range(len(all_boundaries) - 1)
    )


def _repair_boundaries(boundaries: list[int], length: int, min_segment_frames: int) -> list[int]:
    out = [0]
    for boundary in sorted(set(int(v) for v in boundaries if 0 < int(v) < length)):
        if boundary - out[-1] >= min_segment_frames:
            out.append(boundary)
    if length - out[-1] < min_segment_frames and len(out) > 1:
        out.pop()
    out.append(length)
    return out


def _segments_from_boundaries(boundaries: list[int], texts: list[str], proposal: str) -> list[dict[str, Any]]:
    segments = []
    for idx in range(len(boundaries) - 1):
        segments.append(
            {
                "start_frame": int(boundaries[idx]),
                "end_frame": int(boundaries[idx + 1]),
                "action_text": texts[min(idx, len(texts) - 1)],
                "proposal": proposal,
            }
        )
    return segments


def _propose_fixed(length: int, default_text: str, segment_frames: int, min_segment_frames: int) -> list[dict[str, Any]]:
    return _default_segments(length, default_text, segment_frames, min_segment_frames)


def _propose_signal_segments(
    signals: SignalBundle,
    length: int,
    default_text: str,
    max_segments: int,
    min_segment_frames: int,
) -> list[dict[str, Any]]:
    if max_segments <= 1:
        return _segments_from_boundaries([0, length], [default_text], "signals")

    activity = signals.activity
    candidate_scores = []
    for idx in range(min_segment_frames, max(min_segment_frames, length - min_segment_frames)):
        left = activity[max(0, idx - min_segment_frames // 2) : idx]
        right = activity[idx : min(length, idx + min_segment_frames // 2)]
        if left.size == 0 or right.size == 0:
            continue
        score = abs(float(right.mean() - left.mean())) + float(activity[idx])
        if signals.tactile_delta is not None:
            score += float(_robust_norm(signals.tactile_delta)[idx])
        candidate_scores.append((score, idx))

    boundaries = [0, length]
    for _, boundary in sorted(candidate_scores, reverse=True):
        if len(boundaries) >= max_segments + 1:
            break
        if _valid_boundary(boundary, boundaries, length, min_segment_frames):
            boundaries.append(boundary)
    boundaries = _repair_boundaries(boundaries, length, min_segment_frames)
    return _segments_from_boundaries(boundaries, [default_text], "signals")


def _propose_peg_insert_segments(
    signals: SignalBundle,
    length: int,
    min_segment_frames: int,
    contact_quantile: float,
    min_event_run: int,
) -> list[dict[str, Any]]:
    activity = _robust_norm(signals.activity)
    motion_start = _first_true_run(activity > 0.15, min_event_run)
    motion_end = _last_true_run(activity > 0.12, min_event_run)

    contact_onset = None
    peak_contact = None
    if signals.tactile_energy is not None and signals.tactile_energy.size:
        tactile_norm = _robust_norm(signals.tactile_energy)
        threshold = float(np.quantile(tactile_norm, contact_quantile))
        threshold = max(0.20, threshold)
        contact_onset = _first_true_run(tactile_norm > threshold, min_event_run)
        peak_contact = int(np.argmax(tactile_norm))

    boundaries = [0, length]
    if motion_start is not None and motion_start >= min_segment_frames:
        boundaries.append(motion_start)
    if contact_onset is not None:
        boundaries.append(contact_onset)
    if peak_contact is not None and contact_onset is not None:
        insert_boundary = int((contact_onset + peak_contact) // 2)
        boundaries.append(insert_boundary)
    if motion_end is not None and length - motion_end >= min_segment_frames:
        boundaries.append(motion_end)

    boundaries = _repair_boundaries(boundaries, length, min_segment_frames)
    num_segments = len(boundaries) - 1
    if num_segments <= 1:
        texts = [PEG_INSERT_TEXTS["insert"]]
    elif num_segments == 2:
        texts = [PEG_INSERT_TEXTS["approach"], PEG_INSERT_TEXTS["insert"]]
    elif num_segments == 3:
        texts = [PEG_INSERT_TEXTS["approach"], PEG_INSERT_TEXTS["contact_align"], PEG_INSERT_TEXTS["insert"]]
    else:
        texts = [
            PEG_INSERT_TEXTS["approach"],
            PEG_INSERT_TEXTS["contact_align"],
            PEG_INSERT_TEXTS["insert"],
            PEG_INSERT_TEXTS["stabilize"],
        ]
    return _segments_from_boundaries(boundaries, texts, "peg_insert")


def _plot_signals(
    signals: SignalBundle,
    episode_index: int,
    segments: list[dict[str, Any]],
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skip signal plot")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
    x = np.arange(len(signals.activity))
    axes[0].plot(x, _robust_norm(signals.activity), label="activity", color="#222222")
    axes[0].set_title(f"Episode {episode_index} proposed segmentation")
    axes[1].plot(x, _robust_norm(signals.action_speed), label="action speed", color="#1f77b4")
    axes[2].plot(x, _robust_norm(signals.gripper_delta), label="gripper delta", color="#ff7f0e")
    if signals.tactile_energy is not None:
        axes[3].plot(x, _robust_norm(signals.tactile_energy), label="tactile energy", color="#2ca02c")
    if signals.tactile_delta is not None:
        axes[3].plot(x, _robust_norm(signals.tactile_delta), label="tactile delta", color="#d62728", alpha=0.75)

    for ax in axes:
        for segment in segments:
            ax.axvline(int(segment["start_frame"]), color="#9467bd", linestyle="--", alpha=0.7)
        ax.axvline(int(segments[-1]["end_frame"]), color="#9467bd", linestyle="--", alpha=0.7)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def auto_csv(args: argparse.Namespace) -> None:
    episodes = _read_jsonl(_episodes_path(args.dataset_root))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    plot_dir = args.plot_dir
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            length = int(episode["length"])
            if args.max_episodes > 0 and episode_index >= args.max_episodes:
                continue

            if args.auto_strategy == "fixed":
                segments = _propose_fixed(
                    length,
                    args.default_text,
                    args.segment_frames,
                    args.min_segment_frames,
                )
                signals = None
            else:
                signals = _load_signals(args.dataset_root, episode_index, args.smoothing_window)
                if args.auto_strategy == "signals":
                    segments = _propose_signal_segments(
                        signals,
                        length,
                        args.default_text,
                        args.max_segments,
                        args.min_segment_frames,
                    )
                elif args.auto_strategy == "peg_insert":
                    segments = _propose_peg_insert_segments(
                        signals,
                        length,
                        args.min_segment_frames,
                        args.contact_quantile,
                        args.min_event_run,
                    )
                else:
                    raise ValueError(f"Unknown auto strategy: {args.auto_strategy}")

            for segment_id, segment in enumerate(segments):
                writer.writerow(
                    {
                        "episode_index": episode_index,
                        "length": length,
                        "segment_id": segment_id,
                        "start_frame": int(segment["start_frame"]),
                        "end_frame": int(segment["end_frame"]),
                        "action_text": segment["action_text"],
                        "proposal": segment.get("proposal", args.auto_strategy),
                        "notes": "",
                    }
                )
            if plot_dir is not None and signals is not None:
                _plot_signals(
                    signals,
                    episode_index,
                    segments,
                    plot_dir / f"episode_{episode_index:06d}_signals.png",
                )
            print(f"[OK] episode {episode_index}: {len(segments)} segment(s)")

    print(f"[OK] wrote auto annotation CSV: {args.output_csv}")
    if plot_dir is not None:
        print(f"[OK] signal plots: {plot_dir}")


def _read_annotation_csv(path: Path) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [column for column in CSV_COLUMNS[:6] if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        for row in reader:
            text = (row.get("action_text") or "").strip()
            if not text:
                continue
            episode_index = int(row["episode_index"])
            grouped.setdefault(episode_index, []).append(
                {
                    "start_frame": int(float(row["start_frame"])),
                    "end_frame": int(float(row["end_frame"])),
                    "action_text": text,
                }
            )
    return grouped


def _validate_segments(segments: list[dict[str, Any]], length: int, episode_index: int) -> list[dict[str, Any]]:
    if not segments:
        raise ValueError(f"Episode {episode_index} has no annotated segments")
    segments = sorted(segments, key=lambda item: (item["start_frame"], item["end_frame"]))
    if segments[0]["start_frame"] != 0:
        raise ValueError(f"Episode {episode_index}: first segment must start at 0")
    if segments[-1]["end_frame"] != length:
        raise ValueError(f"Episode {episode_index}: last segment must end at length={length}")
    prev_end = 0
    out = []
    for idx, segment in enumerate(segments):
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        if start != prev_end:
            raise ValueError(
                f"Episode {episode_index}: gap/overlap before segment {idx}; "
                f"expected start={prev_end}, got {start}"
            )
        if end <= start:
            raise ValueError(f"Episode {episode_index}: segment {idx} has non-positive length")
        if end > length:
            raise ValueError(f"Episode {episode_index}: segment {idx} end exceeds length={length}")
        out.append(
            {
                "start_frame": start,
                "end_frame": end,
                "action_text": segment["action_text"],
            }
        )
        prev_end = end
    return out


def import_csv(dataset_root: Path, input_csv: Path, only_episodes_in_csv: bool, dry_run: bool) -> None:
    episodes_path = _episodes_path(dataset_root)
    episodes = _read_jsonl(episodes_path)
    annotations = _read_annotation_csv(input_csv)
    changed = 0

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        if episode_index not in annotations:
            if only_episodes_in_csv:
                continue
            raise ValueError(f"Episode {episode_index} missing from annotation CSV")
        length = int(episode["length"])
        episode["action_config"] = _validate_segments(
            annotations[episode_index],
            length,
            episode_index,
        )
        changed += 1

    if dry_run:
        print(f"[DRY-RUN] validated {changed} episode(s); no file written")
        return

    backup_path = _write_jsonl(episodes_path, episodes, create_backup=True)
    print(f"[OK] updated {changed} episode(s) in {episodes_path}")
    if backup_path is not None:
        print(f"[OK] backup: {backup_path}")


def summary(dataset_root: Path) -> None:
    episodes = _read_jsonl(_episodes_path(dataset_root))
    num_segments = 0
    lengths = []
    text_counts: dict[str, int] = {}
    bad = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        length = int(episode["length"])
        segments = episode.get("action_config") or []
        num_segments += len(segments)
        for segment in segments:
            start = int(segment["start_frame"])
            end = int(segment["end_frame"])
            lengths.append(end - start)
            text = segment.get("action_text", "")
            text_counts[text] = text_counts.get(text, 0) + 1
        try:
            _validate_segments(
                [
                    {
                        "start_frame": int(segment["start_frame"]),
                        "end_frame": int(segment["end_frame"]),
                        "action_text": segment.get("action_text", ""),
                    }
                    for segment in segments
                ],
                length,
                episode_index,
            )
        except Exception as exc:
            bad.append((episode_index, str(exc)))

    print(f"episodes: {len(episodes)}")
    print(f"segments: {num_segments}")
    if lengths:
        print(
            "segment frames: "
            f"min={min(lengths)} median={int(np.median(lengths))} "
            f"mean={np.mean(lengths):.1f} max={max(lengths)}"
        )
    print("top action_text:")
    for text, count in sorted(text_counts.items(), key=lambda item: item[1], reverse=True)[:20]:
        print(f"  {count:5d}  {text}")
    if bad:
        print("[WARN] invalid episodes:")
        for episode_index, reason in bad[:20]:
            print(f"  episode {episode_index}: {reason}")
    else:
        print("[OK] all action_config segments are contiguous and length-matched")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual/automatic action_config annotation helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_dataset = argparse.ArgumentParser(add_help=False)
    common_dataset.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/data_realworld/lerobot_export_dataset/local/insert-peg-cylinder-realmachine"),
    )

    export_parser = subparsers.add_parser("export-csv", parents=[common_dataset])
    export_parser.add_argument("--output-csv", type=Path, required=True)
    export_parser.add_argument("--default-text", type=str, default="insert the peg into the cylinder hole")
    export_parser.add_argument("--segment-frames", type=int, default=243)
    export_parser.add_argument("--min-segment-frames", type=int, default=30)

    auto_parser = subparsers.add_parser("auto-csv", parents=[common_dataset])
    auto_parser.add_argument("--output-csv", type=Path, required=True)
    auto_parser.add_argument("--plot-dir", type=Path, default=None)
    auto_parser.add_argument("--auto-strategy", choices=["fixed", "signals", "peg_insert"], default="peg_insert")
    auto_parser.add_argument("--default-text", type=str, default="insert the peg into the cylinder hole")
    auto_parser.add_argument("--segment-frames", type=int, default=243)
    auto_parser.add_argument("--min-segment-frames", type=int, default=60)
    auto_parser.add_argument("--max-segments", type=int, default=4)
    auto_parser.add_argument("--smoothing-window", type=int, default=15)
    auto_parser.add_argument("--contact-quantile", type=float, default=0.65)
    auto_parser.add_argument("--min-event-run", type=int, default=12)
    auto_parser.add_argument("--max-episodes", type=int, default=0)

    import_parser = subparsers.add_parser("import-csv", parents=[common_dataset])
    import_parser.add_argument("--input-csv", type=Path, required=True)
    import_parser.add_argument("--only-episodes-in-csv", action="store_true")
    import_parser.add_argument("--dry-run", action="store_true")

    subparsers.add_parser("summary", parents=[common_dataset])
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.dataset_root = args.dataset_root.expanduser().resolve()

    if args.command == "export-csv":
        export_csv(
            args.dataset_root,
            args.output_csv.expanduser().resolve(),
            args.default_text,
            args.segment_frames,
            args.min_segment_frames,
        )
    elif args.command == "auto-csv":
        args.output_csv = args.output_csv.expanduser().resolve()
        if args.plot_dir is not None:
            args.plot_dir = args.plot_dir.expanduser().resolve()
        auto_csv(args)
    elif args.command == "import-csv":
        import_csv(
            args.dataset_root,
            args.input_csv.expanduser().resolve(),
            args.only_episodes_in_csv,
            args.dry_run,
        )
    elif args.command == "summary":
        summary(args.dataset_root)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
