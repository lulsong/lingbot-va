#!/usr/bin/env python3
"""Visualize raw PIKA robot episodes with camera, depth, tactile, and state panels."""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_ROOT = Path("/data/Datasets/PIKA_real_original/surface_slide")
DEFAULT_CSV_NAME = "teleop_log.csv"

CAMERA_COLUMNS = {
    "front": "filepath_color",
    "fisheye": "filepath_fisheye",
    "side": "filepath_side_rgb",
}
DEPTH_COLUMN = "filepath_depth"
TACTILE_COLUMNS = (
    "filepath_tactile_gripper",
    "filepath_tactile_sense",
)

PANEL_W = 520
PANEL_H = 390
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_COLOR = (235, 235, 235)
MUTED = (145, 155, 165)
BG = (8, 10, 12)
PANEL_BG = (18, 22, 27)

# Match /home/tujian/Projects/pika_sdk/PIKA_RM65B_data_acquisition/main_teleop.py.
TACTILE_HEATMAP_VMIN = 0.0
TACTILE_HEATMAP_VMAX = 0.33
TACTILE_HEATMAP_BLACK_THRESHOLD = 0.002
TACTILE_TARGET_KEYS = ("0", "1")
TACTILE_PIXEL_SIZE = 20
TACTILE_LABEL_AREA_H = 110


@dataclass(frozen=True)
class FrameRecord:
    index: int
    row: dict[str, str]


@dataclass(frozen=True)
class KeyframeSample:
    selected_index: int
    record: FrameRecord
    tactile_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one raw PIKA_real_original episode. By default it opens "
            "an OpenCV playback window; use --output with --no-display on servers."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Dataset root that contains episode folders. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--episode",
        required=True,
        help="Episode name, number, or full path, e.g. episode9, 9, /data/.../episode9.",
    )
    parser.add_argument(
        "--csv-name",
        default=DEFAULT_CSV_NAME,
        help=f"CSV filename inside episode folder. Default: {DEFAULT_CSV_NAME}",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index in CSV rows.")
    parser.add_argument("--end", type=int, default=None, help="Exclusive end frame index.")
    parser.add_argument("--stride", type=int, default=1, help="Visualize every Nth CSV row.")
    parser.add_argument("--fps", type=float, default=15.0, help="Playback/output FPS.")
    parser.add_argument("--panel-width", type=int, default=PANEL_W)
    parser.add_argument("--panel-height", type=int, default=PANEL_H)
    parser.add_argument("--output", type=Path, default=None, help="Optional output video path.")
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=None,
        help="Optional directory for academic-style static figures.",
    )
    parser.add_argument(
        "--figure-count",
        type=int,
        default=6,
        help="Number of keyframes in the montage figure.",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=300,
        help="DPI for saved figures.",
    )
    parser.add_argument(
        "--figure-format",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="Figure file formats to save.",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="OpenCV VideoWriter fourcc. Use MJPG for .avi if mp4v is slow.",
    )
    parser.add_argument("--no-display", action="store_true", help="Do not open OpenCV window.")
    parser.add_argument("--loop", action="store_true", help="Loop playback until q/Esc.")
    parser.add_argument(
        "--tactile-vmax",
        type=float,
        default=TACTILE_HEATMAP_VMAX,
        help="Tactile heatmap max. Default matches main_teleop.py: 0.33.",
    )
    parser.add_argument(
        "--tactile-threshold",
        type=float,
        default=TACTILE_HEATMAP_BLACK_THRESHOLD,
        help="Values below this are drawn black. Default matches main_teleop.py: 0.002.",
    )
    return parser.parse_args()


def resolve_episode(root: Path, episode: str) -> Path:
    candidate = Path(episode).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate.resolve()
    if episode.isdigit():
        episode = f"episode{episode}"
    return (root.expanduser() / episode).resolve()


def load_records(csv_path: Path, start: int, end: int | None, stride: int) -> list[FrameRecord]:
    if stride <= 0:
        raise ValueError("--stride must be positive")
    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    stop = len(rows) if end is None else min(end, len(rows))
    start = max(start, 0)
    return [FrameRecord(i, rows[i]) for i in range(start, stop, stride)]


def is_valid_path(value: str | None) -> bool:
    if value is None:
        return False
    value = value.strip()
    return bool(value) and value.upper() != "N/A"


def resolve_data_path(episode_dir: Path, value: str | None) -> Path | None:
    if not is_valid_path(value):
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return episode_dir / path


def make_panel(title: str, panel_w: int, panel_h: int, subtitle: str | None = None) -> np.ndarray:
    panel = np.full((panel_h, panel_w, 3), PANEL_BG, dtype=np.uint8)
    cv2.putText(panel, title, (12, 28), FONT, 0.72, FONT_COLOR, 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(panel, subtitle, (12, 54), FONT, 0.45, MUTED, 1, cv2.LINE_AA)
    return panel


def put_lines(
    image: np.ndarray,
    lines: Iterable[str],
    x: int,
    y: int,
    scale: float = 0.48,
    color: tuple[int, int, int] = FONT_COLOR,
    step: int = 23,
) -> None:
    for line in lines:
        cv2.putText(image, str(line), (x, y), FONT, scale, color, 1, cv2.LINE_AA)
        y += step


def letterbox(image: np.ndarray, width: int, height: int, fill: tuple[int, int, int] = BG) -> np.ndarray:
    canvas = np.full((height, width, 3), fill, dtype=np.uint8)
    if image is None or image.size == 0:
        return canvas
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_bgr_image(episode_dir: Path, row: dict[str, str], column: str) -> np.ndarray | None:
    path = resolve_data_path(episode_dir, row.get(column))
    if path is None:
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def load_depth_rgb(episode_dir: Path, row: dict[str, str]) -> np.ndarray | None:
    path = resolve_data_path(episode_dir, row.get(DEPTH_COLUMN))
    if path is None:
        return None
    try:
        depth = np.load(path, allow_pickle=False).astype(np.float32)
    except Exception:
        return None
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return None
    vmax = np.percentile(finite, 98)
    if not math.isfinite(float(vmax)) or vmax <= 0:
        vmax = float(np.max(finite))
    vmax = max(float(vmax), 1e-6)
    normalized = np.clip(depth, 0, vmax) / vmax
    depth_u8 = (normalized * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    color[~np.isfinite(depth)] = (0, 0, 0)
    return bgr_to_rgb(color)


def image_panel(episode_dir: Path, row: dict[str, str], column: str, title: str, panel_w: int, panel_h: int) -> np.ndarray:
    panel = make_panel(title, panel_w, panel_h)
    path = resolve_data_path(episode_dir, row.get(column))
    if path is None:
        put_lines(panel, ["N/A"], 12, panel_h // 2, color=(80, 80, 80), scale=0.7)
        return panel
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        put_lines(panel, [f"Missing/unreadable:", path.name], 12, panel_h // 2, color=(80, 80, 255))
        return panel
    content = letterbox(image, panel_w, panel_h - 64)
    panel[64:, :] = content
    cv2.putText(panel, path.name, (12, panel_h - 12), FONT, 0.38, MUTED, 1, cv2.LINE_AA)
    return panel


def depth_panel(episode_dir: Path, row: dict[str, str], panel_w: int, panel_h: int) -> np.ndarray:
    panel = make_panel("depth", panel_w, panel_h)
    path = resolve_data_path(episode_dir, row.get(DEPTH_COLUMN))
    if path is None:
        put_lines(panel, ["N/A"], 12, panel_h // 2, color=(80, 80, 80), scale=0.7)
        return panel
    try:
        depth = np.load(path, allow_pickle=False).astype(np.float32)
    except Exception as exc:
        put_lines(panel, ["Depth load failed:", str(exc)[:64]], 12, panel_h // 2, color=(80, 80, 255))
        return panel
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        put_lines(panel, ["Depth has no finite values"], 12, panel_h // 2, color=(80, 80, 255))
        return panel
    vmax = np.percentile(finite, 98)
    if not math.isfinite(float(vmax)) or vmax <= 0:
        vmax = float(np.max(finite))
    vmax = max(float(vmax), 1e-6)
    normalized = np.clip(depth, 0, vmax) / vmax
    depth_u8 = (normalized * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    color[~np.isfinite(depth)] = (0, 0, 0)
    panel[64:, :] = letterbox(color, panel_w, panel_h - 64)
    cv2.putText(panel, f"p98={vmax:.1f}  {path.name}", (12, panel_h - 12), FONT, 0.38, MUTED, 1, cv2.LINE_AA)
    return panel


def load_tactile_npz(episode_dir: Path, row: dict[str, str]) -> tuple[dict[str, np.ndarray] | None, str]:
    for column in TACTILE_COLUMNS:
        path = resolve_data_path(episode_dir, row.get(column))
        if path is None:
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                tactile = {str(key): np.asarray(data[key], dtype=np.float32) for key in data.files}
            return tactile, path.name
        except Exception:
            continue
    return None, "N/A"


def tactile_sensor_panel(
    matrix: np.ndarray | None,
    key: str,
    vmax: float,
    black_threshold: float,
) -> np.ndarray:
    h, w = (32, 58)
    if matrix is not None:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape != (h, w):
            try:
                matrix = matrix.reshape((h, w))
            except Exception:
                matrix = None

    # Match teleop_utils._create_tactile_vis: rotate each split clockwise before drawing.
    if matrix is not None:
        matrix = cv2.rotate(matrix, cv2.ROTATE_90_CLOCKWISE)
        h, w = w, h

    image_area_h = h * TACTILE_PIXEL_SIZE
    target_w = w * TACTILE_PIXEL_SIZE
    target_h = h * TACTILE_PIXEL_SIZE
    panel = np.zeros((image_area_h + TACTILE_LABEL_AREA_H, target_w, 3), dtype=np.uint8)

    if matrix is None or matrix.size == 0:
        cv2.putText(panel, "NO DATA", (12, max(32, image_area_h // 2)), FONT, 0.7, (70, 70, 70), 2, cv2.LINE_AA)
        force = 0.0
        peak = 0.0
    else:
        force = float(np.nansum(matrix))
        peak = float(np.nanmax(matrix))
        matrix_clipped = np.clip(matrix, TACTILE_HEATMAP_VMIN, vmax)
        denom = max(float(vmax - TACTILE_HEATMAP_VMIN), 1e-8)
        norm_img = ((matrix_clipped - TACTILE_HEATMAP_VMIN) / denom * 255.0).astype(np.uint8)
        color_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
        color_img[matrix < black_threshold] = (0, 0, 0)
        scaled = cv2.resize(color_img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        panel[:target_h, :target_w] = scaled

    label_y = image_area_h + 34
    cv2.putText(panel, f"sensor {key}", (18, label_y), FONT, 1.0, FONT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(panel, f"force sum: {force:.4f}", (18, label_y + 34), FONT, 0.72, FONT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(panel, f"max: {peak:.5f}", (18, label_y + 66), FONT, 0.72, FONT_COLOR, 2, cv2.LINE_AA)
    return panel


def tactile_heatmap_bgr(
    matrix: np.ndarray | None,
    vmax: float,
    black_threshold: float,
    pixel_size: int = 8,
) -> np.ndarray | None:
    h, w = (32, 58)
    if matrix is None:
        return None
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (h, w):
        try:
            matrix = matrix.reshape((h, w))
        except Exception:
            return None
    matrix = cv2.rotate(matrix, cv2.ROTATE_90_CLOCKWISE)
    matrix_clipped = np.clip(matrix, TACTILE_HEATMAP_VMIN, vmax)
    denom = max(float(vmax - TACTILE_HEATMAP_VMIN), 1e-8)
    norm_img = ((matrix_clipped - TACTILE_HEATMAP_VMIN) / denom * 255.0).astype(np.uint8)
    color_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
    color_img[matrix < black_threshold] = (0, 0, 0)
    return cv2.resize(
        color_img,
        (matrix.shape[1] * pixel_size, matrix.shape[0] * pixel_size),
        interpolation=cv2.INTER_NEAREST,
    )


def tactile_figure_rgb(
    episode_dir: Path,
    row: dict[str, str],
    vmax: float,
    black_threshold: float,
) -> np.ndarray | None:
    tactile, _ = load_tactile_npz(episode_dir, row)
    if tactile is None:
        return None
    panels = []
    for key in TACTILE_TARGET_KEYS:
        panel = tactile_heatmap_bgr(tactile.get(key), vmax, black_threshold)
        if panel is not None:
            panels.append(panel)
    if not panels:
        return None
    return bgr_to_rgb(np.hstack(panels))


def tactile_panel(
    episode_dir: Path,
    row: dict[str, str],
    panel_w: int,
    panel_h: int,
    vmax: float,
    black_threshold: float,
) -> np.ndarray:
    panel = make_panel("tactile heatmap", panel_w, panel_h, f"JET vmin=0 vmax={vmax:g} black<{black_threshold:g}")
    tactile, source_name = load_tactile_npz(episode_dir, row)
    if tactile is None:
        put_lines(panel, ["N/A"], 12, panel_h // 2, color=(80, 80, 80), scale=0.7)
        return panel
    sensor_panels = [tactile_sensor_panel(tactile.get(key), key, vmax, black_threshold) for key in TACTILE_TARGET_KEYS]
    raw = np.hstack(sensor_panels)
    panel[64:, :] = letterbox(raw, panel_w, panel_h - 64)
    cv2.putText(panel, source_name, (12, panel_h - 12), FONT, 0.38, MUTED, 1, cv2.LINE_AA)
    return panel


def state_panel(
    row: dict[str, str],
    selected_index: int,
    source_index: int,
    total_frames: int,
    panel_w: int,
    panel_h: int,
) -> np.ndarray:
    timestamp = row.get("timestamp", "")
    panel = make_panel(
        "state / action",
        panel_w,
        panel_h,
        f"frame {selected_index + 1}/{total_frames}  row={source_index}  t={timestamp}",
    )

    def fmt(name: str, default: str = "N/A") -> str:
        value = row.get(name)
        if value is None or value == "":
            return default
        try:
            number = float(value)
        except ValueError:
            return value
        return f"{number:.5f}"

    lines = [
        "arm position xyz:",
        f"  {fmt('arm_position_x')}  {fmt('arm_position_y')}  {fmt('arm_position_z')}",
        "arm rotation rpy:",
        f"  {fmt('arm_rotation_roll')}  {fmt('arm_rotation_pitch')}  {fmt('arm_rotation_yaw')}",
        "action delta xyz:",
        f"  {fmt('action_delta_x')}  {fmt('action_delta_y')}  {fmt('action_delta_z')}",
        "action delta quat:",
        f"  {fmt('action_delta_qx')}  {fmt('action_delta_qy')}  {fmt('action_delta_qz')}  {fmt('action_delta_qw')}",
        f"gripper distance: {fmt('gripper_distance_mm')} mm",
        f"gripper current:  {fmt('gripper_current_mA')} mA",
        f"pose clamped:     {row.get('is_pose_clamped', 'N/A')}",
    ]
    put_lines(panel, lines, 16, 88, scale=0.48, step=24)
    return panel


def compose_frame(
    episode_dir: Path,
    record: FrameRecord,
    selected_index: int,
    total_frames: int,
    panel_w: int,
    panel_h: int,
    tactile_vmax: float,
    tactile_threshold: float,
) -> np.ndarray:
    row = record.row
    top = [
        image_panel(episode_dir, row, CAMERA_COLUMNS["fisheye"], "fisheye", panel_w, panel_h),
        image_panel(episode_dir, row, CAMERA_COLUMNS["front"], "front", panel_w, panel_h),
        image_panel(episode_dir, row, CAMERA_COLUMNS["side"], "side", panel_w, panel_h),
    ]
    bottom = [
        depth_panel(episode_dir, row, panel_w, panel_h),
        tactile_panel(episode_dir, row, panel_w, panel_h, tactile_vmax, tactile_threshold),
        state_panel(row, selected_index, record.index, total_frames, panel_w, panel_h),
    ]
    return np.vstack([np.hstack(top), np.hstack(bottom)])


def open_writer(output_path: Path, codec: str, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec[:4])
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")
    return writer


def import_matplotlib():
    cache_root = Path("/tmp/pika-raw-visualizer-cache")
    (cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache_root / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )
    return plt


def save_figure(fig, fig_dir: Path, stem: str, formats: Iterable[str], dpi: int) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = fig_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi)
        print(f"Saved figure: {path}")


def even_sample(records: list[FrameRecord], count: int) -> list[KeyframeSample]:
    if count <= 0 or not records:
        return []
    if len(records) <= count:
        return [KeyframeSample(i, record, float("nan")) for i, record in enumerate(records)]
    positions = np.linspace(0, len(records) - 1, count).round().astype(int)
    unique_positions = []
    for pos in positions:
        value = int(pos)
        if value not in unique_positions:
            unique_positions.append(value)
    return [KeyframeSample(pos, records[pos], float("nan")) for pos in unique_positions]


def blank_rgb(width: int = 640, height: int = 480, text: str = "N/A") -> np.ndarray:
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(image, text, (width // 2 - 40, height // 2), FONT, 1.0, (110, 110, 110), 2, cv2.LINE_AA)
    return image


def row_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def tactile_stats_for_row(
    episode_dir: Path,
    row: dict[str, str],
    threshold: float,
) -> dict[str, float]:
    tactile, _ = load_tactile_npz(episode_dir, row)
    stats: dict[str, float] = {}
    for key in TACTILE_TARGET_KEYS:
        matrix = tactile.get(key) if tactile else None
        if matrix is None:
            stats[f"sum_{key}"] = float("nan")
            stats[f"max_{key}"] = float("nan")
            stats[f"area_{key}"] = float("nan")
            continue
        matrix = np.asarray(matrix, dtype=np.float32)
        stats[f"sum_{key}"] = float(np.nansum(matrix))
        stats[f"max_{key}"] = float(np.nanmax(matrix))
        stats[f"area_{key}"] = float(np.mean(matrix >= threshold) * 100.0)
    return stats


def tactile_score(stats: dict[str, float]) -> float:
    values = [stats.get(f"sum_{key}", float("nan")) for key in TACTILE_TARGET_KEYS]
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(sum(finite_values))


def tactile_peak_sample(
    records: list[FrameRecord],
    tactile_stats: list[dict[str, float]],
    count: int,
) -> list[KeyframeSample]:
    if count <= 0 or not records:
        return []
    if len(records) <= count:
        return [
            KeyframeSample(i, record, tactile_score(tactile_stats[i]))
            for i, record in enumerate(records)
        ]

    scores = np.array([tactile_score(stats) for stats in tactile_stats], dtype=np.float32)
    finite_mask = np.isfinite(scores)
    if not np.any(finite_mask):
        return even_sample(records, count)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.max(scores)) <= 0.0:
        return even_sample(records, count)

    if len(scores) >= 5:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        ranked_scores = np.convolve(scores, kernel, mode="same")
    else:
        ranked_scores = scores

    local_peaks: list[int] = []
    for idx in range(len(ranked_scores)):
        left = ranked_scores[idx - 1] if idx > 0 else -np.inf
        right = ranked_scores[idx + 1] if idx + 1 < len(ranked_scores) else -np.inf
        if ranked_scores[idx] >= left and ranked_scores[idx] >= right:
            local_peaks.append(idx)

    min_separation = max(1, len(records) // max(count * 3, 1))
    selected: list[int] = []

    def add_candidates(candidates: Iterable[int], enforce_separation: bool) -> None:
        nonlocal selected
        for idx in candidates:
            if len(selected) >= count:
                break
            if idx in selected:
                continue
            if enforce_separation and any(abs(idx - prev) < min_separation for prev in selected):
                continue
            selected.append(idx)

    peak_candidates = sorted(local_peaks, key=lambda idx: ranked_scores[idx], reverse=True)
    add_candidates(peak_candidates, enforce_separation=True)
    all_candidates = sorted(range(len(scores)), key=lambda idx: ranked_scores[idx], reverse=True)
    add_candidates(all_candidates, enforce_separation=True)
    add_candidates(all_candidates, enforce_separation=False)

    selected = sorted(selected[:count])
    return [
        KeyframeSample(idx, records[idx], float(scores[idx]))
        for idx in selected
    ]


def save_montage_figure(
    plt,
    episode_dir: Path,
    records: list[FrameRecord],
    tactile_stats: list[dict[str, float]],
    fig_dir: Path,
    formats: Iterable[str],
    dpi: int,
    count: int,
    tactile_vmax: float,
    tactile_threshold: float,
) -> None:
    samples = tactile_peak_sample(records, tactile_stats, count)
    if not samples:
        return
    rows_text = ", ".join(
        f"{sample.record.index}(score={sample.tactile_score:.3g})"
        for sample in samples
    )
    print(f"Tactile-guided keyframes: {rows_text}")
    col_titles = ["Front RGB", "Side RGB", "Depth", "Tactile heatmap"]
    fig, axes = plt.subplots(len(samples), len(col_titles), figsize=(7.2, 1.65 * len(samples)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row_i, sample in enumerate(samples):
        selected_index = sample.selected_index
        record = sample.record
        row = record.row
        front = load_bgr_image(episode_dir, row, CAMERA_COLUMNS["front"])
        side = load_bgr_image(episode_dir, row, CAMERA_COLUMNS["side"])
        depth = load_depth_rgb(episode_dir, row)
        tactile = tactile_figure_rgb(episode_dir, row, tactile_vmax, tactile_threshold)
        images = [
            bgr_to_rgb(front) if front is not None else blank_rgb(text="front N/A"),
            bgr_to_rgb(side) if side is not None else blank_rgb(text="side N/A"),
            depth if depth is not None else blank_rgb(text="depth N/A"),
            tactile if tactile is not None else blank_rgb(text="tactile N/A"),
        ]
        timestamp = row_float(row, "timestamp")
        time_label = f"{timestamp - row_float(records[0].row, 'timestamp'):.2f}s" if math.isfinite(timestamp) else "N/A"
        for col_i, image in enumerate(images):
            ax = axes[row_i, col_i]
            ax.imshow(image)
            ax.set_axis_off()
            if row_i == 0:
                ax.set_title(col_titles[col_i], pad=4)
            if col_i == 0:
                ax.text(
                    -0.04,
                    0.5,
                    f"row {record.index}\n{time_label}\ntactile {sample.tactile_score:.3g}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=7,
                )
    fig.suptitle(f"{episode_dir.name}: tactile-peak multimodal keyframes", y=1.01)
    save_figure(fig, fig_dir, "key_frame", formats, dpi)
    plt.close(fig)


def save_timeseries_figure(
    plt,
    episode_dir: Path,
    records: list[FrameRecord],
    tactile_stats: list[dict[str, float]],
    fig_dir: Path,
    formats: Iterable[str],
    dpi: int,
    tactile_threshold: float,
) -> None:
    t0 = row_float(records[0].row, "timestamp")
    times = np.array([row_float(record.row, "timestamp") - t0 for record in records], dtype=np.float32)
    if not np.all(np.isfinite(times)):
        times = np.arange(len(records), dtype=np.float32)
        x_label = "Selected frame"
    else:
        x_label = "Time (s)"

    sum_0 = np.array([stats["sum_0"] for stats in tactile_stats], dtype=np.float32)
    sum_1 = np.array([stats["sum_1"] for stats in tactile_stats], dtype=np.float32)
    max_0 = np.array([stats["max_0"] for stats in tactile_stats], dtype=np.float32)
    max_1 = np.array([stats["max_1"] for stats in tactile_stats], dtype=np.float32)
    area_0 = np.array([stats["area_0"] for stats in tactile_stats], dtype=np.float32)
    area_1 = np.array([stats["area_1"] for stats in tactile_stats], dtype=np.float32)

    gripper = np.array([row_float(record.row, "gripper_distance_mm") for record in records], dtype=np.float32)
    current = np.array([row_float(record.row, "gripper_current_mA") for record in records], dtype=np.float32)
    pos = np.array(
        [
            [
                row_float(record.row, "arm_position_x"),
                row_float(record.row, "arm_position_y"),
                row_float(record.row, "arm_position_z"),
            ]
            for record in records
        ],
        dtype=np.float32,
    )
    action_xyz = np.array(
        [
            [
                row_float(record.row, "action_delta_x"),
                row_float(record.row, "action_delta_y"),
                row_float(record.row, "action_delta_z"),
            ]
            for record in records
        ],
        dtype=np.float32,
    )
    action_norm = np.linalg.norm(action_xyz, axis=1)

    fig, axes = plt.subplots(4, 1, figsize=(7.0, 6.8), sharex=True, constrained_layout=True)
    axes[0].plot(times, sum_0, label="sensor 0", color="#20639b", linewidth=1.3)
    axes[0].plot(times, sum_1, label="sensor 1", color="#c0392b", linewidth=1.3)
    axes[0].set_ylabel("Force sum")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].grid(alpha=0.22)

    axes[1].plot(times, max_0, label="max 0", color="#20639b", linewidth=1.1)
    axes[1].plot(times, max_1, label="max 1", color="#c0392b", linewidth=1.1)
    axes[1].plot(times, area_0 / 100.0, label="contact area 0", color="#5dade2", linewidth=0.9, linestyle="--")
    axes[1].plot(times, area_1 / 100.0, label="contact area 1", color="#ec7063", linewidth=0.9, linestyle="--")
    axes[1].set_ylabel("Peak / area")
    axes[1].legend(frameon=False, ncol=2)
    axes[1].grid(alpha=0.22)

    axes[2].plot(times, gripper, label="distance", color="#2c3e50", linewidth=1.2)
    ax_current = axes[2].twinx()
    ax_current.plot(times, current, label="current", color="#7f8c8d", linewidth=0.9, alpha=0.85)
    axes[2].set_ylabel("Gripper (mm)")
    ax_current.set_ylabel("Current (mA)")
    axes[2].grid(alpha=0.22)

    axes[3].plot(times, pos[:, 0], label="x", color="#1f77b4", linewidth=1.0)
    axes[3].plot(times, pos[:, 1], label="y", color="#ff7f0e", linewidth=1.0)
    axes[3].plot(times, pos[:, 2], label="z", color="#2ca02c", linewidth=1.0)
    axes[3].plot(times, action_norm, label="action norm", color="#111111", linewidth=0.9, linestyle="--")
    axes[3].set_ylabel("Pose/action")
    axes[3].set_xlabel(x_label)
    axes[3].legend(frameon=False, ncol=4)
    axes[3].grid(alpha=0.22)

    fig.suptitle(f"{episode_dir.name}: tactile, gripper, and motion traces")
    save_figure(fig, fig_dir, "temporal_traces", formats, dpi)
    plt.close(fig)


def save_academic_figures(
    episode_dir: Path,
    records: list[FrameRecord],
    fig_dir: Path,
    formats: Iterable[str],
    dpi: int,
    count: int,
    tactile_vmax: float,
    tactile_threshold: float,
) -> None:
    plt = import_matplotlib()
    tactile_stats = [tactile_stats_for_row(episode_dir, record.row, tactile_threshold) for record in records]
    save_montage_figure(
        plt,
        episode_dir,
        records,
        tactile_stats,
        fig_dir,
        formats,
        dpi,
        count,
        tactile_vmax,
        tactile_threshold,
    )
    save_timeseries_figure(plt, episode_dir, records, tactile_stats, fig_dir, formats, dpi, tactile_threshold)


def run() -> None:
    args = parse_args()
    episode_dir = resolve_episode(args.root, args.episode)
    csv_path = episode_dir / args.csv_name
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"Episode directory not found: {episode_dir}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV log not found: {csv_path}")

    records = load_records(csv_path, args.start, args.end, args.stride)
    if not records:
        raise RuntimeError("No frames selected. Check --start/--end/--stride.")

    frame_size = (args.panel_width * 3, args.panel_height * 2)
    writer = open_writer(args.output, args.codec, args.fps, frame_size) if args.output else None
    delay_ms = max(1, int(round(1000.0 / max(args.fps, 1e-6))))

    print(f"Episode: {episode_dir}")
    print(f"Frames: {len(records)}  source rows: {records[0].index}..{records[-1].index}")
    if args.fig_dir:
        print(f"Writing figures: {args.fig_dir}")
        save_academic_figures(
            episode_dir,
            records,
            args.fig_dir,
            args.figure_format,
            args.figure_dpi,
            args.figure_count,
            args.tactile_vmax,
            args.tactile_threshold,
        )
    if args.output:
        print(f"Writing video: {args.output}")
    if args.no_display and writer is None:
        return
    if not args.no_display:
        print("Controls: q/Esc quit, space pause/resume, left/right step while paused")

    paused = False
    position = 0
    try:
        while True:
            record = records[position]
            frame = compose_frame(
                episode_dir,
                record,
                position,
                len(records),
                args.panel_width,
                args.panel_height,
                args.tactile_vmax,
                args.tactile_threshold,
            )
            if writer is not None:
                writer.write(frame)

            if args.no_display:
                position += 1
                if position >= len(records):
                    break
                continue

            cv2.imshow("PIKA Raw Episode Visualizer", frame)
            key = cv2.waitKey(0 if paused else delay_ms)
            if key < 0:
                if paused:
                    time.sleep(0.01)
                continue
            key &= 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            elif paused and key in (81, ord("a")):
                position = max(0, position - 1)
                continue
            elif paused and key in (83, ord("d")):
                position = min(len(records) - 1, position + 1)
                continue

            if not paused:
                position += 1
                if position >= len(records):
                    if args.loop:
                        position = 0
                    else:
                        break
    finally:
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
