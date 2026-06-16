#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import gc
import inspect
import json
import os
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro
from PIL import Image


JOINT_COLUMNS = [
    "arm_j1",
    "arm_j2",
    "arm_j3",
    "arm_j4",
    "arm_j5",
    "arm_j6",
    "arm_j7",
]

POSE_POSITION_COLUMNS = [
    "arm_position_x",
    "arm_position_y",
    "arm_position_z",
]

POSE_QUAT_COLUMNS = [
    "arm_Quaternion_x",
    "arm_Quaternion_y",
    "arm_Quaternion_z",
    "arm_Quaternion_w",
]

GRIPPER_DISTANCE_COLUMN = ["gripper_distance_mm"]

STATE_COLUMNS = JOINT_COLUMNS + POSE_POSITION_COLUMNS + POSE_QUAT_COLUMNS + GRIPPER_DISTANCE_COLUMN

ACTION_COLUMNS = [
    "arm_target_x",
    "arm_target_y",
    "arm_target_z",
    "arm_target_qx",
    "arm_target_qy",
    "arm_target_qz",
    "arm_target_qw",
    "gripper_target_distance_mm",
]

DELTA_ACTION_COLUMNS = [
    "action_delta_x",
    "action_delta_y",
    "action_delta_z",
    "action_delta_qx",
    "action_delta_qy",
    "action_delta_qz",
    "action_delta_qw",
]

TARGET_JOINT_COLUMNS = [
    "arm_target_j1",
    "arm_target_j2",
    "arm_target_j3",
    "arm_target_j4",
    "arm_target_j5",
    "arm_target_j6",
    "arm_target_j7",
]

ACTION_LAYOUT_EEF_GRIPPER = "eef_gripper"
ACTION_LAYOUT_EEF_JOINT_GRIPPER = "eef_joint_gripper"
ACTION_LAYOUT_CHOICES = {
    ACTION_LAYOUT_EEF_GRIPPER,
    ACTION_LAYOUT_EEF_JOINT_GRIPPER,
}


@dataclass
class Args:
    input_root: Path = Path("/data/Datasets/PIKA_real_original/insert_peg_cylinder_RealMachine")
    repo_id: str = "local/insert-peg-cylinder-realmachine"
    output_root: Path = Path("/data/data_realworld/lerobot_export_dataset")
    fps: int = 30
    robot_type: str = "rm75b"
    task_name: str = "insert_peg_cylinder"
    overwrite: bool = False
    push_to_hub: bool = False
    private_hub_repo: bool = False
    max_episodes: int = 0
    max_frames_per_episode: int = 0
    image_width: int = 256
    image_height: int = 256
    image_writer_threads: int = 1
    image_writer_processes: int = 0
    metadata_buffer_size: int = 1
    parallel_video_encoding: bool = False
    action_text: str = "insert the peg into the cylinder hole"
    raw_action_config_path: Path | None = None
    action_config_segment_frames: int = 243
    action_config_min_segment_frames: int = 30
    action_config_overlap_frames: int = 0
    action_zero_pose_window: int = 5
    action_clamp_position_mps: float = 0.5
    action_clamp_rotation_rps: float = 1.0
    action_layout: str = ACTION_LAYOUT_EEF_GRIPPER
    joint_target_shift: int = 1
    dry_run: bool = False


REPORT_FILENAME = "lerobot_conversion_report.json"


def _resolve_lerobot_api() -> tuple[Any, Any]:
    """
    功能：兼容不同 LeRobot 版本的导入路径。
    输入：无。
    输出：`(HF_LEROBOT_HOME, LeRobotDataset)`。
    """
    try:
        from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

        return HF_LEROBOT_HOME, LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

        return HF_LEROBOT_HOME, LeRobotDataset


def _call_with_supported_kwargs(func: Any, /, **kwargs: Any) -> Any:
    """
    功能：兼容 LeRobot 0.3.x 与更新版本的函数签名差异。
    输入：函数对象与候选关键字参数。
    输出：函数调用结果；自动丢弃当前版本不支持的参数。
    """
    signature = inspect.signature(func)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return func(**kwargs)
    supported_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return func(**supported_kwargs)


def _sorted_episode_dirs(root: Path) -> list[Path]:
    """
    功能：按 episode 序号排序目录（episode1, episode2, ...）。
    输入：`root` 原始数据根目录。
    输出：排序后的 episode 路径列表。
    """
    episodes = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("episode")]

    def episode_key(path: Path) -> tuple[int, str]:
        suffix = path.name.replace("episode", "")
        return (int(suffix), path.name) if suffix.isdigit() else (10**9, path.name)

    return sorted(episodes, key=episode_key)


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """
    功能：读取 `teleop_log.csv` 为行字典列表。
    输入：`csv_path`。
    输出：`list[dict[str, str]]`。
    """
    with csv_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _image_resize_size(args: Args) -> tuple[int, int] | None:
    if args.image_width <= 0 or args.image_height <= 0:
        return None
    return (int(args.image_width), int(args.image_height))


def _load_rgb(image_path: Path, resize_size: tuple[int, int] | None = None) -> np.ndarray:
    """
    功能：读取一张 RGB 图像并转为 `uint8` 数组。
    输入：`image_path`。
    输出：形状 `(H, W, 3)` 的 `np.ndarray`。
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if resize_size is not None and image.size != resize_size:
            image = image.resize(resize_size, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def _load_tactile(npz_path: Path) -> np.ndarray:
    """
    功能：读取 PA-STE 触觉 npz，并拼成双指触觉张量。
    输入：`npz_path`（期望包含两个二维阵列键，如 `0` 与 `1`）。
    输出：形状 `(2, H, W)` 的 `float32` 数组。
    """
    data = np.load(npz_path)
    keys = sorted(data.files)
    if len(keys) < 2:
        raise ValueError(f"触觉文件至少应包含两个阵列，当前为 {keys}，文件：{npz_path}")

    left = data[keys[0]].astype(np.float32)
    right = data[keys[1]].astype(np.float32)
    if left.shape != right.shape:
        raise ValueError(f"触觉左右阵列尺寸不一致：{left.shape} vs {right.shape}，文件：{npz_path}")

    return np.stack([left, right], axis=0)


def _float_row(row: dict[str, str], columns: list[str], csv_path: Path) -> np.ndarray:
    """
    功能：从 CSV 行中提取指定列并转换为 `float32` 向量。
    输入：`row`、`columns`、`csv_path`（仅用于报错提示）。
    输出：`np.ndarray(float32)`。
    """
    values: list[float] = []
    for col in columns:
        if col not in row:
            raise KeyError(f"CSV 缺少列 `{col}`，文件：{csv_path}")
        values.append(float(row[col]))
    return np.asarray(values, dtype=np.float32)


def _normalize_quaternion(quaternion_xyzw: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion_xyzw))
    if norm < 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return (quaternion_xyzw / norm).astype(np.float64)


def _align_quaternion_sign(reference_xyzw: np.ndarray, candidate_xyzw: np.ndarray) -> np.ndarray:
    if float(np.dot(reference_xyzw, candidate_xyzw)) < 0.0:
        return -candidate_xyzw
    return candidate_xyzw


def _quat_multiply_xyzw(lhs_xyzw: np.ndarray, rhs_xyzw: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = lhs_xyzw
    rx, ry, rz, rw = rhs_xyzw
    return np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def _quat_angle_distance(lhs_xyzw: np.ndarray, rhs_xyzw: np.ndarray) -> float:
    lhs = _normalize_quaternion(lhs_xyzw)
    rhs = _normalize_quaternion(rhs_xyzw)
    dot = float(np.clip(np.abs(np.dot(lhs, rhs)), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def _quat_slerp(lhs_xyzw: np.ndarray, rhs_xyzw: np.ndarray, t: float) -> np.ndarray:
    lhs = _normalize_quaternion(lhs_xyzw)
    rhs = _normalize_quaternion(rhs_xyzw)
    rhs = _align_quaternion_sign(lhs, rhs)

    dot = float(np.clip(np.dot(lhs, rhs), -1.0, 1.0))
    if dot > 0.9995:
        blended = lhs + t * (rhs - lhs)
        return _normalize_quaternion(blended)

    theta = float(np.arccos(dot))
    sin_theta = float(np.sin(theta))
    if sin_theta < 1e-8:
        return _normalize_quaternion(lhs)

    ratio_lhs = float(np.sin((1.0 - t) * theta) / sin_theta)
    ratio_rhs = float(np.sin(t * theta) / sin_theta)
    return _normalize_quaternion(ratio_lhs * lhs + ratio_rhs * rhs)


def _average_quaternions_xyzw(quaternions_xyzw: list[np.ndarray]) -> np.ndarray:
    if not quaternions_xyzw:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    reference = _normalize_quaternion(quaternions_xyzw[0])
    aligned = [_align_quaternion_sign(reference, _normalize_quaternion(quat)) for quat in quaternions_xyzw]
    mean_quat = np.mean(np.stack(aligned, axis=0), axis=0)
    return _normalize_quaternion(mean_quat)


def _resolve_action_source(row: dict[str, str], csv_path: Path) -> str:
    columns = set(row.keys())
    has_target = all(column in columns for column in ACTION_COLUMNS)
    has_delta = all(column in columns for column in DELTA_ACTION_COLUMNS + ["gripper_target_distance_mm"])
    if has_target:
        return "target"
    if has_delta:
        return "delta"

    missing_target = [column for column in ACTION_COLUMNS if column not in columns]
    missing_delta = [column for column in DELTA_ACTION_COLUMNS if column not in columns]
    raise KeyError(
        f"CSV 动作列不完整，既无法直接读取 arm_target，也无法由 action_delta 重构。"
        f"缺失 arm_target 列: {missing_target}; 缺失 action_delta 列: {missing_delta}; 文件: {csv_path}"
    )


def _estimate_zero_pose_from_rows(rows: list[dict[str, str]], csv_path: Path, window: int) -> tuple[np.ndarray, np.ndarray]:
    sample_count = max(1, min(window, len(rows)))
    positions: list[np.ndarray] = []
    quaternions: list[np.ndarray] = []

    for row in rows[:sample_count]:
        position = _float_row(row, POSE_POSITION_COLUMNS, csv_path).astype(np.float64)
        quaternion = _float_row(row, POSE_QUAT_COLUMNS, csv_path).astype(np.float64)
        positions.append(position)
        quaternions.append(quaternion)

    zero_position = np.mean(np.stack(positions, axis=0), axis=0)
    zero_quaternion = _average_quaternions_xyzw(quaternions)
    return zero_position, zero_quaternion


def _extract_action_target(
    row: dict[str, str],
    csv_path: Path,
    action_source: str,
    zero_position: np.ndarray | None,
    zero_quaternion: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    if action_source == "target":
        target_position = _float_row(row, ACTION_COLUMNS[:3], csv_path).astype(np.float64)
        target_quaternion = _normalize_quaternion(_float_row(row, ACTION_COLUMNS[3:7], csv_path).astype(np.float64))
        target_gripper = float(row["gripper_target_distance_mm"])
        return target_position, target_quaternion, target_gripper, False

    if zero_position is None or zero_quaternion is None:
        raise RuntimeError(f"动作重构失败：缺少零点位姿，文件: {csv_path}")

    delta_position = _float_row(row, DELTA_ACTION_COLUMNS[:3], csv_path).astype(np.float64)
    delta_quaternion = _normalize_quaternion(_float_row(row, DELTA_ACTION_COLUMNS[3:7], csv_path).astype(np.float64))
    target_position = zero_position + delta_position
    target_quaternion = _normalize_quaternion(_quat_multiply_xyzw(delta_quaternion, zero_quaternion))
    target_gripper = float(row["gripper_target_distance_mm"])
    return target_position, target_quaternion, target_gripper, True


def _validate_action_layout(action_layout: str) -> str:
    if action_layout not in ACTION_LAYOUT_CHOICES:
        raise ValueError(
            f"Unsupported action_layout={action_layout!r}; "
            f"choose from {sorted(ACTION_LAYOUT_CHOICES)}"
        )
    return action_layout


def _action_names(action_layout: str) -> list[str]:
    _validate_action_layout(action_layout)
    eef_names = ["eef_x", "eef_y", "eef_z", "eef_qx", "eef_qy", "eef_qz", "eef_qw"]
    if action_layout == ACTION_LAYOUT_EEF_GRIPPER:
        return eef_names + ["gripper"]
    joint_names = [f"joint_{idx}" for idx in range(1, 8)]
    return eef_names + joint_names + ["gripper"]


def _has_valid_columns(row: dict[str, str], columns: list[str]) -> bool:
    return all(column in row and not _is_missing_value(row.get(column)) for column in columns)


def _extract_joint_action(
    row: dict[str, str],
    csv_path: Path,
    rows: list[dict[str, str]] | None = None,
    frame_idx: int | None = None,
    joint_target_shift: int = 1,
) -> tuple[np.ndarray, str]:
    if _has_valid_columns(row, TARGET_JOINT_COLUMNS):
        return _float_row(row, TARGET_JOINT_COLUMNS, csv_path), "target_joint"

    if rows is not None and frame_idx is not None and joint_target_shift > 0:
        target_idx = min(len(rows) - 1, frame_idx + int(joint_target_shift))
        if target_idx > frame_idx and _has_valid_columns(rows[target_idx], JOINT_COLUMNS):
            return (
                _float_row(rows[target_idx], JOINT_COLUMNS, csv_path),
                f"future_observed_joint_t+{target_idx - frame_idx}",
            )

    return _float_row(row, JOINT_COLUMNS, csv_path), "current_observed_joint_fallback"


def _build_action_vector(
    action_layout: str,
    clamped_position: np.ndarray,
    clamped_quaternion: np.ndarray,
    joint_action: np.ndarray,
    target_gripper: float,
) -> np.ndarray:
    _validate_action_layout(action_layout)
    eef_action = np.asarray(
        [
            float(clamped_position[0]),
            float(clamped_position[1]),
            float(clamped_position[2]),
            float(clamped_quaternion[0]),
            float(clamped_quaternion[1]),
            float(clamped_quaternion[2]),
            float(clamped_quaternion[3]),
        ],
        dtype=np.float32,
    )
    gripper_action = np.asarray([float(target_gripper)], dtype=np.float32)
    if action_layout == ACTION_LAYOUT_EEF_GRIPPER:
        return np.concatenate([eef_action, gripper_action], axis=0).astype(np.float32)
    return np.concatenate(
        [eef_action, joint_action.astype(np.float32, copy=False), gripper_action],
        axis=0,
    ).astype(np.float32)


def _clamp_target_jump(
    previous_position: np.ndarray | None,
    previous_quaternion: np.ndarray | None,
    current_position: np.ndarray,
    current_quaternion: np.ndarray,
    max_position_step: float,
    max_rotation_step: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if previous_position is None or previous_quaternion is None:
        return current_position, current_quaternion, False

    clamped = False
    output_position = current_position.copy()
    output_quaternion = current_quaternion.copy()

    if max_position_step > 0.0:
        delta = output_position - previous_position
        distance = float(np.linalg.norm(delta))
        if distance > max_position_step and distance > 1e-12:
            output_position = previous_position + delta * (max_position_step / distance)
            clamped = True

    if max_rotation_step > 0.0:
        angle = _quat_angle_distance(previous_quaternion, output_quaternion)
        if angle > max_rotation_step and angle > 1e-12:
            ratio = max_position_step  # placeholder to keep branch structure simple
            ratio = max_rotation_step / angle
            output_quaternion = _quat_slerp(previous_quaternion, output_quaternion, float(np.clip(ratio, 0.0, 1.0)))
            clamped = True

    output_quaternion = _align_quaternion_sign(previous_quaternion, _normalize_quaternion(output_quaternion))
    return output_position, output_quaternion, clamped


def _build_features(
    front_shape: tuple[int, int, int],
    side_rgb_shape: tuple[int, int, int],
    fisheye_shape: tuple[int, int, int],
    tactile_shape: tuple[int, int, int],
    action_names: list[str],
) -> dict[str, dict[str, Any]]:
    
    return {
        "observation.images.cam_front": {
            "dtype": "image",
            "shape": front_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.images.cam_side": {
            "dtype": "image",
            "shape": side_rgb_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.images.cam_fisheye": {
            "dtype": "image",
            "shape": fisheye_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.tactile": {
            "dtype": "float32",
            "shape": tactile_shape,
            "names": ["finger", "row", "col"],
        },
        "observation.joint_position": {
            "dtype": "float32",
            "shape": (len(JOINT_COLUMNS),),
            "names": ["joint"],
        },
        "observation.ee_pose": {
            "dtype": "float32",
            "shape": (len(POSE_POSITION_COLUMNS) + len(POSE_QUAT_COLUMNS),),
            "names": ["pose"],
        },
        "observation.gripper_distance": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["distance_mm"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_COLUMNS),),
            "names": ["state"],
        },
        "observation.timestamp": {
            "dtype": "float64",
            "shape": (1,),
            "names": ["time"],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": ["action"],
        },
    }


def _safe_add_frame(dataset: Any, frame: dict[str, Any], task: str) -> None:
    """
    功能：兼容不同 LeRobot 版本的 `add_frame` 调用签名。
    输入：`dataset`、`frame`、`task`。
    输出：无（写入一帧到内存缓冲）。
    """
    try:
        dataset.add_frame(frame, task=task)
    except TypeError:
        frame_with_task = dict(frame)
        frame_with_task["task"] = task
        dataset.add_frame(frame_with_task)


def _load_annotations(path: Path) -> dict[str, Any]:
    
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_raw_action_config_path(input_root: Path, explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"raw action_config annotation file not found: {path}")
        return path

    for name in ("raw_action_config.json", "action_config_annotations.json"):
        path = input_root / name
        if path.exists():
            return path
    return None


def _load_raw_action_configs(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    episodes = data.get("episodes", data)
    if not isinstance(episodes, dict):
        raise ValueError(f"raw action_config file must contain an episodes object: {path}")

    out: dict[str, list[dict[str, Any]]] = {}
    for episode_name, value in episodes.items():
        segments = value.get("segments", value) if isinstance(value, dict) else value
        if not isinstance(segments, list):
            raise ValueError(f"raw action_config for {episode_name!r} must be a list")
        out[str(episode_name)] = [
            {
                "start_frame": int(segment["start_frame"]),
                "end_frame": int(segment["end_frame"]),
                "action_text": str(segment["action_text"]),
            }
            for segment in segments
        ]
    return out


def _clip_raw_action_config(
    raw_segments: list[dict[str, Any]],
    length: int,
    fallback_text: str,
) -> list[dict[str, Any]]:
    clipped: list[dict[str, Any]] = []
    for segment in sorted(raw_segments, key=lambda item: (int(item["start_frame"]), int(item["end_frame"]))):
        start = max(0, min(length, int(segment["start_frame"])))
        end = max(0, min(length, int(segment["end_frame"])))
        text = str(segment.get("action_text") or fallback_text)
        if end <= start:
            continue
        clipped.append({"start_frame": start, "end_frame": end, "action_text": text})

    if not clipped:
        return [{"start_frame": 0, "end_frame": length, "action_text": fallback_text}]

    # Enforce contiguous action_config because latent extraction assumes complete segments.
    repaired: list[dict[str, Any]] = []
    cursor = 0
    for segment in clipped:
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        if start > cursor:
            repaired.append(
                {
                    "start_frame": cursor,
                    "end_frame": start,
                    "action_text": fallback_text,
                }
            )
        elif start < cursor:
            start = cursor
        if end > start:
            repaired.append(
                {
                    "start_frame": start,
                    "end_frame": end,
                    "action_text": str(segment["action_text"]),
                }
            )
            cursor = end
    if cursor < length:
        repaired.append({"start_frame": cursor, "end_frame": length, "action_text": fallback_text})
    return repaired


def _phase_for_index(annotations: dict[str, Any], episode_name: str, idx: int) -> str | None:
    """
    功能：根据帧索引查询该帧所属阶段标签（如 approaching/grasping）。
    输入：`annotations`、`episode_name`、`idx`。
    输出：阶段标签字符串或 `None`。
    """
    phases = annotations.get(episode_name)
    if not isinstance(phases, list):
        return None

    for phase in phases:
        start_idx = phase.get("start_idx")
        end_idx = phase.get("end_idx")
        if isinstance(start_idx, int) and isinstance(end_idx, int) and start_idx <= idx <= end_idx:
            label = phase.get("label")
            return str(label) if label else None
    return None


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
    action_text = default_text or (tasks[0] if tasks else "robot manipulation")
    length = int(length)
    segment_frames = int(segment_frames)
    min_segment_frames = max(1, int(min_segment_frames))
    overlap_frames = max(0, int(overlap_frames))

    if length <= 0:
        return []
    if segment_frames <= 0 or segment_frames >= length:
        return [{"start_frame": 0, "end_frame": length, "action_text": action_text}]

    overlap_frames = min(overlap_frames, segment_frames - 1)
    step = max(1, segment_frames - overlap_frames)
    segments: list[dict[str, Any]] = []
    start_frame = 0

    while start_frame < length:
        end_frame = min(length, start_frame + segment_frames)
        if segments and end_frame - start_frame < min_segment_frames:
            segments[-1]["end_frame"] = length
            break
        segments.append(
            {
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "action_text": action_text,
            }
        )
        if end_frame >= length:
            break
        start_frame += step

    return segments


def _create_episodes_jsonl_from_parquet(
    dataset_root: Path,
    default_text: str,
    segment_frames: int,
    min_segment_frames: int,
    overlap_frames: int,
) -> int:
    episodes_dir = dataset_root / "meta" / "episodes"
    parquet_paths = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    if not parquet_paths:
        return 0

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot wrote meta/episodes parquet files, but pandas is unavailable to create "
            "meta/episodes.jsonl for LingBot-VA latent extraction."
        ) from exc

    items: list[dict[str, Any]] = []
    for parquet_path in parquet_paths:
        dataframe = pd.read_parquet(parquet_path)
        for _, row in dataframe.iterrows():
            episode_index = int(row["episode_index"])
            length = int(row["length"])
            tasks = _tasks_to_list(row.get("tasks"))
            item = {
                "episode_index": episode_index,
                "tasks": tasks,
                "length": length,
                "action_config": _default_action_config(
                    length,
                    tasks,
                    default_text,
                    segment_frames,
                    min_segment_frames,
                    overlap_frames,
                ),
            }
            items.append(item)

    items.sort(key=lambda item: int(item["episode_index"]))
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    episodes_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8",
    )
    return len(items)


def _add_default_action_config(
    dataset_root: Path,
    default_text: str,
    segment_frames: int,
    min_segment_frames: int,
    overlap_frames: int,
) -> int:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return _create_episodes_jsonl_from_parquet(
            dataset_root,
            default_text,
            segment_frames,
            min_segment_frames,
            overlap_frames,
        )

    changed = 0
    updated_lines = []
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if "action_config" not in item:
            tasks = item.get("tasks") or []
            item["action_config"] = _default_action_config(
                int(item["length"]),
                tasks,
                default_text,
                segment_frames,
                min_segment_frames,
                overlap_frames,
            )
            changed += 1
        updated_lines.append(json.dumps(item, ensure_ascii=False))

    if changed:
        backup_path = episodes_path.with_suffix(".jsonl.bak")
        if not backup_path.exists():
            backup_path.write_text(episodes_path.read_text(encoding="utf-8"), encoding="utf-8")
        episodes_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    return changed


def _apply_raw_action_configs_to_episodes_jsonl(
    dataset_root: Path,
    episode_stats: list[dict[str, Any]],
    raw_action_configs: dict[str, list[dict[str, Any]]],
    fallback_text: str,
) -> int:
    if not raw_action_configs:
        return 0

    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found after conversion: {episodes_path}")

    episode_index_to_name = {
        episode_index: item["episode"]
        for episode_index, item in enumerate(episode_stats)
        if "episode" in item
    }
    updated_lines = []
    changed = 0
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        episode_index = int(item["episode_index"])
        episode_name = episode_index_to_name.get(episode_index)
        if episode_name in raw_action_configs:
            item["action_config"] = _clip_raw_action_config(
                raw_action_configs[episode_name],
                int(item["length"]),
                fallback_text,
            )
            changed += 1
        updated_lines.append(json.dumps(item, ensure_ascii=False))

    if changed:
        backup_path = episodes_path.with_suffix(".jsonl.before_raw_action_config.bak")
        if not backup_path.exists():
            backup_path.write_text(episodes_path.read_text(encoding="utf-8"), encoding="utf-8")
        episodes_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    return changed


def _check_required_columns(rows: list[dict[str, str]], csv_path: Path) -> str:
    """
    功能：检查状态/动作必需列是否存在。
    输入：CSV 行列表、CSV 路径。
    输出：动作来源（`target` 或 `delta`）；缺失时抛异常。
    """
    if not rows:
        raise RuntimeError(f"CSV 为空：{csv_path}")

    first = rows[0]
    required = [
        "timestamp",
        "filepath_color",
        "filepath_side_rgb",
        "filepath_fisheye",
        "filepath_tactile_gripper",
    ] + STATE_COLUMNS

    action_source = _resolve_action_source(first, csv_path)
    if action_source == "target":
        required += ACTION_COLUMNS
    else:
        required += DELTA_ACTION_COLUMNS + ["gripper_target_distance_mm"]

    missing = [col for col in required if col not in first]
    if missing:
        raise KeyError(f"CSV 缺少关键列：{missing}，文件：{csv_path}")
    return action_source


def _is_missing_value(value: str | None) -> bool:
    
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in {"", "n/a", "na", "none", "null"}


def convert(args: Args) -> None:
    action_layout = _validate_action_layout(args.action_layout)
    action_names = _action_names(action_layout)

    input_root = args.input_root.expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_root}")

    episode_dirs = _sorted_episode_dirs(input_root)
    if args.max_episodes > 0:
        episode_dirs = episode_dirs[: args.max_episodes]

    if not episode_dirs:
        raise RuntimeError(f"在目录 {input_root} 下未找到 episode 子目录")

    sample_csv = episode_dirs[0] / "teleop_log.csv"
    if not sample_csv.exists():
        raise RuntimeError(f"样本 episode 缺少 teleop_log.csv：{episode_dirs[0]}")

    sample_rows = _read_csv_rows(sample_csv)
    sample_action_source = _check_required_columns(sample_rows, sample_csv)
    resize_size = _image_resize_size(args)

    sample_row = sample_rows[0]
    sample_front = _load_rgb(episode_dirs[0] / sample_row["filepath_color"], resize_size)
    sample_side_rgb = _load_rgb(episode_dirs[0] / sample_row["filepath_side_rgb"], resize_size)
    sample_fisheye = _load_rgb(episode_dirs[0] / sample_row["filepath_fisheye"], resize_size)
    sample_tactile = _load_tactile(episode_dirs[0] / sample_row["filepath_tactile_gripper"])

    features = _build_features(
        front_shape=tuple(sample_front.shape),
        side_rgb_shape=tuple(sample_side_rgb.shape),
        fisheye_shape=tuple(sample_fisheye.shape),
        tactile_shape=tuple(sample_tactile.shape),
        action_names=action_names,
    )

    annotations = _load_annotations(input_root / "annotations.json")
    raw_action_config_path = _resolve_raw_action_config_path(input_root, args.raw_action_config_path)
    raw_action_configs = _load_raw_action_configs(raw_action_config_path)

    if args.dry_run:
        print("[DRY-RUN] schema 检查通过：")
        print(f"  episodes               : {len(episode_dirs)}")
        print(f"  cam_front shape        : {sample_front.shape}")
        print(f"  cam_side shape         : {sample_side_rgb.shape}")
        print(f"  cam_fisheye shape      : {sample_fisheye.shape}")
        print(f"  image resize           : {resize_size or 'disabled'}")
        print(f"  tactile shape          : {sample_tactile.shape}")
        print(f"  observation.state dim  : {len(STATE_COLUMNS)}")
        print(f"  action layout          : {action_layout}")
        print(f"  action dim             : {len(action_names)}")
        print(f"  action source          : {sample_action_source}")
        print(f"  action order           : {action_names}")
        if action_layout == ACTION_LAYOUT_EEF_JOINT_GRIPPER:
            _, joint_action_source = _extract_joint_action(
                sample_row,
                sample_csv,
                rows=sample_rows,
                frame_idx=0,
                joint_target_shift=args.joint_target_shift,
            )
            print(f"  joint action source    : {joint_action_source}")
            print(f"  joint target shift     : {args.joint_target_shift}")
        print(f"  action_config text     : {args.action_text}")
        print(f"  raw action_config      : {raw_action_config_path or 'not provided'}")
        print(f"  action_config segment  : {args.action_config_segment_frames} frames")
        print("  action rebuild/clamp   :")
        print(f"    zero pose window     : {args.action_zero_pose_window}")
        print(f"    clamp pos (m/s)      : {args.action_clamp_position_mps}")
        print(f"    clamp rot (rad/s)    : {args.action_clamp_rotation_rps}")
        print(f"  features               : {list(features.keys())}")
        return

    HF_LEROBOT_HOME, LeRobotDataset = _resolve_lerobot_api()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = (output_root / args.repo_id).resolve()

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"输出已存在：{output_dir}，请加 `--overwrite` 覆盖")

    dataset = _call_with_supported_kwargs(
        LeRobotDataset.create,
        repo_id=args.repo_id,
        root=str(output_dir),
        robot_type=args.robot_type,
        fps=args.fps,
        features=features,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        metadata_buffer_size=args.metadata_buffer_size,
    )

    written_episodes = 0
    total_frames = 0
    skipped_frames = 0
    field_missing_stats: dict[str, int] = {column: 0 for column in STATE_COLUMNS + ACTION_COLUMNS}
    field_missing_stats.update(
        {
            "timestamp": 0,
            "filepath_color": 0,
            "filepath_side_rgb": 0,
            "filepath_fisheye": 0,
            "filepath_tactile_gripper": 0,
        }
    )
    error_type_stats: dict[str, int] = {}
    episode_stats: list[dict[str, Any]] = []
    reconstructed_action_frames = 0
    clamped_action_frames = 0
    joint_action_source_stats: dict[str, int] = {}

    for episode_dir in episode_dirs:
        csv_path = episode_dir / "teleop_log.csv"
        if not csv_path.exists():
            print(f"[SKIP] {episode_dir.name}: 缺少 teleop_log.csv")
            continue

        rows = _read_csv_rows(csv_path)
        if not rows:
            print(f"[SKIP] {episode_dir.name}: teleop_log.csv 为空")
            continue

        action_source = _check_required_columns(rows, csv_path)

        zero_position: np.ndarray | None = None
        zero_quaternion: np.ndarray | None = None
        if action_source == "delta":
            zero_position, zero_quaternion = _estimate_zero_pose_from_rows(rows, csv_path, args.action_zero_pose_window)

        if args.max_frames_per_episode > 0:
            rows = rows[: args.max_frames_per_episode]

        episode_added = 0
        episode_skipped = 0
        episode_total = len(rows)
        episode_reconstructed = 0
        episode_clamped = 0
        episode_joint_action_source_stats: dict[str, int] = {}

        previous_timestamp: float | None = None
        previous_clamped_position: np.ndarray | None = None
        previous_clamped_quaternion: np.ndarray | None = None
        for frame_idx, row in enumerate(rows):
            try:
                required_row_fields = [
                    "timestamp",
                    "filepath_color",
                    "filepath_side_rgb",
                    "filepath_fisheye",
                    "filepath_tactile_gripper",
                ] + STATE_COLUMNS
                if action_source == "target":
                    required_row_fields += ACTION_COLUMNS
                else:
                    required_row_fields += DELTA_ACTION_COLUMNS + ["gripper_target_distance_mm"]

                missing_in_row = [field for field in required_row_fields if _is_missing_value(row.get(field))]
                if missing_in_row:
                    for field in missing_in_row:
                        if field in field_missing_stats:
                            field_missing_stats[field] += 1
                    skipped_frames += 1
                    episode_skipped += 1
                    continue

                front = _load_rgb(episode_dir / row["filepath_color"], resize_size)
                side_rgb = _load_rgb(episode_dir / row["filepath_side_rgb"], resize_size)
                fisheye = _load_rgb(episode_dir / row["filepath_fisheye"], resize_size)
                tactile = _load_tactile(episode_dir / row["filepath_tactile_gripper"])

                joint_position = _float_row(row, JOINT_COLUMNS, csv_path)
                ee_pose = _float_row(row, POSE_POSITION_COLUMNS + POSE_QUAT_COLUMNS, csv_path)
                gripper_distance = _float_row(row, GRIPPER_DISTANCE_COLUMN, csv_path)
                state = _float_row(row, STATE_COLUMNS, csv_path)
                timestamp_value = float(row["timestamp"])
                timestamp = np.asarray([timestamp_value], dtype=np.float64)

                target_position, target_quaternion, target_gripper, reconstructed = _extract_action_target(
                    row=row,
                    csv_path=csv_path,
                    action_source=action_source,
                    zero_position=zero_position,
                    zero_quaternion=zero_quaternion,
                )

                dt = (1.0 / float(args.fps)) if previous_timestamp is None else max(1e-6, timestamp_value - previous_timestamp)
                max_position_step = max(0.0, args.action_clamp_position_mps) * dt
                max_rotation_step = max(0.0, args.action_clamp_rotation_rps) * dt

                clamped_position, clamped_quaternion, clamped = _clamp_target_jump(
                    previous_position=previous_clamped_position,
                    previous_quaternion=previous_clamped_quaternion,
                    current_position=target_position,
                    current_quaternion=target_quaternion,
                    max_position_step=max_position_step,
                    max_rotation_step=max_rotation_step,
                )

                joint_action, joint_action_source = _extract_joint_action(
                    row,
                    csv_path,
                    rows=rows,
                    frame_idx=frame_idx,
                    joint_target_shift=args.joint_target_shift,
                )
                joint_action_source_stats[joint_action_source] = joint_action_source_stats.get(joint_action_source, 0) + 1
                episode_joint_action_source_stats[joint_action_source] = (
                    episode_joint_action_source_stats.get(joint_action_source, 0) + 1
                )
                action = _build_action_vector(
                    action_layout=action_layout,
                    clamped_position=clamped_position,
                    clamped_quaternion=clamped_quaternion,
                    joint_action=joint_action,
                    target_gripper=target_gripper,
                )

                previous_timestamp = timestamp_value
                previous_clamped_position = clamped_position
                previous_clamped_quaternion = clamped_quaternion

                if reconstructed:
                    reconstructed_action_frames += 1
                    episode_reconstructed += 1
                if clamped:
                    clamped_action_frames += 1
                    episode_clamped += 1

                frame: dict[str, Any] = {
                    "observation.images.cam_front": front,
                    "observation.images.cam_side": side_rgb,
                    "observation.images.cam_fisheye": fisheye,
                    "observation.tactile": tactile,
                    "observation.joint_position": joint_position,
                    "observation.ee_pose": ee_pose,
                    "observation.gripper_distance": gripper_distance,
                    "observation.state": state,
                    "observation.timestamp": timestamp,
                    "action": action,
                }

                phase = _phase_for_index(annotations, episode_dir.name, frame_idx)
                task = args.task_name if phase is None else f"{args.task_name}:{phase}"
                _safe_add_frame(dataset, frame, task=task)

                episode_added += 1
                total_frames += 1
            except Exception as exc:
                skipped_frames += 1
                episode_skipped += 1
                error_name = type(exc).__name__
                error_type_stats[error_name] = error_type_stats.get(error_name, 0) + 1
                print(f"[WARN] {episode_dir.name} frame={frame_idx}: {exc}")

        if episode_added == 0:
            print(f"[SKIP] {episode_dir.name}: 无有效帧")
            continue

        try:
            _call_with_supported_kwargs(dataset.save_episode, parallel_encoding=args.parallel_video_encoding)
        except Exception:
            print(f"[ERROR] {episode_dir.name}: dataset.save_episode() failed")
            traceback.print_exc()
            raise
        written_episodes += 1
        print(f"[OK] {episode_dir.name}: {episode_added} 帧")
        episode_stats.append(
            {
                "episode": episode_dir.name,
                "action_source": action_source,
                "action_layout": action_layout,
                "action_dim": len(action_names),
                "total_rows": episode_total,
                "written_frames": episode_added,
                "skipped_frames": episode_skipped,
                "reconstructed_action_frames": episode_reconstructed,
                "clamped_action_frames": episode_clamped,
                "joint_target_shift": args.joint_target_shift,
                "joint_action_source_counts": episode_joint_action_source_stats,
            }
        )
        gc.collect()

    metadata = getattr(dataset, "meta", None)
    metadata_close = getattr(metadata, "_close_writer", None)
    metadata_flush = getattr(metadata, "_flush_metadata_buffer", None)
    if callable(metadata_close):
        metadata_close()
    elif callable(metadata_flush):
        metadata_flush()

    report = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "dataset_output_dir": str(output_dir),
        "repo_id": args.repo_id,
        "robot_type": args.robot_type,
        "fps": args.fps,
        "converted_episodes": written_episodes,
        "written_frames": total_frames,
        "skipped_frames": skipped_frames,
        "reconstructed_action_frames": reconstructed_action_frames,
        "clamped_action_frames": clamped_action_frames,
        "action_layout": action_layout,
        "action_dim": len(action_names),
        "action_order": action_names,
        "joint_target_shift": args.joint_target_shift,
        "joint_action_source_counts": joint_action_source_stats,
        "action_config": {
            "action_text": args.action_text,
            "raw_action_config_path": str(raw_action_config_path) if raw_action_config_path else None,
            "segment_frames": args.action_config_segment_frames,
            "min_segment_frames": args.action_config_min_segment_frames,
            "overlap_frames": args.action_config_overlap_frames,
        },
        "missing_field_counts": field_missing_stats,
        "error_type_counts": error_type_stats,
        "episodes": episode_stats,
    }
    report_path = output_root / REPORT_FILENAME
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)

    print("=" * 68)
    print("转换完成")
    print(f"写入 episode 数 : {written_episodes}")
    print(f"写入帧数        : {total_frames}")
    print(f"跳过帧数        : {skipped_frames}")
    print(f"action layout   : {action_layout} ({len(action_names)}D)")
    print(f"输出目录        : {output_dir}")
    print(f"质检报告        : {report_path}")
    print("=" * 68)

    if written_episodes == 0:
        raise RuntimeError("未成功写入任何 episode，请检查数据完整性。")

    action_config_count = _add_default_action_config(
        output_dir,
        args.action_text,
        args.action_config_segment_frames,
        args.action_config_min_segment_frames,
        args.action_config_overlap_frames,
    )
    print(f"action_config 写入 episode 数: {action_config_count}")
    raw_action_config_count = _apply_raw_action_configs_to_episodes_jsonl(
        output_dir,
        episode_stats,
        raw_action_configs,
        args.action_text,
    )
    if raw_action_configs:
        print(f"raw action_config 覆盖 episode 数: {raw_action_config_count}")

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["real-world", "tactile", "pa-ste", args.robot_type],
            private=args.private_hub_repo,
            push_videos=True,
            license="apache-2.0",
        )


def main() -> None:
    """
    功能：命令行入口。
    输入：命令行参数。
    输出：无。
    """
    convert(tyro.cli(Args))


if __name__ == "__main__":
    main()
