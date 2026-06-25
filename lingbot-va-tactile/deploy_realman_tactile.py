#!/usr/bin/env python3
"""Deploy LingBot-VA-Tactile on the local RealMan + Pika hardware.

This is a websocket client. Start ``tactile_va.server_tactile`` separately, then
run this script on the machine connected to the robot/cameras/gripper.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WAN_VA_ROOT = REPO_ROOT / "wan_va"
for path in (str(WAN_VA_ROOT), str(REPO_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

TACTILE_SHAPE = (2, 32, 58)
SUPPORTED_ACTION_CHANNELS = (8, 15, 30)
DEFAULT_PIKA_PROJECT_ROOT = "/home/tujian/Projects/pika_sdk/PIKA_RM65B_data_acquisition"

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_COLOR = (235, 235, 235)
MUTED = (145, 155, 165)
BG = (8, 10, 12)
PANEL_BG = (18, 22, 27)


def parse_csv_floats(text: str, expected: int | None = None) -> np.ndarray:
    values = np.asarray([float(part) for part in text.split(",") if part.strip()], dtype=np.float32)
    if expected is not None and values.size != expected:
        raise ValueError(f"Expected {expected} comma-separated values, got {values.size}: {text!r}")
    return values


def parse_csv_ints(text: str, expected: int | None = None) -> list[int]:
    values = [int(part) for part in text.split(",") if part.strip()]
    if expected is not None and len(values) != expected:
        raise ValueError(f"Expected {expected} comma-separated integer values, got {len(values)}: {text!r}")
    return values


def parse_csv_strings(text: str) -> list[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def parse_prompt_sequence(text: str) -> list[str]:
    return [part.strip() for part in text.split("|") if part.strip()]


def build_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt

    stages: list[str] = []
    if args.prompt_file:
        path = Path(args.prompt_file).expanduser()
        with path.open("r", encoding="utf-8") as f:
            stages.extend(line.strip() for line in f if line.strip() and not line.lstrip().startswith("#"))
    if args.prompt_sequence:
        stages.extend(parse_prompt_sequence(args.prompt_sequence))

    if stages:
        joined = " ".join(f"Stage {idx + 1}: {stage}" for idx, stage in enumerate(stages))
        return f"Complete the long-horizon manipulation task. {joined}"

    return input("Task instruction: ").strip()


def split_action_vector(action: np.ndarray, action_layout: str = "joint_gripper") -> dict[str, np.ndarray | float | None]:
    """Decode deployment action vectors.

    Supported layouts:
    - 8D joint_gripper: [joint7, gripper] recommended single-arm layout.
    - 8D eef_gripper: [EEF xyz + quat_xyzw, gripper] legacy layout.
    - 15D eef_joint_gripper: [EEF7, joint7, gripper] SkinWorld single-arm layout.
    - 30D: LingBot-VA standard layout; uses left-arm EEF/joint/gripper slots.
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size == 8:
        if action_layout == "joint_gripper":
            return {"eef": None, "joints": action[:7], "gripper": float(action[7]), "channels": action.size}
        if action_layout == "eef_gripper":
            return {"eef": action[:7], "joints": None, "gripper": float(action[7]), "channels": action.size}
        raise ValueError(f"8D action requires --action-layout joint_gripper or eef_gripper, got {action_layout!r}")
    if action_layout == "joint_gripper":
        raise ValueError(f"joint_gripper deployment expects 8D actions, got {action.size}D")
    if action.size == 15:
        if action_layout != "eef_joint_gripper":
            raise ValueError(f"15D action requires --action-layout eef_joint_gripper, got {action_layout!r}")
        return {"eef": action[:7], "joints": action[7:14], "gripper": float(action[14]), "channels": action.size}
    if action.size == 30:
        return {"eef": action[:7], "joints": action[14:21], "gripper": float(action[28]), "channels": action.size}
    raise ValueError(f"Expected action channel count in {SUPPORTED_ACTION_CHANNELS}, got {action.size}")


def overwrite_action_proprioception(
    action: np.ndarray,
    action_layout: str = "joint_gripper",
    eef: np.ndarray | None = None,
    joints: np.ndarray | None = None,
    gripper: float | None = None,
) -> np.ndarray:
    """Map measured RealMan proprioception back into the model action layout.

    The server-side ``state`` input is still an action-token stream, not a new
    proprioception modality. We therefore only overwrite slots that have the
    same semantics as training labels.
    """
    feedback = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if feedback.size == 8:
        if action_layout == "joint_gripper":
            if joints is not None:
                feedback[:7] = np.asarray(joints, dtype=np.float32).reshape(7)
        elif action_layout == "eef_gripper":
            if eef is not None:
                feedback[:7] = np.asarray(eef, dtype=np.float32).reshape(7)
        else:
            raise ValueError(f"8D feedback requires joint_gripper or eef_gripper, got {action_layout!r}")
        if gripper is not None:
            feedback[7] = float(gripper)
        return feedback
    if feedback.size == 15:
        if eef is not None:
            feedback[:7] = np.asarray(eef, dtype=np.float32).reshape(7)
        if joints is not None:
            feedback[7:14] = np.asarray(joints, dtype=np.float32).reshape(7)
        if gripper is not None:
            feedback[14] = float(gripper)
        return feedback
    if feedback.size == 30:
        if eef is not None:
            feedback[:7] = np.asarray(eef, dtype=np.float32).reshape(7)
        if joints is not None:
            feedback[14:21] = np.asarray(joints, dtype=np.float32).reshape(7)
        if gripper is not None:
            feedback[28] = float(gripper)
        return feedback
    raise ValueError(f"Expected action channel count in {SUPPORTED_ACTION_CHANNELS}, got {feedback.size}")


def add_pika_sdk_paths(project_root: str | Path):
    """Match the path layout used by PIKA_RM65B_data_acquisition/main_teleop.py."""
    root = Path(project_root).expanduser()
    candidates = [
        root,
        root / "Usb-API-Stable-Tujian_0.2.0",
        root.parent,
    ]
    for path in candidates:
        path_text = str(path)
        if path.exists() and path_text not in sys.path:
            sys.path.insert(0, path_text)


def resolve_project_path(path_text: str, project_root: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    root = Path(project_root).expanduser()
    candidate = root / path
    if candidate.exists():
        return candidate
    return path


def process_rgb(ret: bool, frame: np.ndarray | None, size: tuple[int, int]) -> np.ndarray:
    if not ret or frame is None:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    frame = cv2.resize(frame, size)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def describe_value(value: Any, depth: int = 0) -> str:
    if depth > 2:
        return type(value).__name__
    if value is None:
        return "None"
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, dict):
        parts = []
        for idx, (key, item) in enumerate(value.items()):
            if idx >= 6:
                parts.append("...")
                break
            parts.append(f"{key!r}: {describe_value(item, depth + 1)}")
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, (list, tuple)):
        parts = [describe_value(item, depth + 1) for item in list(value)[:6]]
        if len(value) > 6:
            parts.append("...")
        return f"{type(value).__name__}(" + ", ".join(parts) + ")"
    return f"{type(value).__name__}({value!r})"


def make_panel(title: str, panel_w: int, panel_h: int, subtitle: str | None = None) -> np.ndarray:
    panel = np.full((panel_h, panel_w, 3), PANEL_BG, dtype=np.uint8)
    cv2.putText(panel, title, (12, 28), FONT, 0.72, FONT_COLOR, 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(panel, subtitle, (12, 54), FONT, 0.45, MUTED, 1, cv2.LINE_AA)
    return panel


def put_lines(
    image: np.ndarray,
    lines: list[str],
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


def normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def quat_xyzw_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quat_xyzw(quat)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.sign(sinp) * np.pi / 2.0 if abs(sinp) >= 1.0 else np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def euler_xyz_to_quat_xyzw(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(euler, dtype=np.float64).reshape(3)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    quat = np.asarray(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )
    return normalize_quat_xyzw(quat)


def quat_xyzw_to_rotvec(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quat_xyzw(quat)
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    angle = 2.0 * np.arctan2(np.linalg.norm([x, y, z]), w)
    if angle < 1e-12:
        return np.zeros(3, dtype=np.float64)
    axis = np.asarray([x, y, z], dtype=np.float64) / np.sin(angle / 2.0)
    return axis * angle


def tactile_from_any(value: Any, shape: tuple[int, int, int] = TACTILE_SHAPE) -> np.ndarray:
    """Normalize SDK tactile return values to [2, 32, 58]."""
    if value is None:
        raise ValueError("empty tactile value")

    if isinstance(value, dict):
        preferred = ["left", "right", "0", "1"]
        parts = [value[key] for key in preferred if key in value]
        if len(parts) < 2:
            parts = [value[key] for key in sorted(value.keys())[:2]]
        value = parts

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        left = np.asarray(value[0], dtype=np.float32)
        right = np.asarray(value[1], dtype=np.float32)
        if left.shape == shape[1:] and right.shape == shape[1:]:
            return np.stack([left, right], axis=0)

    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == shape:
        return arr
    if arr.shape == (shape[1], shape[2], shape[0]):
        return np.transpose(arr, (2, 0, 1))
    if arr.size == int(np.prod(shape)):
        return arr.reshape(shape)
    raise ValueError(f"Cannot interpret tactile shape {arr.shape}; expected {shape}")


def load_tactile_config_shapes(path: Path, target_keys: list[str]) -> dict[str, tuple[int, int]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as exc:
        print(f"[WARN] tactile mapping config not readable: {path} ({exc})")
        return {}

    shapes: dict[str, tuple[int, int]] = {}
    data_shape = config.get("data_shape")
    if isinstance(data_shape, list):
        for key, item in zip(target_keys, data_shape):
            if isinstance(item, list) and len(item) == 2:
                shapes[str(key)] = (int(item[0]), int(item[1]))

    range_mapping = config.get("range_mapping")
    if isinstance(range_mapping, dict):
        for key, item in range_mapping.items():
            if isinstance(item, list) and len(item) >= 7:
                shapes.setdefault(str(key), (int(item[5]), int(item[6])))
    return shapes


def normalize_force_dict_keys(force_dict: dict[Any, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in force_dict.items()}


def looks_like_force_dict(
    value: Any,
    target_keys: list[str],
    sheet_shape: tuple[int, int] = TACTILE_SHAPE[1:],
) -> bool:
    if not isinstance(value, dict):
        return False
    normalized = normalize_force_dict_keys(value)
    if all(key in normalized for key in target_keys):
        return True
    matching_sheets = 0
    expected_size = int(np.prod(sheet_shape))
    for item in normalized.values():
        try:
            arr = np.asarray(item)
        except Exception:
            continue
        if arr.shape == sheet_shape or arr.size == expected_size:
            matching_sheets += 1
    return matching_sheets >= len(target_keys)


def extract_force_dict(value: Any, target_keys: list[str], depth: int = 0) -> dict[str, Any]:
    if depth > 4:
        raise ValueError(f"cannot find tactile force dict in {describe_value(value)}")

    if looks_like_force_dict(value, target_keys):
        return normalize_force_dict_keys(value)

    if isinstance(value, dict):
        preferred = (
            "normal_force",
            "normal_force_dict",
            "force",
            "force_dict",
            "forces",
            "data",
            "tactile",
            "raw",
        )
        for key in preferred:
            if key in value:
                try:
                    return extract_force_dict(value[key], target_keys, depth + 1)
                except Exception:
                    pass
        for item in value.values():
            try:
                return extract_force_dict(item, target_keys, depth + 1)
            except Exception:
                continue

    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return extract_force_dict(item, target_keys, depth + 1)
            except Exception:
                continue

    raise ValueError(f"cannot find tactile force dict in {describe_value(value)}")


def tactile_from_force_dict(
    force_dict: Any,
    target_keys: list[str],
    shape: tuple[int, int, int] = TACTILE_SHAPE,
    per_key_shapes: dict[str, tuple[int, int]] | None = None,
) -> np.ndarray:
    """Convert TactileDataProvider normal-force dict to LingBot tactile tensor.

    The raw acquisition script saves get_latest_data()[0], whose useful keys are
    "0" and "1". Each key is one 32x58 tactile sheet.
    """
    if not isinstance(force_dict, dict):
        return tactile_from_any(force_dict, shape)

    normalized = {str(key): value for key, value in force_dict.items()}
    if len(target_keys) != shape[0]:
        raise ValueError(f"Expected {shape[0]} tactile target keys, got {target_keys}")

    sheets = []
    for key in target_keys:
        if key not in normalized:
            raise KeyError(f"missing tactile key {key!r}; available={sorted(normalized.keys())}")
        arr = np.asarray(normalized[key], dtype=np.float32)
        expected_shape = (per_key_shapes or {}).get(str(key), shape[1:])
        if arr.shape != expected_shape:
            if arr.size != int(np.prod(expected_shape)):
                raise ValueError(f"tactile key {key!r} has shape {arr.shape}, expected {expected_shape}")
            arr = arr.reshape(expected_shape)
        if arr.shape != shape[1:]:
            if arr.size != int(np.prod(shape[1:])):
                raise ValueError(f"tactile key {key!r} has shape {arr.shape}, expected model shape {shape[1:]}")
            arr = arr.reshape(shape[1:])
        sheets.append(arr)
    return np.stack(sheets, axis=0).astype(np.float32, copy=False)


class PikaTactileReader:
    """Runtime tactile reader matching PIKA_RM65B_data_acquisition/main_teleop.py."""

    STARTUP_WAIT_SEC = 5.0
    RUNTIME_WAIT_SEC = 0.02
    POLL_INTERVAL_SEC = 0.005
    MAX_STALE_FRAMES = 10

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.provider = None
        self.target_keys = parse_csv_strings(args.tactile_target_keys)
        self.last_valid = np.zeros(TACTILE_SHAPE, dtype=np.float32)
        self.last_force_dict = None
        self.no_data_count = 0
        self.cached_use_count = 0

        if args.mock_hardware or not args.enable_tactile:
            return

        add_pika_sdk_paths(args.pika_project_root)

        try:
            from Usb_API_stable_Tujian.TactileDataProvider import TactileDataProvider
        except ImportError as exc:
            missing_module = getattr(exc, "name", None)
            if missing_module == "usb":
                dependency_hint = "Install pyusb in the active Lingbot-va environment."
            elif missing_module == "crcmod":
                dependency_hint = "Install crcmod in the active Lingbot-va environment."
            elif missing_module:
                dependency_hint = f"Install missing tactile SDK dependency {missing_module!r} in the active Lingbot-va environment."
            else:
                dependency_hint = "Check tactile SDK dependencies in the active Lingbot-va environment."
            raise ImportError(
                "Cannot import Usb_API_stable_Tujian.TactileDataProvider. "
                "Install the tactile SDK wheel into the active Lingbot-va environment; do not run it directly from "
                "the .whl zip because the SDK reads JSON resource files by filesystem path. "
                f"{dependency_hint}"
            ) from exc

        sensor_shape = parse_csv_ints(args.tactile_sensor_shape, expected=2)
        splitted_file = resolve_project_path(args.tactile_splitted_file, args.pika_project_root)
        calibrate_file = resolve_project_path(args.tactile_calibrate_file, args.pika_project_root)
        self.per_key_shapes = load_tactile_config_shapes(splitted_file, self.target_keys)

        try:
            self.provider = TactileDataProvider(
                port=args.tactile_port,
                sensor_class=args.tactile_sensor_class,
                sensor_shape=sensor_shape,
                splitted_file_path=str(splitted_file),
                calibrate_file_path=str(calibrate_file),
                filtering_coloring=True,
                y_lim=[args.tactile_vmin, args.tactile_vmax],
                tfd=True,
                timeout=args.tactile_timeout,
            )
            self.provider.start()
            time.sleep(args.tactile_warmup_sec)
            print("[INFO] waiting for tactile data...")
            first_force_dict = self._read_latest_force_dict(timeout_s=self.STARTUP_WAIT_SEC, allow_cached=False)
            first_tactile = tactile_from_force_dict(
                first_force_dict,
                self.target_keys,
                per_key_shapes=self.per_key_shapes,
            )
            self.last_valid = first_tactile.copy()
            if args.tactile_zero_calibration:
                self.provider.zero_calibration()
                print("[INFO] tactile zero calibration complete")
                post_zero_force_dict = self._read_latest_force_dict(
                    timeout_s=min(self.RUNTIME_WAIT_SEC, 0.2),
                    allow_cached=True,
                )
                self.last_valid = tactile_from_force_dict(
                    post_zero_force_dict,
                    self.target_keys,
                    per_key_shapes=self.per_key_shapes,
                ).copy()
        except Exception:
            self.close()
            raise

        print(
            "[INFO] tactile provider started: "
            f"port={args.tactile_port}, keys={self.target_keys}, "
            f"splitted={splitted_file}, calibrate={calibrate_file}"
        )

    @staticmethod
    def _copy_force_dict(force_dict: dict[Any, Any]) -> dict[str, np.ndarray]:
        return {str(key): np.asarray(value, dtype=np.float32).copy() for key, value in force_dict.items()}

    def _read_latest_force_dict(self, timeout_s: float, allow_cached: bool) -> dict[str, Any]:
        if self.provider is None:
            raise RuntimeError("tactile provider is not initialized")

        deadline = time.perf_counter() + max(float(timeout_s), 0.0)
        last_error = None
        while True:
            latest = self.provider.get_latest_data()
            if latest and not (isinstance(latest, (list, tuple)) and len(latest) > 0 and latest[0] is None):
                try:
                    force_dict = extract_force_dict(latest, self.target_keys)
                    if force_dict:
                        self.last_force_dict = self._copy_force_dict(force_dict)
                        if self.no_data_count > 0:
                            print(f"[INFO] tactile data recovered after {self.no_data_count} empty read(s)")
                        self.no_data_count = 0
                        self.cached_use_count = 0
                        return force_dict
                except Exception as exc:
                    last_error = exc

            if time.perf_counter() >= deadline:
                break
            time.sleep(self.POLL_INTERVAL_SEC)

        self.no_data_count += 1
        if allow_cached and self.last_force_dict is not None:
            self.cached_use_count += 1
            if self.cached_use_count > self.MAX_STALE_FRAMES:
                raise RuntimeError(
                    "tactile data has been stale for too long: "
                    f"cached_use={self.cached_use_count}, max_stale={self.MAX_STALE_FRAMES}"
                )
            if self.cached_use_count == 1 or self.cached_use_count % 20 == 0:
                print(
                    "[WARN] tactile frame empty, reusing last valid frame: "
                    f"cached_use={self.cached_use_count}, no_data={self.no_data_count}"
                )
            return self.last_force_dict

        detail = f"; last_error={last_error}" if last_error is not None else ""
        raise RuntimeError(
            "tactile data unavailable; cannot build model input: "
            f"wait_s={timeout_s}, port={self.args.tactile_port}, "
            f"splitted={self.args.tactile_splitted_file}, calibrate={self.args.tactile_calibrate_file}{detail}"
        )

    def read(self) -> np.ndarray:
        if self.provider is None:
            return self.last_valid.copy()

        force_dict = self._read_latest_force_dict(timeout_s=self.RUNTIME_WAIT_SEC, allow_cached=True)
        tactile = tactile_from_force_dict(force_dict, self.target_keys, per_key_shapes=self.per_key_shapes)
        self.last_valid = tactile.copy()
        return tactile.copy()

    def close(self):
        if self.provider is not None:
            self.provider.stop()
            self.provider = None


class CameraSystem:
    CAMERA_API_WIDTH = 640
    CAMERA_API_HEIGHT = 480
    CAMERA_FPS = 30
    STARTUP_WARMUP_FRAMES = 8
    STARTUP_WARMUP_SLEEP_SEC = 0.02
    CAMERA_MAX_STALE_FRAMES = 2

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.size = (args.camera_width, args.camera_height)
        self.front = None
        self.fisheye = None
        self.side = None if args.side_camera_id < 0 else cv2.VideoCapture(args.side_camera_id)
        self.frame_status = {"front": "disabled", "fisheye": "disabled", "side": "disabled"}
        self.warned_camera_errors: set[str] = set()
        self.last_rgb_frames: dict[str, np.ndarray] = {}
        self.camera_cached_use_count: dict[str, int] = {}

        self._init_pika_cameras(args)
        self._init_side_camera(args)
        self._warmup()
        self._prime_enabled_cameras()

    def _init_pika_cameras(self, args: argparse.Namespace) -> None:
        if not args.realsense and args.fisheye_camera_id < 0:
            return

        add_pika_sdk_paths(args.pika_project_root)

        if args.realsense:
            try:
                try:
                    from pika.camera.realsense import RealSenseCamera
                except ImportError:
                    from pika_sdk.pika.camera.realsense import RealSenseCamera

                self.front = RealSenseCamera(
                    self.CAMERA_API_WIDTH,
                    self.CAMERA_API_HEIGHT,
                    self.CAMERA_FPS,
                    args.front_realsense_serial or None,
                )
                if not self._connect_realsense_color_only(self.front):
                    self.front = None
                self.frame_status["front"] = "starting" if self.front else "missing"
            except Exception as exc:
                print(f"[WARN] PIKA RealSense camera initialization failed: {exc}")
                self.front = None
            if self.front is None:
                raise RuntimeError(f"PIKA RealSense camera initialization failed: serial={args.front_realsense_serial}")

        if args.fisheye_camera_id >= 0:
            try:
                try:
                    from pika.camera.fisheye import FisheyeCamera
                except ImportError:
                    from pika_sdk.pika.camera.fisheye import FisheyeCamera

                self.fisheye = FisheyeCamera(
                    self.CAMERA_API_WIDTH,
                    self.CAMERA_API_HEIGHT,
                    self.CAMERA_FPS,
                    args.fisheye_camera_id,
                )
                if not self.fisheye.connect():
                    self.fisheye = None
                self.frame_status["fisheye"] = "starting" if self.fisheye else "missing"
            except Exception as exc:
                print(f"[WARN] PIKA fisheye camera initialization failed: {exc}")
                self.fisheye = None
            if self.fisheye is None or not bool(getattr(self.fisheye, "is_connected", True)):
                raise RuntimeError(f"PIKA fisheye camera initialization failed: index={args.fisheye_camera_id}")

        print(
            "[INFO] PIKA cameras initialized: "
            f"front_serial={args.front_realsense_serial or 'default'}, fisheye_id={args.fisheye_camera_id}"
        )

    def _connect_realsense_color_only(self, camera: Any) -> bool:
        if getattr(camera, "rs", None) is None:
            return False
        try:
            camera.pipeline = camera.rs.pipeline()
            camera.config = camera.rs.config()
            serial_number = getattr(camera, "serial_number", None)
            if serial_number:
                camera.config.enable_device(serial_number)
            camera.config.enable_stream(
                camera.rs.stream.color,
                camera.camera_width,
                camera.camera_height,
                camera.rs.format.bgr8,
                camera.camera_fps,
            )
            camera.pipeline.start(camera.config)
            camera.is_connected = True
            return True
        except Exception as exc:
            print(f"[WARN] PIKA RealSense color stream connect failed: {exc}")
            try:
                camera.disconnect()
            except Exception:
                pass
            return False

    def _init_side_camera(self, args: argparse.Namespace) -> None:
        if self.side is None:
            return
        if not self.side.isOpened():
            self.side.release()
            self.side = None
            raise RuntimeError(f"side camera initialization failed: index={args.side_camera_id}")
        self.side.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.side.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_API_WIDTH)
        self.side.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_API_HEIGHT)
        self.side.set(cv2.CAP_PROP_FPS, self.CAMERA_FPS)
        self.frame_status["side"] = "starting"

    def _warmup(self) -> None:
        for _ in range(self.STARTUP_WARMUP_FRAMES):
            self._read_pika_realsense()
            self._read_pika_fisheye()
            self._read_cv("side", self.side)
            time.sleep(self.STARTUP_WARMUP_SLEEP_SEC)

    def _prime_one_camera(self, name: str, reader, enabled: bool, max_attempts: int) -> None:
        if not enabled:
            return
        for attempt in range(max_attempts):
            ret, raw = reader()
            if ret and raw is not None:
                rgb = process_rgb(True, raw, self.size)
                self.last_rgb_frames[name] = rgb.copy()
                self.camera_cached_use_count[name] = 0
                return
            if attempt + 1 < max_attempts:
                time.sleep(0.03)
        raise RuntimeError(f"{name} camera failed to provide an initial valid frame")

    def _prime_enabled_cameras(self) -> None:
        self._prime_one_camera("front", self._read_pika_realsense, self.front is not None, max_attempts=5)
        self._prime_one_camera("fisheye", self._read_pika_fisheye, self.fisheye is not None, max_attempts=20)
        self._prime_one_camera("side", lambda: self._read_cv("side", self.side), self.side is not None, max_attempts=5)

    def _read_cv(self, name: str, cap):
        if cap is None:
            return False, None
        try:
            ret, frame = cap.read()
        except Exception as exc:
            self._warn_camera(name, exc)
            ret, frame = False, None
        self._record_camera_status(name, ret, frame)
        return ret, frame

    def _read_pika_realsense(self):
        if self.front is None:
            return False, None
        try:
            if not getattr(self.front, "is_connected", False) or getattr(self.front, "pipeline", None) is None:
                raise RuntimeError("PIKA RealSense camera is not connected")
            deadline = time.perf_counter() + 0.25
            rgb = None
            while time.perf_counter() < deadline:
                frames = self.front.pipeline.poll_for_frames()
                if frames:
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        rgb = np.asanyarray(color_frame.get_data())
                        break
                time.sleep(0.005)
            ret = rgb is not None
        except Exception as exc:
            self._warn_camera("front", exc)
            ret, rgb = False, None
        self._record_camera_status("front", ret, rgb)
        return ret, rgb

    def _read_pika_fisheye(self):
        if self.fisheye is None:
            return False, None
        try:
            ret, frame = self.fisheye.get_frame()
        except Exception as exc:
            self._warn_camera("fisheye", exc)
            ret, frame = False, None
        self._record_camera_status("fisheye", ret, frame)
        return ret, frame

    def _warn_camera(self, name: str, exc: Exception) -> None:
        if name not in self.warned_camera_errors:
            print(f"[WARN] {name} camera read failed: {exc}")
            self.warned_camera_errors.add(name)

    def _record_camera_status(self, name: str, ok: bool, frame: np.ndarray | None) -> None:
        if ok and frame is not None:
            self.frame_status[name] = f"ok {frame.shape[1]}x{frame.shape[0]}"
        else:
            self.frame_status[name] = "missing"

    def _process_or_cached(self, name: str, ret: bool, raw_frame: np.ndarray | None, enabled: bool) -> np.ndarray:
        if not enabled:
            return np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)

        if ret and raw_frame is not None:
            rgb = process_rgb(True, raw_frame, self.size)
            self.last_rgb_frames[name] = rgb.copy()
            self.camera_cached_use_count[name] = 0
            return rgb

        cached = self.last_rgb_frames.get(name)
        if cached is not None:
            used = self.camera_cached_use_count.get(name, 0) + 1
            self.camera_cached_use_count[name] = used
            if used > self.CAMERA_MAX_STALE_FRAMES:
                raise RuntimeError(
                    f"{name} camera has no fresh frame and cache is stale: "
                    f"cached_use={used}, max_stale={self.CAMERA_MAX_STALE_FRAMES}"
                )
            if used == 1:
                print(f"[WARN] {name} camera frame empty, reusing last valid frame")
            return cached.copy()

        raise RuntimeError(f"{name} camera capture failed and no cached frame is available")

    def get_frames(self) -> dict[str, np.ndarray]:
        ret_front, raw_front = self._read_pika_realsense()
        ret_fisheye, raw_fisheye = self._read_pika_fisheye()
        ret_side, raw_side = self._read_cv("side", self.side)
        front = self._process_or_cached("front", ret_front, raw_front, self.front is not None)
        fisheye = self._process_or_cached("fisheye", ret_fisheye, raw_fisheye, self.fisheye is not None)
        side = self._process_or_cached("side", ret_side, raw_side, self.side is not None)

        return {
            "observation.images.cam_front": front,
            "observation.images.cam_side": side,
            "observation.images.cam_fisheye": fisheye,
        }

    def status_text(self) -> str:
        return " | ".join(f"{name}:{self.frame_status.get(name, 'unknown')}" for name in ("front", "fisheye", "side"))

    def close(self):
        for camera in (self.front, self.fisheye):
            if camera is not None:
                try:
                    camera.disconnect()
                except Exception as exc:
                    print(f"[WARN] camera disconnect failed: {exc}")
        self.front = None
        self.fisheye = None
        if self.side is not None:
            self.side.release()
            self.side = None


class StopRequested(RuntimeError):
    pass


class ReplanRequested(RuntimeError):
    """Raised only when a safety transition invalidates the current chunk."""


class RealtimeVisualizer:
    """OpenCV runtime monitor with q/Esc polling."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.panel_w = args.visualize_panel_width
        self.panel_h = args.visualize_panel_height
        self.window_name = args.visualize_window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.panel_w * 2, self.panel_h * 2)

    def image_panel(self, title: str, image_rgb: np.ndarray | None, subtitle: str = "") -> np.ndarray:
        panel = make_panel(title, self.panel_w, self.panel_h, subtitle)
        if image_rgb is None or image_rgb.size == 0:
            put_lines(panel, ["N/A"], 12, self.panel_h // 2, color=(80, 80, 80), scale=0.7)
            return panel
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        panel[64:, :] = letterbox(image_bgr, self.panel_w, self.panel_h - 64)
        if not np.any(image_rgb):
            put_lines(
                panel,
                ["NO FRAME", "check camera id/serial"],
                18,
                self.panel_h // 2,
                scale=0.62,
                color=(80, 120, 255),
                step=30,
            )
        return panel

    def tactile_sensor_panel(self, matrix: np.ndarray | None, key: str) -> np.ndarray:
        h, w = TACTILE_SHAPE[1:]
        matrix_ok = matrix is not None
        if matrix_ok:
            matrix = np.asarray(matrix, dtype=np.float32)
            if matrix.shape != (h, w):
                try:
                    matrix = matrix.reshape((h, w))
                except Exception:
                    matrix_ok = False

        if matrix_ok:
            matrix = cv2.rotate(matrix, cv2.ROTATE_90_CLOCKWISE)
            h, w = w, h

        pixel = self.args.visualize_tactile_pixel_size
        label_h = 86
        target_w = w * pixel
        target_h = h * pixel
        panel = np.zeros((target_h + label_h, target_w, 3), dtype=np.uint8)

        if not matrix_ok:
            cv2.putText(panel, "NO DATA", (12, max(32, target_h // 2)), FONT, 0.7, (70, 70, 70), 2, cv2.LINE_AA)
            force = 0.0
            peak = 0.0
        else:
            force = float(np.nansum(matrix))
            peak = float(np.nanmax(matrix))
            vmin = float(self.args.tactile_vmin)
            vmax = max(float(self.args.tactile_vmax), vmin + 1e-8)
            clipped = np.clip(matrix, vmin, vmax)
            norm_img = ((clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
            color_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
            color_img[matrix < self.args.visualize_tactile_threshold] = (0, 0, 0)
            panel[:target_h, :target_w] = cv2.resize(
                color_img,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )

        label_y = target_h + 28
        cv2.putText(panel, f"sensor {key}", (14, label_y), FONT, 0.72, FONT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(panel, f"sum {force:.4f}", (14, label_y + 26), FONT, 0.48, FONT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(panel, f"max {peak:.5f}", (14, label_y + 50), FONT, 0.48, FONT_COLOR, 1, cv2.LINE_AA)
        return panel

    def tactile_panel(self, tactile: np.ndarray | None) -> np.ndarray:
        panel = make_panel(
            "tactile heatmap",
            self.panel_w,
            self.panel_h,
            f"q/Esc: emergency stop | JET vmax={self.args.tactile_vmax:g}",
        )
        if tactile is None:
            put_lines(panel, ["N/A"], 12, self.panel_h // 2, color=(80, 80, 80), scale=0.7)
            return panel
        tactile = np.asarray(tactile, dtype=np.float32)
        if tactile.shape != TACTILE_SHAPE:
            try:
                tactile = tactile.reshape(TACTILE_SHAPE)
            except Exception:
                put_lines(panel, [f"bad tactile shape: {tactile.shape}"], 12, self.panel_h // 2, color=(80, 80, 255))
                return panel
        sensor_panels = [
            self.tactile_sensor_panel(tactile[i], key)
            for i, key in enumerate(parse_csv_strings(self.args.tactile_target_keys)[: TACTILE_SHAPE[0]])
        ]
        raw = np.hstack(sensor_panels)
        panel[64:, :] = letterbox(raw, self.panel_w, self.panel_h - 64)
        return panel

    def update(self, obs: dict[str, np.ndarray], tactile: np.ndarray, status: str = "") -> int | None:
        subtitle = status[:96]
        top = [
            self.image_panel("fisheye", obs.get("observation.images.cam_fisheye"), subtitle),
            self.image_panel("front", obs.get("observation.images.cam_front"), "live camera"),
        ]
        bottom = [
            self.tactile_panel(tactile),
            self.image_panel("side", obs.get("observation.images.cam_side"), "missing side falls back by config"),
        ]
        frame = np.vstack([np.hstack(top), np.hstack(bottom)])
        cv2.imshow(self.window_name, frame)
        return self.poll_key()

    @staticmethod
    def poll_key() -> int | None:
        key = cv2.waitKey(1) & 0xFF
        return None if key == 255 else key

    def close(self):
        cv2.destroyWindow(self.window_name)


class RealmanPikaHardware:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.robot = None
        self.gripper = None
        self.tactile_reader = PikaTactileReader(args)
        self.last_pose_position = None
        self.last_joint_command = None
        self.last_gripper = None
        self._first_joint_command_checked = False
        self._warned_feedback_8d = False
        self._warned_feedback_read = False
        self._warned_feedback_pose = False

        if args.mock_hardware:
            print("[WARN] mock hardware mode: robot commands are not sent")
            return

        try:
            add_pika_sdk_paths(args.pika_project_root)

            try:
                from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
            except ImportError as exc:
                raise ImportError(
                    "Cannot import Robotic_Arm.rm_robot_interface. "
                    "Install it in the active Lingbot-va environment with `pip install Robotic_Arm`."
                ) from exc
            try:
                from pika.gripper import Gripper
            except ImportError:
                from pika_sdk.pika.gripper import Gripper

            self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
            handle = self.robot.rm_create_robot_arm(args.robot_ip, args.robot_port)
            print("RealMan robot id:", handle.id)

            gripper_needed = args.enable_gripper and (
                args.action_mode != "print" or args.feedback_mode != "command"
            )
            if gripper_needed:
                self.gripper = Gripper(args.gripper_port)
                if not self.gripper.connect():
                    raise RuntimeError(f"Failed to connect Pika gripper at {args.gripper_port}")
                if not self.gripper.enable():
                    raise RuntimeError("Failed to enable Pika gripper")
                time.sleep(1.0)
            elif args.enable_gripper:
                print("[INFO] gripper serial skipped: action_mode=print and feedback_mode=command")
            else:
                print("[WARN] gripper disabled by --no-enable-gripper")
        except Exception:
            self.close()
            raise

    def home(self):
        if self.args.mock_hardware:
            return
        joints = parse_csv_floats(self.args.home_joints, expected=7)
        joints_list = joints.tolist()
        print(f"[INFO] moving RealMan to safe RESET_JOINT: {np.round(joints, 3).tolist()}")
        ret = self.robot.rm_movej(joints_list, 100, 0, 0, 1)
        if ret not in (None, 0):
            raise RuntimeError(f"RealMan RESET_JOINT move failed: ret={ret}")
        self.last_joint_command = joints.copy()
        self.last_gripper = float(self.args.home_gripper)
        if self.gripper is not None:
            self.gripper.set_gripper_distance(float(self.args.home_gripper))

    def get_joint_state(self) -> np.ndarray:
        if self.args.mock_hardware:
            return np.zeros(7, dtype=np.float32)
        ret, degree = self.robot.rm_get_joint_degree()
        if ret != 0:
            print(f"[WARN] rm_get_joint_degree ret={ret}")
        return np.asarray(degree, dtype=np.float32)

    @staticmethod
    def _pose6_to_eef7(pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        if pose.size < 6:
            raise ValueError(f"Expected RealMan pose [x,y,z,rx,ry,rz], got {pose}")
        quat = euler_xyz_to_quat_xyzw(pose[3:6])
        return np.concatenate([pose[:3], quat]).astype(np.float32)

    def get_robot_proprioception(self, include_eef: bool) -> tuple[np.ndarray | None, np.ndarray]:
        if self.args.mock_hardware:
            eef = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32) if include_eef else None
            return eef, np.zeros(7, dtype=np.float32)

        if include_eef:
            ret, state = self.robot.rm_get_current_arm_state()
            if ret != 0:
                raise RuntimeError(f"rm_get_current_arm_state ret={ret}")
            eef = self._pose6_to_eef7(np.asarray(state["pose"], dtype=np.float32))
            joints = np.asarray(state["joint"], dtype=np.float32).reshape(7)
            return eef, joints

        return None, self.get_joint_state()

    def get_gripper_state(self) -> float:
        if self.args.mock_hardware or self.gripper is None:
            return float(self.args.home_gripper)
        return float(self.gripper.get_gripper_distance())

    def build_feedback_action(self, commanded_action: np.ndarray) -> np.ndarray:
        """Return the action-token state used for KV-cache update.

        ``command`` preserves the original LingBot-VA deployment behavior.
        ``measured_joint`` replaces joint/gripper channels with real hardware
        readings while keeping EEF channels as the issued command. ``measured_full``
        also replaces EEF channels using RealMan's current TCP pose.
        """
        commanded_action = np.asarray(commanded_action, dtype=np.float32).reshape(-1)
        if self.args.feedback_mode == "command":
            return commanded_action.copy()

        eef = None
        joints = None
        gripper = None
        include_eef = self.args.feedback_mode == "measured_full"
        try:
            eef, joints = self.get_robot_proprioception(include_eef=include_eef)
        except Exception as exc:
            if include_eef and not self._warned_feedback_pose:
                print(f"[WARN] measured full feedback pose read failed; falling back to measured_joint: {exc}")
                self._warned_feedback_pose = True
            try:
                eef = None
                joints = self.get_joint_state()
            except Exception as joint_exc:
                if not self._warned_feedback_read:
                    print(f"[WARN] measured feedback read failed; falling back to command state: {joint_exc}")
                    self._warned_feedback_read = True
                return commanded_action.copy()

        try:
            gripper = self.get_gripper_state()
        except Exception as exc:
            if not self._warned_feedback_read:
                print(f"[WARN] measured gripper feedback read failed; keeping command gripper: {exc}")
                self._warned_feedback_read = True

        if commanded_action.size == 8 and not self._warned_feedback_8d:
            print("[WARN] 8D action has no joint slots; measured feedback will skip joints")
            self._warned_feedback_8d = True

        return overwrite_action_proprioception(
            commanded_action,
            action_layout=self.args.action_layout,
            eef=eef,
            joints=joints,
            gripper=gripper,
        )

    def _motion_block_arg(self) -> int:
        return 1 if self.args.blocking_motion else 0

    def _clip_gripper(self, value: float) -> float:
        return float(np.clip(value + self.args.gripper_offset, self.args.gripper_min, self.args.gripper_max))

    def _limit_gripper_step(self, gripper: float) -> float:
        if self.last_gripper is None or self.args.max_gripper_step <= 0:
            return gripper
        return float(
            np.clip(
                gripper,
                self.last_gripper - self.args.max_gripper_step,
                self.last_gripper + self.args.max_gripper_step,
            )
        )

    def _send_joint_target(self, joints: np.ndarray, gripper: float):
        if self.args.mock_hardware:
            print("[MOCK] move_joints", np.round(joints, 3), round(float(gripper), 3))
            self.last_joint_command = np.asarray(joints, dtype=np.float32).copy()
            self.last_gripper = float(gripper)
            return

        joints_list = np.asarray(joints, dtype=np.float32).tolist()
        if self.args.joint_motion_method == "rm_movej_canfd":
            ret = self.robot.rm_movej_canfd(
                joints_list,
                bool(self.args.canfd_follow),
                0,
                self.args.canfd_trajectory_mode,
                self.args.canfd_radio,
            )
        else:
            ret = self.robot.rm_movej(
                joints_list,
                self.args.robot_speed,
                self.args.blend_radius,
                self.args.trajectory_connect,
                self._motion_block_arg(),
            )
        if ret not in (None, 0):
            print(f"[WARN] {self.args.joint_motion_method} returned {ret}")
        if self.gripper is not None:
            self.gripper.set_gripper_distance(float(gripper))
        self.last_joint_command = np.asarray(joints, dtype=np.float32).copy()
        self.last_gripper = float(gripper)

    def _joint_safety_reference(self) -> np.ndarray:
        if self.last_joint_command is not None:
            return np.asarray(self.last_joint_command, dtype=np.float32).copy()
        return self.get_joint_state().astype(np.float32, copy=False)

    def _transition_to_joint_target(
        self,
        current: np.ndarray,
        target: np.ndarray,
        gripper: float,
        reason: str,
    ) -> None:
        max_step = float(self.args.max_joint_step_deg)
        if max_step <= 0:
            raise RuntimeError("joint transition requires --max-joint-step-deg > 0")

        delta = target - current
        steps = int(np.ceil(float(np.max(np.abs(delta))) / max_step))
        steps = max(1, steps)
        print(
            f"[SAFETY] {reason}; "
            f"transitioning in {steps} step(s), max_joint_step_deg={max_step}"
        )
        transition_dt = max(float(self.args.joint_transition_dt), 0.0)
        for idx in range(1, steps + 1):
            alpha = idx / steps
            waypoint = current + delta * alpha
            self._send_joint_target(waypoint.astype(np.float32), gripper)
            if transition_dt > 0 and idx < steps:
                time.sleep(transition_dt)

    def _transition_to_first_joint_target(self, current: np.ndarray, target: np.ndarray, gripper: float) -> None:
        self._transition_to_joint_target(
            current,
            target,
            gripper,
            reason="first joint target is far from current state",
        )

    def _handle_first_joint_target(self, target: np.ndarray, gripper: float) -> bool:
        if self._first_joint_command_checked:
            return False
        self._first_joint_command_checked = True

        current = self._joint_safety_reference()
        delta = target - current
        max_abs = float(np.max(np.abs(delta)))
        limit = float(self.args.first_joint_delta_limit_deg)
        if limit <= 0 or max_abs <= limit:
            self.last_joint_command = current.copy()
            return False

        axis = int(np.argmax(np.abs(delta))) + 1
        message = (
            "first joint target exceeds safety limit: "
            f"max_delta={max_abs:.2f}deg on joint {axis}, limit={limit:.2f}deg, "
            f"current={np.round(current, 3).tolist()}, target={np.round(target, 3).tolist()}"
        )
        if self.args.first_joint_delta_mode == "abort":
            self.stop_motion("first joint target exceeds safety limit")
            raise RuntimeError(message + "; use --first-joint-delta-mode transition to approach it slowly")
        if self.args.first_joint_delta_mode == "transition":
            print(f"[SAFETY] {message}")
            self._transition_to_first_joint_target(current, target, gripper)
            self._first_joint_command_checked = False
            raise ReplanRequested("first joint transition complete; discard stale chunk and infer fresh chunk")
        raise ValueError(f"Unknown first_joint_delta_mode={self.args.first_joint_delta_mode!r}")

    def _limit_joint_step(self, target: np.ndarray, gripper: float) -> np.ndarray | None:
        max_step = float(self.args.max_joint_step_deg)
        if max_step <= 0:
            return target
        current = self._joint_safety_reference()
        delta = target - current
        max_abs = float(np.max(np.abs(delta)))
        if max_abs <= max_step:
            return target

        axis = int(np.argmax(np.abs(delta))) + 1
        if self.args.joint_step_limit_mode == "clamp":
            limited = current + np.clip(delta, -max_step, max_step)
            print(
                "[SAFETY] joint command clamped: "
                f"max_delta={max_abs:.2f}deg on joint {axis}, max_joint_step_deg={max_step:.2f}"
            )
            return limited.astype(np.float32, copy=False)
        if self.args.joint_step_limit_mode == "transition":
            self._transition_to_joint_target(
                current,
                target,
                gripper,
                reason=(
                    "joint command exceeds step limit: "
                    f"max_delta={max_abs:.2f}deg on joint {axis}, "
                    f"max_joint_step_deg={max_step:.2f}, "
                    f"current={np.round(current, 3).tolist()}, target={np.round(target, 3).tolist()}"
                ),
            )
            return None
        raise ValueError(f"Unknown joint_step_limit_mode={self.args.joint_step_limit_mode!r}")

    def move_joints(self, joints: np.ndarray, gripper: float):
        target_joints = np.asarray(joints, dtype=np.float32).reshape(7)
        target_gripper = self._limit_gripper_step(self._clip_gripper(gripper))
        if self._handle_first_joint_target(target_joints, target_gripper):
            return
        safe_joints = self._limit_joint_step(target_joints, target_gripper)
        if safe_joints is None:
            return
        self._send_joint_target(safe_joints, target_gripper)

    def read_tactile(self) -> np.ndarray:
        return self.tactile_reader.read()

    def _pose_command_from_eef(self, eef7: np.ndarray, gripper_value: float) -> tuple[np.ndarray, float]:
        pos = np.asarray(eef7[:3], dtype=np.float64)
        quat = normalize_quat_xyzw(eef7[3:7])
        gripper = self._clip_gripper(gripper_value)

        if self.last_pose_position is not None and self.args.max_position_step > 0:
            delta = pos - self.last_pose_position
            norm = float(np.linalg.norm(delta))
            if norm > self.args.max_position_step:
                pos = self.last_pose_position + delta * (self.args.max_position_step / norm)

        if self.last_gripper is not None and self.args.max_gripper_step > 0:
            gripper = float(
                np.clip(
                    gripper,
                    self.last_gripper - self.args.max_gripper_step,
                    self.last_gripper + self.args.max_gripper_step,
                )
            )

        self.last_pose_position = pos.copy()
        self.last_gripper = gripper

        if self.args.pose_orientation_format == "quat_xyzw":
            pose = np.concatenate([pos, quat])
        elif self.args.pose_orientation_format == "rotvec":
            pose = np.concatenate([pos, quat_xyzw_to_rotvec(quat)])
        else:
            pose = np.concatenate([pos, quat_xyzw_to_euler_xyz(quat)])
        return pose.astype(np.float32), gripper

    def move_pose(self, pose: np.ndarray, gripper: float):
        if self.args.mock_hardware:
            print("[MOCK] move_pose", np.round(pose, 4), "gripper", round(float(gripper), 3))
            return

        method_names = [self.args.pose_method] if self.args.pose_method else ["rm_movel", "rm_movej_p"]
        last_error = None
        for name in method_names:
            method = getattr(self.robot, name, None)
            if method is None:
                continue
            try:
                pose_list = np.asarray(pose, dtype=np.float32).tolist()
                if name in ("rm_movel", "rm_movej_p"):
                    ret = method(
                        pose_list,
                        self.args.robot_speed,
                        self.args.blend_radius,
                        self.args.trajectory_connect,
                        self._motion_block_arg(),
                    )
                elif name == "rm_movep_canfd":
                    ret = method(
                        pose_list,
                        bool(self.args.canfd_follow),
                        self.args.canfd_trajectory_mode,
                        self.args.canfd_radio,
                    )
                else:
                    raise ValueError(f"Unsupported pose method {name!r}")
                if ret not in (None, 0):
                    print(f"[WARN] {name} returned {ret}")
                if self.gripper is not None:
                    self.gripper.set_gripper_distance(float(gripper))
                return
            except Exception as exc:
                last_error = exc
                print(f"[WARN] {name} failed: {exc}")
                continue
        raise RuntimeError(f"No usable RealMan Cartesian move method found; last error={last_error}")

    def stop_motion(self, reason: str = ""):
        reason_text = f" ({reason})" if reason else ""
        if self.args.mock_hardware:
            print(f"[MOCK] emergency stop{reason_text}")
            return

        if self.args.stop_mode == "slow":
            stop_methods = ("rm_set_arm_slow_stop", "rm_set_arm_stop", "rm_set_arm_pause")
        else:
            stop_methods = ("rm_set_arm_stop", "rm_set_arm_slow_stop", "rm_set_arm_pause")
        stop_methods += ("move_stop", "rm_move_stop", "rm_stop", "move_pause")
        for name in stop_methods:
            method = getattr(self.robot, name, None)
            if method is None:
                continue
            for call_args in ((), (0,), (1,)):
                try:
                    ret = method(*call_args)
                    print(f"[SAFETY] robot stop via {name}{reason_text}; ret={ret}")
                    return
                except TypeError:
                    continue
                except Exception as exc:
                    print(f"[WARN] robot stop method {name} failed: {exc}")
                    break
        print(f"[WARN] no usable robot stop method found{reason_text}")

    def execute_action(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        decoded = split_action_vector(action, self.args.action_layout)
        gripper_value = float(decoded["gripper"])

        if self.args.action_mode == "print":
            print(
                f"[ACTION {decoded['channels']}D]",
                "eef=",
                np.round(decoded["eef"], 4),
                "joints=",
                None if decoded["joints"] is None else np.round(decoded["joints"], 4),
                "gripper=",
                round(gripper_value, 4),
            )
            return
        if self.args.action_mode == "joint":
            joints = decoded["joints"]
            if joints is None:
                raise ValueError("joint action_mode requires 15D [EEF7,joint7,gripper] or 30D LingBot-VA action")
            self.move_joints(joints, gripper_value)
            return
        if self.args.action_mode == "ee_pose":
            if decoded["eef"] is None:
                raise ValueError("ee_pose action_mode requires an EEF action layout, not joint_gripper")
            pose, gripper = self._pose_command_from_eef(decoded["eef"], gripper_value)
            self.move_pose(pose, gripper)
            return
        raise ValueError(f"Unknown action_mode={self.args.action_mode!r}")

    def close(self):
        self.tactile_reader.close()
        if self.gripper is not None:
            for name in ("disable", "disconnect"):
                method = getattr(self.gripper, name, None)
                if method is not None:
                    try:
                        method()
                    except Exception as exc:
                        print(f"[WARN] gripper {name} failed: {exc}")
            self.gripper = None
        if self.robot is not None:
            method = getattr(self.robot, "rm_delete_robot_arm", None)
            if method is not None:
                try:
                    method()
                except TypeError:
                    try:
                        method(self.robot)
                    except Exception as exc:
                        print(f"[WARN] rm_delete_robot_arm failed: {exc}")
                except Exception as exc:
                    print(f"[WARN] rm_delete_robot_arm failed: {exc}")
            self.robot = None


class RealmanTactileEnv:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cameras = None
        self.hw = None
        try:
            self.hw = RealmanPikaHardware(args)
            if args.reset_to_home:
                self.hw.home()
            self.cameras = CameraSystem(args)
        except Exception:
            self.close()
            raise

    def get_image_obs(self) -> dict[str, np.ndarray]:
        if self.cameras is None:
            raise RuntimeError("camera system is not initialized")
        return self.cameras.get_frames()

    def get_tactile(self) -> np.ndarray:
        if self.hw is None:
            raise RuntimeError("hardware system is not initialized")
        return self.hw.read_tactile()

    def camera_status(self) -> str:
        if self.cameras is None:
            return "cameras:uninitialized"
        return self.cameras.status_text()

    def stop_motion(self, reason: str = ""):
        if self.hw is not None:
            self.hw.stop_motion(reason)

    def close(self):
        if self.cameras is not None:
            self.cameras.close()
            self.cameras = None
        if self.hw is not None:
            self.hw.close()
            self.hw = None


def flatten_action_chunk(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 4 and action.shape[0] == 1:
        action = action[0]
    if action.ndim != 3:
        raise ValueError(f"Expected action [C,F,N] or [1,C,F,N], got {action.shape}")
    if action.shape[0] not in SUPPORTED_ACTION_CHANNELS:
        raise ValueError(f"Expected action channels in {SUPPORTED_ACTION_CHANNELS}, got {action.shape}")
    return action


def is_stop_key(key: int | None) -> bool:
    return key in (ord("q"), ord("Q"), 27)


def raise_if_stop_key(env: RealmanTactileEnv, key: int | None) -> None:
    if is_stop_key(key):
        env.stop_motion("q/Esc pressed")
        raise StopRequested("q/Esc pressed")


def check_stop_requested(
    env: RealmanTactileEnv,
    visualizer: RealtimeVisualizer | None,
    stop_state: dict[str, bool],
    reason: str = "operator request",
) -> None:
    key = visualizer.poll_key() if visualizer is not None else None
    raise_if_stop_key(env, key)
    if stop_state.get("requested", False):
        env.stop_motion(reason)
        raise StopRequested(reason)


def run(args: argparse.Namespace):
    from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy

    policy = WebsocketClientPolicy(host=args.server_host, port=args.server_port)
    env = RealmanTactileEnv(args)
    visualizer = None
    stop_state = {"requested": False}
    previous_signal_handlers = {}

    def request_stop(signum, _frame):
        stop_state["requested"] = True
        print(f"[SAFETY] received signal {signum}; stopping at next control poll")
        try:
            env.stop_motion(f"signal {signum}")
        except Exception as exc:
            print(f"[WARN] signal stop failed: {exc}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_signal_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, request_stop)

    try:
        visualizer = RealtimeVisualizer(args) if args.visualize else None
        prompt = build_prompt(args)
        print(f"[INFO] deployment prompt: {prompt}")
        print(f"[INFO] KV-cache feedback mode: {args.feedback_mode}")

        print("[INFO] reset tactile LingBot-VA server")
        policy.infer({"reset": True, "prompt": prompt})

        first = True
        latest_obs = env.get_image_obs()
        latest_tactile = env.get_tactile()
        if visualizer is not None:
            raise_if_stop_key(
                env,
                visualizer.update(latest_obs, latest_tactile, f"server reset complete | {env.camera_status()}"),
            )

        chunk_idx = 0
        while chunk_idx < args.max_chunks:
            check_stop_requested(env, visualizer, stop_state, "signal before inference")
            print(f"[INFO] infer chunk {chunk_idx}")
            request = {
                "obs": latest_obs,
                "prompt": prompt,
                "tactile": latest_tactile,
            }
            result = policy.infer(request)
            check_stop_requested(env, visualizer, stop_state, "signal after inference")
            action = flatten_action_chunk(result["action"])
            feedback_action = action.copy()
            frame_count = action.shape[1]
            steps_per_frame = action.shape[2]
            start_frame = 1 if first else 0
            print(
                f"[INFO] execute chunk {chunk_idx}: frames={frame_count}, "
                f"steps_per_frame={steps_per_frame}, start_frame={start_frame}"
            )
            key_frames: list[dict[str, np.ndarray]] = []
            tactile_history: list[np.ndarray] = []
            discard_chunk_reason = None

            if first:
                tactile_history.extend([latest_tactile.copy()] * steps_per_frame)
                if args.feedback_mode != "command":
                    for step_id in range(steps_per_frame):
                        feedback_action[:, 0, step_id] = env.hw.build_feedback_action(action[:, 0, step_id])

            for frame_id in range(start_frame, frame_count):
                if discard_chunk_reason is not None:
                    break
                for step_id in range(steps_per_frame):
                    check_stop_requested(env, visualizer, stop_state, "signal before action")
                    action_step = action[:, frame_id, step_id]
                    try:
                        env.hw.execute_action(action_step)
                    except ReplanRequested as exc:
                        discard_chunk_reason = str(exc)
                        latest_obs = env.get_image_obs()
                        latest_tactile = env.get_tactile()
                        if visualizer is not None:
                            raise_if_stop_key(
                                env,
                                visualizer.update(
                                    latest_obs,
                                    latest_tactile,
                                    f"{discard_chunk_reason} | {env.camera_status()}",
                                ),
                            )
                        break
                    time.sleep(args.control_dt)
                    feedback_action[:, frame_id, step_id] = env.hw.build_feedback_action(action_step)

                    latest_tactile = env.get_tactile()
                    tactile_history.append(latest_tactile.copy())

                    visual_obs = latest_obs
                    visual_obs_is_fresh = False
                    if visualizer is not None and (len(tactile_history) % args.visualize_every_steps == 0):
                        if args.visualize_live_cameras:
                            visual_obs = env.get_image_obs()
                            visual_obs_is_fresh = True
                        status = (
                            f"chunk {chunk_idx} | frame {frame_id + 1}/{frame_count} | "
                            f"step {step_id + 1}/{steps_per_frame} | action {action.shape[0]}D | "
                            f"feedback {args.feedback_mode} | {env.camera_status()}"
                        )
                        raise_if_stop_key(env, visualizer.update(visual_obs, latest_tactile, status))
                    elif visualizer is not None:
                        check_stop_requested(env, visualizer, stop_state, "signal during action")

                    if (step_id + 1) % steps_per_frame == 0:
                        latest_obs = visual_obs if visual_obs_is_fresh else env.get_image_obs()
                        key_frames.append(latest_obs)

            if discard_chunk_reason is not None:
                print(f"[SAFETY] {discard_chunk_reason}; reset server before the next fresh inference")
                policy.infer({"reset": True, "prompt": prompt})
                first = True
                continue

            if not key_frames:
                latest_obs = env.get_image_obs()
                key_frames.append(latest_obs)
                if visualizer is not None:
                    raise_if_stop_key(
                        env,
                        visualizer.update(latest_obs, latest_tactile, f"chunk {chunk_idx} done | {env.camera_status()}"),
                    )

            if len(key_frames) < 3:
                raise RuntimeError(
                    "KV-cache update needs at least 3 executed observation frames, "
                    f"got {len(key_frames)}. In the first deploy chunk frame 0 is the current "
                    "observation and is skipped, so keep the server frame_chunk_size >= 4."
                )

            tactile_cache = np.stack(tactile_history, axis=0).astype(np.float32)
            print(
                f"[INFO] update KV cache: key_frames={len(key_frames)} "
                f"action={action.shape} state={feedback_action.shape} tactile={tactile_cache.shape}"
            )
            check_stop_requested(env, visualizer, stop_state, "signal before kv-cache update")
            policy.infer(
                {
                    "compute_kv_cache": True,
                    "obs": key_frames,
                    "state": feedback_action,
                    "tactile": tactile_cache,
                }
            )
            first = False
            chunk_idx += 1

    except StopRequested as exc:
        print(f"[SAFETY] deployment stopped: {exc}")
    except KeyboardInterrupt:
        print("[INFO] interrupted by user")
    finally:
        if args.emergency_stop_on_exit:
            env.stop_motion("program exit")
        if visualizer is not None:
            try:
                visualizer.close()
            except Exception:
                pass
        env.close()
        cv2.destroyAllWindows()
        for sig, handler in previous_signal_handlers.items():
            signal.signal(sig, handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy LingBot-VA-Tactile on RealMan + Pika hardware")
    parser.add_argument("--server-host", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=29536)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument(
        "--prompt-sequence",
        type=str,
        default=None,
        help="Optional long-horizon stages separated by '|'. Used only for the initial server reset prompt.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Optional text file with one stage prompt per line. Used only for the initial server reset prompt.",
    )
    parser.add_argument("--max-chunks", type=int, default=20)
    parser.add_argument("--control-dt", type=float, default=1.0 / 30.0)

    parser.add_argument("--fisheye-camera-id", type=int, default=12, help="OpenCV camera index for the gripper fisheye view.")
    parser.add_argument("--side-camera-id", type=int, default=4, help="OpenCV camera index for the external side view.")
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--realsense", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--front-realsense-serial",
        "--realsense-serial",
        dest="front_realsense_serial",
        type=str,
        default="315122271404",
        help="PIKA Gripper RealSense serial for cam_front, i.e. the RGB camera below the fisheye.",
    )

    parser.add_argument("--robot-ip", type=str, default="192.168.1.18")
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--enable-gripper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gripper-port", type=str, default="/dev/ttyUSB82")
    parser.add_argument(
        "--pika-project-root",
        type=str,
        default=DEFAULT_PIKA_PROJECT_ROOT,
        help="Root of PIKA_RM65B_data_acquisition; used for pika imports and tactile config files.",
    )
    parser.add_argument("--robot-speed", type=int, default=30)
    parser.add_argument("--blend-radius", type=int, default=0)
    parser.add_argument("--trajectory-connect", type=int, choices=[0, 1], default=0)
    parser.add_argument("--reset-to-home", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--home-joints", type=str, default="0,-30,0,-60,0,-90,-130")
    parser.add_argument("--home-gripper", type=float, default=90.0)
    parser.add_argument("--mock-hardware", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--visualize-window-name", type=str, default="SkinWorld RealMan Tactile Deploy")
    parser.add_argument("--visualize-panel-width", type=int, default=520)
    parser.add_argument("--visualize-panel-height", type=int, default=390)
    parser.add_argument("--visualize-every-steps", type=int, default=1)
    parser.add_argument("--visualize-live-cameras", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualize-tactile-threshold", type=float, default=0.002)
    parser.add_argument("--visualize-tactile-pixel-size", type=int, default=8)

    parser.add_argument("--enable-tactile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tactile-port", type=int, default=0)
    parser.add_argument("--tactile-splitted-file", type=str, default="config_mapping_gripper.json")
    parser.add_argument("--tactile-calibrate-file", type=str, default="calibration_gripper.json")
    parser.add_argument("--tactile-sensor-class", type=str, default="Usb64")
    parser.add_argument("--tactile-sensor-shape", type=str, default="64,64")
    parser.add_argument("--tactile-target-keys", type=str, default="0,1")
    parser.add_argument("--tactile-timeout", type=float, default=0.03)
    parser.add_argument("--tactile-vmin", type=float, default=0.0)
    parser.add_argument("--tactile-vmax", type=float, default=0.3)
    parser.add_argument("--tactile-warmup-sec", type=float, default=0.5)
    parser.add_argument("--tactile-zero-calibration", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--action-mode", choices=["print", "joint", "ee_pose"], default="print")
    parser.add_argument(
        "--action-layout",
        choices=["joint_gripper", "eef_gripper", "eef_joint_gripper"],
        default="joint_gripper",
    )
    parser.add_argument(
        "--feedback-mode",
        choices=["command", "measured_joint", "measured_full"],
        default="command",
        help=(
            "State stream for KV-cache update. 'command' reuses issued model actions; "
            "'measured_joint' overwrites joint/gripper slots with RealMan/Pika feedback; "
            "'measured_full' also overwrites EEF from RealMan current TCP pose."
        ),
    )
    parser.add_argument("--joint-motion-method", choices=["rm_movej", "rm_movej_canfd"], default="rm_movej")
    parser.add_argument("--max-joint-step-deg", type=float, default=1.0)
    parser.add_argument("--joint-step-limit-mode", choices=["clamp", "transition"], default="clamp")
    parser.add_argument("--first-joint-delta-limit-deg", type=float, default=8.0)
    parser.add_argument("--first-joint-delta-mode", choices=["abort", "transition"], default="abort")
    parser.add_argument("--joint-transition-dt", type=float, default=0.05)
    parser.add_argument("--blocking-motion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--emergency-stop-on-exit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-mode", choices=["immediate", "slow"], default="immediate")
    parser.add_argument("--canfd-follow", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--canfd-trajectory-mode", type=int, default=0)
    parser.add_argument("--canfd-radio", type=int, default=0)
    parser.add_argument("--pose-method", choices=["rm_movel", "rm_movej_p", "rm_movep_canfd"], default=None)
    parser.add_argument("--pose-orientation-format", choices=["euler_xyz", "rotvec", "quat_xyzw"], default="euler_xyz")
    parser.add_argument("--max-position-step", type=float, default=0.015)
    parser.add_argument("--gripper-offset", type=float, default=0.0)
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=100.0)
    parser.add_argument("--max-gripper-step", type=float, default=5.0)
    return parser


def main():
    args = build_parser().parse_args()
    if args.visualize_every_steps <= 0:
        raise ValueError("--visualize-every-steps must be positive")
    if args.visualize_tactile_pixel_size <= 0:
        raise ValueError("--visualize-tactile-pixel-size must be positive")
    if args.max_joint_step_deg < 0:
        raise ValueError("--max-joint-step-deg must be non-negative")
    if args.first_joint_delta_limit_deg < 0:
        raise ValueError("--first-joint-delta-limit-deg must be non-negative")
    if args.joint_transition_dt < 0:
        raise ValueError("--joint-transition-dt must be non-negative")
    if args.first_joint_delta_mode == "transition" and args.max_joint_step_deg <= 0:
        raise ValueError("--first-joint-delta-mode transition requires --max-joint-step-deg > 0")
    if args.joint_step_limit_mode == "transition" and args.max_joint_step_deg <= 0:
        raise ValueError("--joint-step-limit-mode transition requires --max-joint-step-deg > 0")
    if args.feedback_mode != "command" and args.action_mode == "print":
        print(
            f"[WARN] --feedback-mode {args.feedback_mode} with --action-mode print reads current hardware state "
            "but does not execute model actions."
        )
    if args.feedback_mode != "command" and args.mock_hardware:
        print("[WARN] mock hardware measured feedback uses zero EEF/joints and --home-gripper")
    run(args)


if __name__ == "__main__":
    main()
