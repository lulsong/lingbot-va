#!/usr/bin/env python3
"""Deploy LingBot-VA-Tactile on the local RealMan + Pika hardware.

This is a websocket client. Start ``tactile_va.server_tactile`` separately, then
run this script on the machine connected to the robot/cameras/gripper.
"""

from __future__ import annotations

import argparse
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
ACTION_CHANNELS = 8
DEFAULT_PIKA_PROJECT_ROOT = "/home/tujian/Projects/pika_sdk/PIKA_RM65B_data_acquisition"


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


def tactile_from_force_dict(
    force_dict: Any,
    target_keys: list[str],
    shape: tuple[int, int, int] = TACTILE_SHAPE,
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
        if arr.shape != shape[1:]:
            if arr.size != int(np.prod(shape[1:])):
                raise ValueError(f"tactile key {key!r} has shape {arr.shape}, expected {shape[1:]}")
            arr = arr.reshape(shape[1:])
        sheets.append(arr)
    return np.stack(sheets, axis=0).astype(np.float32, copy=False)


class PikaTactileReader:
    """Runtime tactile reader matching PIKA_RM65B_data_acquisition/main_teleop.py."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.provider = None
        self.target_keys = parse_csv_strings(args.tactile_target_keys)
        self.last_valid = np.zeros(TACTILE_SHAPE, dtype=np.float32)
        self.no_data_count = 0

        if args.mock_hardware or not args.enable_tactile:
            return

        add_pika_sdk_paths(args.pika_project_root)

        try:
            from Usb_API_stable_Tujian.TactileDataProvider import TactileDataProvider
        except ImportError as exc:
            raise ImportError(
                "Cannot import Usb_API_stable_Tujian.TactileDataProvider. "
                "Set --pika-project-root to PIKA_RM65B_data_acquisition or install the tactile SDK."
            ) from exc

        sensor_shape = parse_csv_ints(args.tactile_sensor_shape, expected=2)
        splitted_file = resolve_project_path(args.tactile_splitted_file, args.pika_project_root)
        calibrate_file = resolve_project_path(args.tactile_calibrate_file, args.pika_project_root)

        try:
            self.provider = TactileDataProvider(
                port=args.tactile_port,
                sensor_class=args.tactile_sensor_class,
                sensor_shape=sensor_shape,
                splitted_file_path=str(splitted_file),
                calibrate_file_path=str(calibrate_file),
                filtering_coloring=False,
                y_lim=[args.tactile_vmin, args.tactile_vmax],
                tfd=True,
                timeout=args.tactile_timeout,
            )
            self.provider.start()
            time.sleep(args.tactile_warmup_sec)
            if args.tactile_zero_calibration:
                self.provider.zero_calibration()
        except Exception:
            self.close()
            raise

        print(
            "[INFO] tactile provider started: "
            f"port={args.tactile_port}, keys={self.target_keys}, "
            f"splitted={splitted_file}, calibrate={calibrate_file}"
        )

    def read(self) -> np.ndarray:
        if self.provider is None:
            return self.last_valid.copy()

        try:
            latest = self.provider.get_latest_data()
            if not latest:
                raise ValueError("get_latest_data returned empty value")

            force_dict = latest[0] if isinstance(latest, (list, tuple)) else latest
            if force_dict is None:
                raise ValueError("normal force dict is None")

            tactile = tactile_from_force_dict(force_dict, self.target_keys)
            self.last_valid = tactile
            self.no_data_count = 0
            return tactile.copy()
        except Exception as exc:
            self.no_data_count += 1
            if self.no_data_count == 1 or self.no_data_count % self.args.tactile_warn_interval == 0:
                print(f"[WARN] tactile read failed for {self.no_data_count} frame(s): {exc}")
            if self.args.tactile_missing_mode == "zeros":
                return np.zeros(TACTILE_SHAPE, dtype=np.float32)
            return self.last_valid.copy()

    def close(self):
        if self.provider is not None:
            self.provider.stop()
            self.provider = None


class CameraSystem:
    def __init__(self, args: argparse.Namespace):
        self.size = (args.camera_width, args.camera_height)
        self.front = None if args.front_camera_id < 0 else cv2.VideoCapture(args.front_camera_id)
        self.side = None if args.side_camera_id < 0 else cv2.VideoCapture(args.side_camera_id)
        self.missing_side_mode = args.missing_side_mode
        self.missing_fisheye_mode = args.missing_fisheye_mode
        self.pipeline = None

        if args.realsense:
            try:
                import pyrealsense2 as rs

                self.rs = rs
                self.pipeline = rs.pipeline()
                config = rs.config()
                if args.realsense_serial:
                    config.enable_device(args.realsense_serial)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                self.pipeline.start(config)
            except Exception as exc:
                print(f"[WARN] RealSense disabled: {exc}")
                self.pipeline = None

        if self.front is not None and not self.front.isOpened():
            print(f"[WARN] front camera id={args.front_camera_id} is not opened")
        if self.side is not None and not self.side.isOpened():
            print(f"[WARN] side camera id={args.side_camera_id} is not opened")

    def _read_cv(self, cap):
        if cap is None:
            return False, None
        return cap.read()

    def _read_realsense(self):
        if self.pipeline is None:
            return False, None
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return False, None
        return True, np.asanyarray(color_frame.get_data())

    @staticmethod
    def _fallback(mode: str, front: np.ndarray, fisheye: np.ndarray | None = None) -> np.ndarray:
        if mode == "front":
            return front.copy()
        if mode == "fisheye" and fisheye is not None:
            return fisheye.copy()
        return np.zeros_like(front)

    def get_frames(self) -> dict[str, np.ndarray]:
        ret_front, raw_front = self._read_cv(self.front)
        front = process_rgb(ret_front, raw_front, self.size)

        ret_fisheye, raw_fisheye = self._read_realsense()
        fisheye = process_rgb(ret_fisheye, raw_fisheye, self.size)
        if not ret_fisheye:
            fisheye = self._fallback(self.missing_fisheye_mode, front)

        ret_side, raw_side = self._read_cv(self.side)
        side = process_rgb(ret_side, raw_side, self.size)
        if not ret_side:
            side = self._fallback(self.missing_side_mode, front, fisheye)

        return {
            "observation.images.cam_front": front,
            "observation.images.cam_side": side,
            "observation.images.cam_fisheye": fisheye,
        }

    def close(self):
        if self.front is not None:
            self.front.release()
        if self.side is not None:
            self.side.release()
        if self.pipeline is not None:
            self.pipeline.stop()


class RealmanPikaHardware:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.robot = None
        self.gripper = None
        self.tactile_reader = PikaTactileReader(args)
        self.last_pose_position = None
        self.last_gripper = None

        if args.mock_hardware:
            print("[WARN] mock hardware mode: robot commands are not sent")
            return

        try:
            add_pika_sdk_paths(args.pika_project_root)

            from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
            try:
                from pika.gripper import Gripper
            except ImportError:
                from pika_sdk.pika.gripper import Gripper

            self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
            handle = self.robot.rm_create_robot_arm(args.robot_ip, args.robot_port)
            print("RealMan robot id:", handle.id)

            self.gripper = Gripper(args.gripper_port)
            if not self.gripper.connect():
                raise RuntimeError(f"Failed to connect Pika gripper at {args.gripper_port}")
            if not self.gripper.enable():
                raise RuntimeError("Failed to enable Pika gripper")
            time.sleep(1.0)
        except Exception:
            self.close()
            raise

    def home(self):
        if self.args.mock_hardware:
            return
        joints = parse_csv_floats(self.args.home_joints, expected=7)
        self.move_joints(joints, self.args.home_gripper)

    def get_joint_state(self) -> np.ndarray:
        if self.args.mock_hardware:
            return np.zeros(7, dtype=np.float32)
        ret, degree = self.robot.rm_get_joint_degree()
        if ret != 0:
            print(f"[WARN] rm_get_joint_degree ret={ret}")
        return np.asarray(degree, dtype=np.float32)

    def get_gripper_state(self) -> float:
        if self.args.mock_hardware or self.gripper is None:
            return float(self.args.home_gripper)
        return float(self.gripper.get_gripper_distance())

    def move_joints(self, joints: np.ndarray, gripper: float):
        if self.args.mock_hardware:
            print("[MOCK] move_joints", np.round(joints, 3), round(float(gripper), 3))
            return
        self.robot.rm_movej(np.asarray(joints, dtype=np.float32).tolist(), self.args.robot_speed, 0, 0, 1)
        self.gripper.set_gripper_distance(float(gripper))

    def read_tactile(self) -> np.ndarray:
        return self.tactile_reader.read()

    def _pose_command_from_action(self, action8: np.ndarray) -> tuple[np.ndarray, float]:
        pos = np.asarray(action8[:3], dtype=np.float64)
        quat = normalize_quat_xyzw(action8[3:7])
        gripper = float(action8[7] + self.args.gripper_offset)
        gripper = float(np.clip(gripper, self.args.gripper_min, self.args.gripper_max))

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

        method_names = [self.args.pose_method] if self.args.pose_method else []
        method_names += ["rm_movel", "rm_movej_p", "rm_movep"]
        last_error = None
        for name in method_names:
            method = getattr(self.robot, name, None)
            if method is None:
                continue
            for call_args in (
                (pose.tolist(), self.args.robot_speed, 0, 0, 1),
                (pose.tolist(), self.args.robot_speed, 0, 1),
                (pose.tolist(), self.args.robot_speed),
            ):
                try:
                    ret = method(*call_args)
                    if ret not in (None, 0):
                        print(f"[WARN] {name} returned {ret}")
                    self.gripper.set_gripper_distance(float(gripper))
                    return
                except TypeError as exc:
                    last_error = exc
                    continue
        raise RuntimeError(f"No usable RealMan Cartesian move method found; last error={last_error}")

    def execute_action(self, action8: np.ndarray):
        action8 = np.asarray(action8, dtype=np.float32).reshape(-1)
        if action8.size != ACTION_CHANNELS:
            raise ValueError(f"Expected 8D action, got {action8.shape}")

        if self.args.action_mode == "print":
            print("[ACTION]", np.round(action8, 4))
            return
        if self.args.action_mode == "joint":
            joints = action8[:7]
            gripper = float(np.clip(action8[7] + self.args.gripper_offset, self.args.gripper_min, self.args.gripper_max))
            self.move_joints(joints, gripper)
            return
        if self.args.action_mode == "ee_pose":
            pose, gripper = self._pose_command_from_action(action8)
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
            self.cameras = CameraSystem(args)
            self.hw = RealmanPikaHardware(args)
            if args.reset_to_home:
                self.hw.home()
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
    if action.shape[0] != ACTION_CHANNELS:
        raise ValueError(f"Expected {ACTION_CHANNELS} action channels, got {action.shape}")
    return action


def maybe_visualize(args, obs: dict[str, np.ndarray], tactile: np.ndarray):
    if not args.visualize:
        return None
    import matplotlib.pyplot as plt

    plt.ion()
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    titles = ["front", "side", "fisheye", "tactile mean"]
    ims = []
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.axis("off")
    ims.append(axes[0].imshow(obs["observation.images.cam_front"]))
    ims.append(axes[1].imshow(obs["observation.images.cam_side"]))
    ims.append(axes[2].imshow(obs["observation.images.cam_fisheye"]))
    ims.append(axes[3].imshow(tactile.mean(axis=0), cmap="viridis", aspect="auto"))
    plt.tight_layout()
    plt.show(block=False)
    return fig, ims


def update_visualization(fig_and_ims, obs: dict[str, np.ndarray], tactile: np.ndarray):
    if fig_and_ims is None:
        return
    fig, ims = fig_and_ims
    ims[0].set_data(obs["observation.images.cam_front"])
    ims[1].set_data(obs["observation.images.cam_side"])
    ims[2].set_data(obs["observation.images.cam_fisheye"])
    ims[3].set_data(tactile.mean(axis=0))
    fig.canvas.draw()
    fig.canvas.flush_events()


def run(args: argparse.Namespace):
    from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy

    policy = WebsocketClientPolicy(host=args.server_host, port=args.server_port)
    env = RealmanTactileEnv(args)
    prompt = args.prompt or input("Task instruction: ").strip()

    print("[INFO] reset tactile LingBot-VA server")
    policy.infer({"reset": True, "prompt": prompt})

    first = True
    latest_obs = env.get_image_obs()
    latest_tactile = env.get_tactile()
    fig_and_ims = maybe_visualize(args, latest_obs, latest_tactile)

    try:
        for chunk_idx in range(args.max_chunks):
            print(f"[INFO] infer chunk {chunk_idx}")
            request = {
                "obs": latest_obs,
                "prompt": prompt,
                "tactile": latest_tactile,
            }
            result = policy.infer(request)
            action = flatten_action_chunk(result["action"])
            frame_count = action.shape[1]
            steps_per_frame = action.shape[2]
            start_frame = 1 if first else 0
            key_frames: list[dict[str, np.ndarray]] = []
            tactile_history: list[np.ndarray] = []

            if first:
                tactile_history.extend([latest_tactile.copy()] * steps_per_frame)

            for frame_id in range(start_frame, frame_count):
                for step_id in range(steps_per_frame):
                    action_step = action[:, frame_id, step_id]
                    env.hw.execute_action(action_step)
                    time.sleep(args.control_dt)

                    latest_tactile = env.get_tactile()
                    tactile_history.append(latest_tactile.copy())

                    if (step_id + 1) % steps_per_frame == 0:
                        latest_obs = env.get_image_obs()
                        key_frames.append(latest_obs)
                        update_visualization(fig_and_ims, latest_obs, latest_tactile)

            if not key_frames:
                latest_obs = env.get_image_obs()
                key_frames.append(latest_obs)

            tactile_cache = np.stack(tactile_history, axis=0).astype(np.float32)
            print(
                f"[INFO] update KV cache: key_frames={len(key_frames)} "
                f"action={action.shape} tactile={tactile_cache.shape}"
            )
            policy.infer(
                {
                    "compute_kv_cache": True,
                    "obs": key_frames,
                    "state": action,
                    "tactile": tactile_cache,
                }
            )
            first = False

    except KeyboardInterrupt:
        print("[INFO] interrupted by user")
    finally:
        env.close()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy LingBot-VA-Tactile on RealMan + Pika hardware")
    parser.add_argument("--server-host", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=29536)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-chunks", type=int, default=20)
    parser.add_argument("--control-dt", type=float, default=0.02)

    parser.add_argument("--front-camera-id", type=int, default=12)
    parser.add_argument("--side-camera-id", type=int, default=-1)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--missing-side-mode", choices=["front", "fisheye", "zeros"], default="front")
    parser.add_argument("--missing-fisheye-mode", choices=["front", "zeros"], default="front")
    parser.add_argument("--realsense", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--realsense-serial", type=str, default="315122271124")

    parser.add_argument("--robot-ip", type=str, default="192.168.1.18")
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--gripper-port", type=str, default="/dev/ttyUSB82")
    parser.add_argument(
        "--pika-project-root",
        type=str,
        default=DEFAULT_PIKA_PROJECT_ROOT,
        help="Root of PIKA_RM65B_data_acquisition; used for pika imports and tactile config files.",
    )
    parser.add_argument("--robot-speed", type=int, default=30)
    parser.add_argument("--reset-to-home", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--home-joints", type=str, default="0,-30,0,-70,0,-80,-130")
    parser.add_argument("--home-gripper", type=float, default=90.0)
    parser.add_argument("--mock-hardware", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--enable-tactile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tactile-port", type=int, default=0)
    parser.add_argument("--tactile-splitted-file", type=str, default="config_mapping_gripper.json")
    parser.add_argument("--tactile-calibrate-file", type=str, default="calibration_gripper.json")
    parser.add_argument("--tactile-sensor-class", type=str, default="Usb64")
    parser.add_argument("--tactile-sensor-shape", type=str, default="64,64")
    parser.add_argument("--tactile-target-keys", type=str, default="0,1")
    parser.add_argument("--tactile-timeout", type=float, default=0.03)
    parser.add_argument("--tactile-vmin", type=float, default=0.0)
    parser.add_argument("--tactile-vmax", type=float, default=0.33)
    parser.add_argument("--tactile-warmup-sec", type=float, default=0.5)
    parser.add_argument("--tactile-zero-calibration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tactile-missing-mode", choices=["last", "zeros"], default="last")
    parser.add_argument("--tactile-warn-interval", type=int, default=100)

    parser.add_argument("--action-mode", choices=["print", "joint", "ee_pose"], default="print")
    parser.add_argument("--pose-method", type=str, default=None)
    parser.add_argument("--pose-orientation-format", choices=["euler_xyz", "rotvec", "quat_xyzw"], default="euler_xyz")
    parser.add_argument("--max-position-step", type=float, default=0.015)
    parser.add_argument("--gripper-offset", type=float, default=0.0)
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=100.0)
    parser.add_argument("--max-gripper-step", type=float, default=5.0)
    return parser


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
