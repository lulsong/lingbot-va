# Copyright 2024-2025 The Robbyant Team Authors.
# Tactile overlay additions are kept outside the original wan_va package.

import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
from einops import rearrange

from wan_va.dataset.lerobot_latent_dataset import (
    LatentLeRobotDataset,
    get_relative_pose,
    recursive_find_file,
)
from wan_va.utils import logger


def construct_tactile_lerobot(repo_id, config):
    return TactileLatentLeRobotDataset(repo_id=repo_id, config=config)


def construct_tactile_lerobot_multi_processor(config, num_init_worker=None):
    repo_list = recursive_find_file(config.dataset_path, "info.json")
    repo_list = [v.split("/meta/info.json")[0] for v in repo_list]
    if not repo_list:
        raise FileNotFoundError(f"No LeRobot repo found under {config.dataset_path!r}")

    if num_init_worker is None:
        num_init_worker = int(getattr(config, "dataset_init_worker", 8))
    num_init_worker = max(1, min(int(num_init_worker), len(repo_list), os.cpu_count() or 1))
    logger.info(
        f"Found {len(repo_list)} tactile LeRobot repo(s); "
        f"initializing with {num_init_worker} worker(s)."
    )

    construct_func = partial(construct_tactile_lerobot, config=config)
    if num_init_worker == 1:
        return [construct_func(repo_id) for repo_id in repo_list]

    with Pool(num_init_worker) as pool:
        return pool.map(construct_func, repo_list)


class MultiTactileLatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(self, config, num_init_worker=None):
        self._datasets = construct_tactile_lerobot_multi_processor(
            config,
            num_init_worker,
        )
        self.item_id_to_dataset_id, self.acc_dset_num = self._get_item_id_to_dataset_id()
        logger.info(
            f"Tactile dataset ready: {len(self._datasets)} repo(s), "
            f"{len(self)} training segment(s)."
        )

    def __len__(self):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        item_id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[item_id] = dset_id
                item_id += 1
        for dset_id in range(len(self._datasets)):
            acc_dset_num[dset_id] = acc_nums[dset_id]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx):
        assert idx < len(self)
        dset_id = self.item_id_to_dataset_id[idx]
        cur_dset = self._datasets[dset_id]
        local_idx = idx - self.acc_dset_num[dset_id]
        return cur_dset[local_idx]


class TactileLatentLeRobotDataset(LatentLeRobotDataset):
    """LeRobot latent dataset with a real tactile token stream.

    Tactile columns are loaded from the HuggingFace/LeRobot table, aligned to
    the same latent-frame-derived sequence length as actions, normalized by
    quantiles, then returned as ``[C, F, H, W]`` to preserve taxel geometry.
    """

    def __init__(self, repo_id, config=None):
        super().__init__(repo_id=repo_id, config=config)
        self.tactile_keys = list(getattr(config, "tactile_keys", []))
        if not self.tactile_keys:
            raise ValueError("config.tactile_keys must contain at least one tactile column")

        self.tactile_dim = int(config.tactile_dim)
        self.tactile_shape = tuple(getattr(config, "tactile_shape", (2, 32, 58)))
        self.tactile_per_frame = int(
            getattr(config, "tactile_per_frame", config.action_per_frame)
        )
        self.tactile_q01 = np.array(config.tactile_norm_stat["q01"], dtype="float32")[None]
        self.tactile_q99 = np.array(config.tactile_norm_stat["q99"], dtype="float32")[None]
        self.tactile_min_range = float(getattr(config, "tactile_min_range", 1e-4))
        self.tactile_mask_low_range = bool(getattr(config, "tactile_mask_low_range", True))
        if self.tactile_q01.shape[-1] != self.tactile_dim:
            raise ValueError("tactile_norm_stat['q01'] length must equal tactile_dim")
        if self.tactile_q99.shape[-1] != self.tactile_dim:
            raise ValueError("tactile_norm_stat['q99'] length must equal tactile_dim")

        self._hf_torch_view = self.hf_dataset.with_format(
            type="torch",
            columns=["action"] + self.tactile_keys,
            output_all_columns=False,
        )

    def parse_meta(self):
        out = []
        for key, value in self.meta.episodes.items():
            episode_index = value["episode_index"]
            tasks = value.get("tasks", [])
            action_config = value.get("action_config")
            if action_config is None:
                episode_length = value.get("length")
                if episode_length is None and hasattr(self.meta, "episodes_stats"):
                    episode_length = self.meta.episodes_stats[episode_index].get("num_frames")
                if episode_length is None:
                    raise KeyError(
                        "Episode metadata must contain either action_config or length "
                        f"for episode_index={episode_index}"
                    )
                action_text = tasks[0] if tasks else "robot manipulation"
                action_config = [
                    {
                        "start_frame": 0,
                        "end_frame": int(episode_length),
                        "action_text": action_text,
                    }
                ]

            for acfg in action_config:
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                cur_meta.update(acfg)
                if self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["end_frame"],
                    cur_meta["episode_index"],
                ):
                    out.append(cur_meta)
        self.new_metas = out

    def _load_tactile_sequence(self, hf_data_frames):
        tactile_parts = []
        for key in self.tactile_keys:
            if key not in hf_data_frames:
                raise KeyError(f"Tactile key {key!r} not found in LeRobot batch")
            value = hf_data_frames[key]
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            value = np.asarray(value, dtype="float32")
            tactile_parts.append(value)

        if len(tactile_parts) == 1:
            tactile = tactile_parts[0]
        else:
            tactile = np.concatenate(tactile_parts, axis=1)

        tactile_flat = tactile.reshape(tactile.shape[0], -1)
        if tactile_flat.shape[1] != self.tactile_dim:
            raise ValueError(
                f"Expected tactile_dim={self.tactile_dim}, got {tactile_flat.shape[1]}"
            )
        return tactile_flat.reshape(tactile.shape[0], *self.tactile_shape)

    def _resample_tactile(self, tactile, required_num):
        if tactile.shape[0] == required_num:
            return tactile
        if tactile.shape[0] < 2:
            return np.repeat(tactile[:1], required_num, axis=0)

        src_x = np.linspace(0.0, 1.0, tactile.shape[0], dtype="float32")
        dst_x = np.linspace(0.0, 1.0, required_num, dtype="float32")
        flat = tactile.reshape(tactile.shape[0], -1)
        out = np.empty((required_num, flat.shape[1]), dtype="float32")
        for channel in range(flat.shape[1]):
            out[:, channel] = np.interp(dst_x, src_x, flat[:, channel])
        return out.reshape(required_num, *tactile.shape[1:])

    def _tactile_post_process(
        self,
        local_start_frame,
        latent_frame_ids,
        tactile,
    ):
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        tactile = tactile[act_shift:]

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_tactile_num = latent_frame_num * self.tactile_per_frame
        tactile = self._resample_tactile(tactile, required_tactile_num)

        tactile_flat = tactile.reshape(tactile.shape[0], -1)
        tactile_range = self.tactile_q99 - self.tactile_q01
        tactile_valid = tactile_range >= self.tactile_min_range
        if self.tactile_mask_low_range:
            tactile_mask = np.broadcast_to(tactile_valid, tactile_flat.shape).copy()
        else:
            tactile_mask = np.ones_like(tactile_flat, dtype="bool")
        tactile_denom = np.maximum(tactile_range, self.tactile_min_range)
        tactile_flat = (tactile_flat - self.tactile_q01) / tactile_denom * 2.0 - 1.0
        tactile_flat = np.clip(tactile_flat, -1.5, 1.5)
        tactile = tactile_flat.reshape(tactile.shape)
        tactile_mask = tactile_mask.reshape(tactile.shape)

        tactile = rearrange(tactile, "t c h w -> c t h w")
        tactile_mask = rearrange(
            tactile_mask,
            "t c h w -> c t h w",
        )
        tactile *= tactile_mask
        return torch.from_numpy(tactile).float(), torch.from_numpy(tactile_mask).bool()

    def _action_post_process(self, local_start_frame, local_end_frame, latent_frame_ids, action):
        latent_frame_ids = np.asarray(latent_frame_ids)
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        else:
            action = np.asarray(action)

        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = int(latent_frame_ids[1] - latent_frame_ids[0])
        action = action[act_shift:]

        if self.config.env_type == "robotwin_tshape":
            left_action = get_relative_pose(action[:, :7])
            right_action = get_relative_pose(action[:, 8:15])
            if torch.is_tensor(left_action):
                left_action = left_action.detach().cpu().numpy()
            if torch.is_tensor(right_action):
                right_action = right_action.detach().cpu().numpy()
            action = np.concatenate(
                [left_action, action[:, 7:8], right_action, action[:, 15:16]],
                axis=1,
            )

        leading_pad = frame_stride * 4
        action = np.pad(
            action,
            pad_width=((leading_pad, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4
        action = action[:required_action_num]
        valid_action_num = action.shape[0]
        if valid_action_num < required_action_num:
            action = np.pad(
                action,
                pad_width=((0, required_action_num - valid_action_num), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        action_mask = np.zeros_like(action, dtype="bool")
        action_mask[:valid_action_num] = True

        action_padded = np.pad(
            action,
            ((0, 0), (0, 1)),
            mode="constant",
            constant_values=0,
        )
        action_mask_padded = np.pad(
            action_mask,
            ((0, 0), (0, 1)),
            mode="constant",
            constant_values=0,
        )

        action_aligned = action_padded[:, self.config.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]
        action_aligned = (action_aligned - self.q01) / (
            self.q99 - self.q01 + 1e-6
        ) * 2.0 - 1.0
        action_aligned = np.clip(action_aligned, -1.5, 1.5)
        action_aligned = rearrange(
            action_aligned,
            "(f n) c -> c f n 1",
            f=latent_frame_num,
        )
        action_mask_aligned = rearrange(
            action_mask_aligned,
            "(f n) c -> c f n 1",
            f=latent_frame_num,
        )
        action_aligned *= action_mask_aligned
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def __getitem__(self, idx):
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        local_start_frame = start_frame
        local_end_frame = end_frame

        ori_data_dict = self._get_range_latent_data(start_frame, end_frame, episode_index)
        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]

        start_frame = self._get_global_idx(episode_index, start_frame)
        end_frame = self._get_global_idx(episode_index, end_frame)
        hf_data_frames = self._get_range_hf_data(start_frame, end_frame)
        ori_data_dict.update(hf_data_frames)

        out_dict = self._cat_video_latents(ori_data_dict)
        out_dict["actions"], out_dict["actions_mask"] = self._action_post_process(
            local_start_frame,
            local_end_frame,
            latent_frame_ids,
            ori_data_dict["action"],
        )

        tactile = self._load_tactile_sequence(hf_data_frames)
        out_dict["tactile"], out_dict["tactile_mask"] = self._tactile_post_process(
            local_start_frame,
            latent_frame_ids,
            tactile,
        )

        out_dict["latents"] = out_dict["latents"].permute(3, 0, 1, 2)
        return out_dict
