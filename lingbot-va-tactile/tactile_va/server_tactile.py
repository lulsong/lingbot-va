import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SIDECAR_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SIDECAR_ROOT.parent
WAN_VA_ROOT = REPO_ROOT / "wan_va"
for path in (str(SCRIPT_DIR), str(WAN_VA_ROOT), str(REPO_ROOT), str(SIDECAR_ROOT)):
    while path in sys.path:
        sys.path.remove(path)
sys.path[:0] = [str(WAN_VA_ROOT), str(REPO_ROOT), str(SIDECAR_ROOT)]

from tactile_va.configs import TACTILE_CONFIGS
from tactile_va.configs.stats import apply_stats_json
from tactile_va.modules import load_tactile_transformer
from wan_va.distributed.fsdp import shard_model
from wan_va.distributed.util import _configure_model, init_distributed
from wan_va.utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)
from wan_va.wan_va_server import VA_Server


class TactileVA_Server(VA_Server):
    """LingBot-VA server with tactile history cached as a third token stream."""

    def __init__(self, job_config):
        super().__init__(job_config)
        self.tactile_scheduler = FlowMatchScheduler(
            shift=self.job_config.tactile_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.tactile_scheduler.set_timesteps(1000, training=True)
        self.last_predicted_tactile = None

        logger.info("Replacing base transformer with tactile transformer...")
        del self.transformer
        torch.cuda.empty_cache()
        transformer_path = getattr(job_config, "transformer_path", None)
        if transformer_path is None:
            transformer_path = os.path.join(
                job_config.wan22_pretrained_model_name_or_path,
                "transformer",
            )
        elif os.path.isdir(os.path.join(transformer_path, "transformer")):
            transformer_path = os.path.join(transformer_path, "transformer")
        self.transformer = load_tactile_transformer(
            transformer_path,
            torch_dtype=self.dtype,
            torch_device=self.device,
            attn_mode="torch",
            tactile_dim=job_config.tactile_dim,
            tactile_shape=getattr(job_config, "tactile_shape", (2, 32, 58)),
            tactile_token_grid=getattr(job_config, "tactile_token_grid", (4, 6)),
        )
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_model,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=True,
        )

    def preprocess_tactile(self, tactile):
        tactile_model_input = torch.from_numpy(np.asarray(tactile, dtype=np.float32))
        tactile_dim = self.job_config.tactile_dim
        tactile_shape = tuple(getattr(self.job_config, "tactile_shape", (2, 32, 58)))
        if tactile_model_input.ndim == 2:
            tactile_model_input = tactile_model_input.reshape(-1)
            if tactile_model_input.numel() != tactile_dim:
                raise ValueError(f"Flattened tactile input must equal tactile_dim={tactile_dim}")
            tactile_model_input = tactile_model_input.reshape(1, *tactile_shape)
        elif tactile_model_input.ndim == 3:
            if tuple(tactile_model_input.shape) == tactile_shape:
                tactile_model_input = tactile_model_input.unsqueeze(0)
            else:
                flattened = tactile_model_input.reshape(tactile_model_input.shape[0], -1)
                if flattened.shape[1] != tactile_dim:
                    raise ValueError(f"3D tactile input must be [C,H,W] or [F,C*H*W], tactile_dim={tactile_dim}")
                tactile_model_input = flattened.reshape(flattened.shape[0], *tactile_shape)
        elif tactile_model_input.ndim == 4:
            flattened = tactile_model_input.reshape(tactile_model_input.shape[0], -1)
            if flattened.shape[1] != tactile_dim:
                raise ValueError(
                    "4D tactile input is interpreted as [F, finger, row, col]; "
                    f"flattened sensor dimension must equal tactile_dim={tactile_dim}"
                )
            tactile_model_input = flattened.reshape(flattened.shape[0], *tactile_shape)
        if tactile_model_input.ndim != 4:
            raise ValueError("Expected tactile observation shape [C,H,W], [F,C,H,W], or flattened equivalent")

        tactile_q01 = torch.tensor(
            self.job_config.tactile_norm_stat["q01"],
            dtype=torch.float32,
        ).reshape(1, *tactile_shape)
        tactile_q99 = torch.tensor(
            self.job_config.tactile_norm_stat["q99"],
            dtype=torch.float32,
        ).reshape(1, *tactile_shape)
        tactile_model_input = (tactile_model_input - tactile_q01) / (
            tactile_q99 - tactile_q01 + 1e-6
        ) * 2.0 - 1.0
        tactile_model_input = tactile_model_input.clamp(-1.5, 1.5)
        return tactile_model_input.permute(1, 0, 2, 3).unsqueeze(0)

    def _prepare_tactile_input(self, tactile_model_input, tactile_t=0, frame_st_id=0):
        tactile_f_shift = frame_st_id * self.job_config.tactile_per_frame
        return {
            "noisy_latents": tactile_model_input,
            "timesteps": torch.ones(
                [tactile_model_input.shape[2]],
                dtype=torch.float32,
                device=self.device,
            )
            * tactile_t,
            "grid_id": get_mesh_id(
                tactile_model_input.shape[-3],
                self.job_config.tactile_token_grid[0],
                self.job_config.tactile_token_grid[1],
                2,
                1,
                tactile_f_shift,
                action=False,
            ).to(self.device),
            "text_emb": self.prompt_embeds.to(self.dtype).clone(),
        }

    def _reset(self, prompt=None):
        super()._reset(prompt=prompt)
        self.use_cfg = self.use_cfg or (getattr(self.job_config, "tactile_guidance_scale", 1) > 1)
        if self.use_cfg and prompt is not None and self.negative_prompt_embeds is None:
            self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=True,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=512,
                device=self.device,
                dtype=self.dtype,
            )

        patch_size = self.job_config.patch_size
        latent_token_per_chunk = (
            self.job_config.frame_chunk_size * self.latent_height * self.latent_width
        ) // (patch_size[0] * patch_size[1] * patch_size[2])
        action_token_per_chunk = self.job_config.frame_chunk_size * self.action_per_frame
        tactile_token_per_chunk = (
            self.job_config.frame_chunk_size
            * self.job_config.tactile_per_frame
            * self.job_config.tactile_token_grid[0]
            * self.job_config.tactile_token_grid[1]
        )
        self.transformer.clear_cache(self.cache_name)
        self.transformer.create_empty_cache(
            self.cache_name,
            self.job_config.attn_window,
            latent_token_per_chunk,
            action_token_per_chunk,
            dtype=self.dtype,
            device=self.device,
            batch_size=2 if self.use_cfg else 1,
            tactile_token_per_chunk=tactile_token_per_chunk,
        )

    def _sample_future_tactile(self, obs, frame_chunk_size, frame_st_id):
        tactile_channels, tactile_height, tactile_width = self.job_config.tactile_shape
        tactile_num_frames = frame_chunk_size * self.job_config.tactile_per_frame
        tactile = torch.randn(
            1,
            tactile_channels,
            tactile_num_frames,
            tactile_height,
            tactile_width,
            device=self.device,
            dtype=self.dtype,
        )

        tactile_cond = None
        if frame_st_id == 0 and "tactile" in obs:
            tactile_cond = self.preprocess_tactile(obs["tactile"]).to(tactile)
            tactile_cond = tactile_cond[:, :, -1:]

        self.tactile_scheduler.set_timesteps(self.job_config.tactile_num_inference_steps)
        tactile_timesteps = F.pad(
            self.tactile_scheduler.timesteps,
            (0, 1),
            mode="constant",
            value=0,
        )

        for i, t in enumerate(tqdm(tactile_timesteps)):
            last_step = i == len(tactile_timesteps) - 1
            input_dict = self._prepare_tactile_input(
                tactile,
                tactile_t=t,
                frame_st_id=frame_st_id,
            )
            if tactile_cond is not None:
                input_dict["noisy_latents"][:, :, 0:1] = tactile_cond
                input_dict["timesteps"][0:1] *= 0

            tactile_noise_pred = self.transformer(
                self._repeat_input_for_cfg(input_dict),
                update_cache=1 if last_step else 0,
                cache_name=self.cache_name,
                tactile_mode=True,
            )

            if not last_step:
                if getattr(self.job_config, "tactile_guidance_scale", 1) > 1:
                    tactile_noise_pred = (
                        tactile_noise_pred[1:]
                        + self.job_config.tactile_guidance_scale
                        * (tactile_noise_pred[:1] - tactile_noise_pred[1:])
                    )
                else:
                    tactile_noise_pred = tactile_noise_pred[:1]
                tactile = self.tactile_scheduler.step(
                    tactile_noise_pred,
                    t,
                    tactile,
                    return_dict=False,
                )
                if tactile_cond is not None:
                    tactile[:, :, 0:1] = tactile_cond

        save_async(tactile, os.path.join(self.exp_save_root, f"tactile_{frame_st_id}.pt"))
        return tactile

    def _infer(self, obs, frame_st_id=0):
        frame_chunk_size = self.job_config.frame_chunk_size
        if frame_st_id == 0:
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent

        latents = torch.randn(
            1,
            48,
            frame_chunk_size,
            self.latent_height,
            self.latent_width,
            device=self.device,
            dtype=self.dtype,
        )
        actions = torch.randn(
            1,
            self.job_config.action_dim,
            frame_chunk_size,
            self.action_per_frame,
            1,
            device=self.device,
            dtype=self.dtype,
        )

        video_inference_step = self.job_config.num_inference_steps
        action_inference_step = self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = F.pad(self.scheduler.timesteps, (0, 1), mode="constant", value=0)
        action_timesteps = F.pad(
            self.action_scheduler.timesteps,
            (0, 1),
            mode="constant",
            value=0,
        )
        if video_step != -1:
            timesteps = timesteps[:video_step]

        predicted_tactile = None
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps)):
                last_step = i == len(timesteps) - 1
                latent_cond = (
                    init_latent[:, :, 0:1].to(self.dtype)
                    if frame_st_id == 0
                    else None
                )
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    t,
                    t,
                    latent_cond,
                    None,
                    frame_st_id=frame_st_id,
                )

                video_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False,
                )

                if not last_step or video_step != -1:
                    video_noise_pred = data_seq_to_patch(
                        self.job_config.patch_size,
                        video_noise_pred,
                        frame_chunk_size,
                        self.latent_height,
                        self.latent_width,
                        batch_size=2 if self.use_cfg else 1,
                    )
                    if self.job_config.guidance_scale > 1:
                        video_noise_pred = (
                            video_noise_pred[1:]
                            + self.job_config.guidance_scale
                            * (video_noise_pred[:1] - video_noise_pred[1:])
                        )
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    latents = self.scheduler.step(
                        video_noise_pred,
                        t,
                        latents,
                        return_dict=False,
                    )

                if frame_st_id == 0:
                    latents[:, :, 0:1] = latent_cond

            if getattr(self.job_config, "predict_tactile_before_action", True):
                predicted_tactile = self._sample_future_tactile(
                    obs,
                    frame_chunk_size,
                    frame_st_id,
                )

            for i, t in enumerate(tqdm(action_timesteps)):
                last_step = i == len(action_timesteps) - 1
                action_cond = (
                    torch.zeros(
                        [
                            1,
                            self.job_config.action_dim,
                            1,
                            self.action_per_frame,
                            1,
                        ],
                        device=self.device,
                        dtype=self.dtype,
                    )
                    if frame_st_id == 0
                    else None
                )

                input_dict = self._prepare_latent_input(
                    None,
                    actions,
                    t,
                    t,
                    None,
                    action_cond,
                    frame_st_id=frame_st_id,
                )
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=True,
                )

                if not last_step:
                    action_noise_pred = rearrange(
                        action_noise_pred,
                        "b (f n) c -> b c f n 1",
                        f=frame_chunk_size,
                    )
                    if self.job_config.action_guidance_scale > 1:
                        action_noise_pred = (
                            action_noise_pred[1:]
                            + self.job_config.action_guidance_scale
                            * (action_noise_pred[:1] - action_noise_pred[1:])
                        )
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    actions = self.action_scheduler.step(
                        action_noise_pred,
                        t,
                        actions,
                        return_dict=False,
                    )

                if frame_st_id == 0:
                    actions[:, :, 0:1] = action_cond

        actions[:, ~self.action_mask] *= 0

        save_async(latents, os.path.join(self.exp_save_root, f"latents_{frame_st_id}.pt"))
        save_async(actions, os.path.join(self.exp_save_root, f"actions_{frame_st_id}.pt"))

        self.last_predicted_tactile = predicted_tactile
        actions = self.postprocess_action(actions)
        torch.cuda.empty_cache()
        return actions, latents

    def _compute_kv_cache(self, obs):
        self.transformer.clear_pred_cache(self.cache_name)
        save_async(obs["obs"], os.path.join(self.exp_save_root, f"obs_data_{self.frame_st_id}.pt"))

        latent_model_input = self._encode_obs(obs)
        if self.frame_st_id == 0:
            latent_model_input = (
                torch.cat([self.init_latent, latent_model_input], dim=2)
                if latent_model_input is not None
                else self.init_latent
            )

        action_model_input = self.preprocess_action(obs["state"]).to(latent_model_input)
        tactile_model_input = self.preprocess_tactile(obs["tactile"]).to(latent_model_input)

        target_tactile_frames = latent_model_input.shape[2] * self.job_config.tactile_per_frame
        if tactile_model_input.shape[2] != target_tactile_frames:
            tactile_model_input = F.interpolate(
                tactile_model_input,
                size=(
                    target_tactile_frames,
                    tactile_model_input.shape[3],
                    tactile_model_input.shape[4],
                ),
                mode="trilinear",
                align_corners=False,
            )

        logger.info(
            "get KV cache obs: "
            f"video={latent_model_input.shape} action={action_model_input.shape} "
            f"tactile={tactile_model_input.shape}"
        )

        input_dict = self._prepare_latent_input(
            latent_model_input,
            action_model_input,
            frame_st_id=self.frame_st_id,
        )
        tactile_dict = self._prepare_tactile_input(
            tactile_model_input,
            frame_st_id=self.frame_st_id,
        )

        with torch.no_grad():
            self.transformer(
                self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                update_cache=2,
                cache_name=self.cache_name,
                action_mode=False,
            )
            self.transformer(
                self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                update_cache=2,
                cache_name=self.cache_name,
                action_mode=True,
            )
            self.transformer(
                self._repeat_input_for_cfg(tactile_dict),
                update_cache=2,
                cache_name=self.cache_name,
                tactile_mode=True,
            )

        torch.cuda.empty_cache()
        self.frame_st_id += latent_model_input.shape[2]

    @torch.no_grad()
    def infer(self, obs):
        reset = obs.get("reset", False)
        prompt = obs.get("prompt", None)
        compute_kv_cache = obs.get("compute_kv_cache", False)

        if reset:
            logger.info("******************* Reset tactile server ******************")
            self._reset(prompt=prompt)
            return dict()
        if compute_kv_cache:
            logger.info("################# Compute tactile KV Cache #################")
            self._compute_kv_cache(obs)
            return dict()

        logger.info("################# Infer One Tactile Chunk #################")
        action, _ = self._infer(obs, frame_st_id=self.frame_st_id)
        response = dict(action=action)
        if getattr(self.job_config, "return_predicted_tactile", False):
            response["predicted_tactile"] = (
                None
                if self.last_predicted_tactile is None
                else self.last_predicted_tactile.detach().cpu().numpy()
            )
        return response


def run(args):
    config = apply_stats_json(TACTILE_CONFIGS[args.config_name])
    if args.model_path is not None:
        config.wan22_pretrained_model_name_or_path = args.model_path
    if args.transformer_path is not None:
        config.transformer_path = args.transformer_path
    if args.stats_json_path is not None:
        config.stats_json_path = args.stats_json_path
        config = apply_stats_json(config)
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    model = TactileVA_Server(config)
    if config.infer_mode != "server":
        raise ValueError("TactileVA_Server currently supports infer_mode='server'")
    run_async_server_mode(model, local_rank, config.host, port)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="robotwin_tactile")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--transformer-path", type=str, default=None)
    parser.add_argument("--stats-json-path", type=str, default=None)
    parser.add_argument("--save_root", type=str, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
