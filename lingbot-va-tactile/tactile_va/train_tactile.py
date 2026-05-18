import argparse
import gc
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SIDECAR_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SIDECAR_ROOT.parent
WAN_VA_ROOT = REPO_ROOT / "wan_va"

# wan_va/train.py uses original top-level imports such as `from configs import VA_CONFIGS`.
# When this file is executed directly, Python otherwise resolves `configs` to
# tactile_va/configs, so keep wan_va itself first on sys.path.
for path in (str(SCRIPT_DIR), str(WAN_VA_ROOT), str(REPO_ROOT), str(SIDECAR_ROOT)):
    while path in sys.path:
        sys.path.remove(path)
sys.path[:0] = [str(WAN_VA_ROOT), str(REPO_ROOT), str(SIDECAR_ROOT)]

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from tactile_va.configs import TACTILE_CONFIGS
from tactile_va.configs.stats import apply_stats_json
from tactile_va.dataset import MultiTactileLatentLeRobotDataset
from tactile_va.modules import load_tactile_transformer
from wan_va.distributed.fsdp import apply_ac, shard_model
from wan_va.distributed.util import _configure_model, dist_max, dist_mean, init_distributed
from wan_va.train import Trainer as BaseTrainer
from wan_va.utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    sample_timestep_id,
    warmup_constant_lambda,
)


class TactileTrainer(BaseTrainer):
    """Fine-tunes LingBot-VA with an added tactile diffusion/token stream."""

    @staticmethod
    def _set_trainable_scope(model, scope):
        scope = scope or "all"
        if scope == "all":
            model.requires_grad_(True)
            return

        video_prefixes = (
            "patch_embedding_mlp",
            "condition_embedder",
            "proj_out",
        )
        action_prefixes = (
            "action_embedder",
            "condition_embedder_action",
            "action_proj_out",
        )
        tactile_prefixes = (
            "tactile_embedder",
            "tactile_proj_out",
            "condition_embedder_tactile",
        )
        trainable_prefixes = {
            "video": video_prefixes,
            "action": action_prefixes,
            "tactile": tactile_prefixes,
            "tactile_io": (
                "tactile_embedder",
                "tactile_proj_out",
            ),
            "tactile_head": ("tactile_proj_out",),
            "tactile_action": tactile_prefixes + action_prefixes,
            "tactile_video": tactile_prefixes + video_prefixes,
            "modality_heads": video_prefixes + action_prefixes + tactile_prefixes,
            "backbone": (
                "blocks",
                "norm_out",
                "scale_shift_table",
            ),
        }
        if scope not in trainable_prefixes:
            raise ValueError(
                f"Unknown trainable_scope={scope!r}; "
                "choose from all, video, action, tactile, tactile_io, "
                "tactile_head, tactile_action, tactile_video, "
                "modality_heads, backbone"
            )

        model.requires_grad_(False)
        prefixes = trainable_prefixes[scope]
        for name, param in model.named_parameters():
            if name.startswith(prefixes):
                param.requires_grad_(True)

    @staticmethod
    def _count_params(model):
        total = 0
        trainable = 0
        for param in model.parameters():
            numel = param.numel()
            total += numel
            if param.requires_grad:
                trainable += numel
        return total, trainable

    def __init__(self, config):
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size
        self.wandb = None

        if config.enable_wandb and config.rank == 0:
            required_wandb_env = ["WANDB_BASE_URL", "WANDB_API_KEY", "WANDB_TEAM_NAME"]
            missing_wandb_env = [key for key in required_wandb_env if not os.getenv(key)]
            if missing_wandb_env:
                logger.warning(
                    "Disabling wandb because required environment variables are missing: "
                    f"{missing_wandb_env}"
                )
                config.enable_wandb = False
            else:
                import wandb

                wandb.login(
                    host=os.environ["WANDB_BASE_URL"],
                    key=os.environ["WANDB_API_KEY"],
                )
                self.wandb = wandb
                self.wandb.init(
                    entity=os.environ["WANDB_TEAM_NAME"],
                    project=os.getenv("WANDB_PROJECT", "va_robotwin_tactile"),
                    config=config,
                    mode="online",
                    name="lingbot_va_tactile",
                )

        if hasattr(config, "resume_from") and config.resume_from:
            transformer_path = os.path.join(config.resume_from, "transformer")
        else:
            transformer_path = os.path.join(
                config.wan22_pretrained_model_name_or_path,
                "transformer",
            )

        logger.info("Loading tactile transformer...")
        self.transformer = load_tactile_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device="cpu",
            attn_mode="flex",
            tactile_dim=config.tactile_dim,
            tactile_shape=getattr(config, "tactile_shape", (2, 32, 58)),
            tactile_token_grid=getattr(config, "tactile_token_grid", (4, 6)),
        )
        self._set_trainable_scope(
            self.transformer,
            getattr(config, "trainable_scope", "all"),
        )

        logger.info("Setting up activation checkpointing ...")
        apply_ac(self.transformer)

        logger.info("Setting up FSDP...")
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_model,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.transformer.train()
        total_params, trainable_params = self._count_params(self.transformer)
        logger.info(
            "Trainable scope=%s: %.2fM / %.2fM parameters trainable",
            getattr(config, "trainable_scope", "all"),
            trainable_params / 1e6,
            total_params / 1e6,
        )
        if trainable_params == 0:
            raise ValueError("No trainable parameters selected")

        self.optimizer = torch.optim.AdamW(
            [p for p in self.transformer.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: warmup_constant_lambda(
                step,
                warmup_steps=config.warmup_steps,
            ),
        )

        logger.info("Setting up tactile dataset...")
        train_dataset = MultiTactileLatentLeRobotDataset(
            config=config,
            num_init_worker=getattr(config, "dataset_init_worker", None),
        )
        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=config.world_size,
                rank=config.rank,
                shuffle=True,
                seed=42,
            )
            if config.world_size > 1
            else None
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.load_worker,
            sampler=train_sampler,
        )
        self.train_loader_iter = None

        self.train_scheduler_latent = FlowMatchScheduler(
            shift=self.config.snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.train_scheduler_action = FlowMatchScheduler(
            shift=self.config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.train_scheduler_tactile = FlowMatchScheduler(
            shift=self.config.tactile_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)
        self.train_scheduler_tactile.set_timesteps(1000, training=True)

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)

    @torch.no_grad()
    def _crop_training_batch(self, batch_dict):
        keep_frames = int(getattr(self.config, "train_frame_chunk_size", 0) or 0)
        if keep_frames <= 0:
            return batch_dict

        latent_frames = batch_dict["latents"].shape[2]
        if latent_frames <= keep_frames:
            return batch_dict

        start = torch.randint(
            0,
            latent_frames - keep_frames + 1,
            (1,),
        ).item()
        end = start + keep_frames
        tactile_per_frame = int(
            getattr(self.config, "tactile_per_frame", self.config.action_per_frame)
        )
        tactile_start = start * tactile_per_frame
        tactile_end = end * tactile_per_frame

        out = dict(batch_dict)
        out["latents"] = batch_dict["latents"][:, :, start:end].contiguous()
        out["actions"] = batch_dict["actions"][:, :, start:end].contiguous()
        out["actions_mask"] = batch_dict["actions_mask"][:, :, start:end].contiguous()
        out["tactile"] = batch_dict["tactile"][:, :, tactile_start:tactile_end].contiguous()
        out["tactile_mask"] = batch_dict["tactile_mask"][:, :, tactile_start:tactile_end].contiguous()
        return out

    @torch.no_grad()
    def _add_tactile_noise(self, tactile, tactile_mask):
        out = self._add_noise(
            latent=tactile,
            train_scheduler=self.train_scheduler_tactile,
            action_mask=tactile_mask,
            action_mode=True,
            noisy_cond_prob=getattr(self.config, "tactile_noisy_cond_prob", 0.5),
        )
        batch_size = tactile.shape[0]
        token_grid = getattr(self.config, "tactile_token_grid", (4, 6))
        out["grid_id"] = get_mesh_id(
            tactile.shape[-3],
            token_grid[0],
            token_grid[1],
            t=2,
            f_w=1,
            f_shift=0,
            action=False,
        ).to(self.device)[None].repeat(batch_size, 1, 1)
        out["tactile_mask"] = tactile_mask
        return out

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        batch_dict = self._crop_training_batch(batch_dict)
        latent_dict = self._add_noise(
            latent=batch_dict["latents"],
            train_scheduler=self.train_scheduler_latent,
            action_mask=None,
            action_mode=False,
            noisy_cond_prob=0.5,
        )
        action_dict = self._add_noise(
            latent=batch_dict["actions"],
            train_scheduler=self.train_scheduler_action,
            action_mask=batch_dict["actions_mask"],
            action_mode=True,
            noisy_cond_prob=0.0,
        )
        tactile_dict = self._add_tactile_noise(
            batch_dict["tactile"],
            batch_dict["tactile_mask"],
        )

        latent_dict["text_emb"] = batch_dict["text_emb"]
        action_dict["text_emb"] = batch_dict["text_emb"]
        tactile_dict["text_emb"] = batch_dict["text_emb"]
        action_dict["actions_mask"] = batch_dict["actions_mask"]

        return {
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "tactile_dict": tactile_dict,
            "chunk_size": torch.randint(1, 5, (1,)).item(),
            "window_size": torch.randint(4, 65, (1,)).item(),
        }

    def compute_loss(self, input_dict, pred):
        latent_pred, action_pred, tactile_pred = pred
        latent_targets = input_dict["latent_dict"]["targets"]
        action_targets = input_dict["action_dict"]["targets"]
        tactile_targets = input_dict["tactile_dict"]["targets"]

        action_pred = rearrange(
            action_pred,
            "b (f n) c -> b c f n 1",
            f=action_targets.shape[-3],
        )
        latent_pred = data_seq_to_patch(
            self.patch_size,
            latent_pred,
            latent_targets.shape[-3],
            latent_targets.shape[-2],
            latent_targets.shape[-1],
            batch_size=latent_pred.shape[0],
        )

        latent_loss = self._frame_loss(
            latent_pred,
            latent_targets,
            self.train_scheduler_latent.training_weight(
                input_dict["latent_dict"]["timesteps"].flatten()
            ).reshape(input_dict["latent_dict"]["timesteps"].shape),
        )
        action_loss = self._frame_loss(
            action_pred,
            action_targets,
            self.train_scheduler_action.training_weight(
                input_dict["action_dict"]["timesteps"].flatten()
            ).reshape(input_dict["action_dict"]["timesteps"].shape),
            input_dict["action_dict"]["actions_mask"],
        )
        tactile_loss = self._frame_loss(
            tactile_pred,
            tactile_targets,
            self.train_scheduler_tactile.training_weight(
                input_dict["tactile_dict"]["timesteps"].flatten()
            ).reshape(input_dict["tactile_dict"]["timesteps"].shape),
            input_dict["tactile_dict"]["tactile_mask"],
        )
        temporal_loss = self._temporal_tactile_loss(tactile_pred, tactile_targets)
        contact_loss = self._contact_tactile_loss(tactile_pred, tactile_targets)
        tactile_loss = tactile_loss * getattr(self.config, "tactile_loss_weight", 1.0)
        tactile_loss = tactile_loss + temporal_loss + contact_loss
        scale = self.gradient_accumulation_steps
        return latent_loss / scale, action_loss / scale, tactile_loss / scale

    def _temporal_tactile_loss(self, pred, target):
        weight = getattr(self.config, "tactile_temporal_loss_weight", 0.0)
        if weight <= 0 or pred.shape[2] < 2:
            return pred.new_tensor(0.0)
        pred_dt = pred[:, :, 1:] - pred[:, :, :-1]
        target_dt = target[:, :, 1:] - target[:, :, :-1]
        return F.l1_loss(pred_dt.float(), target_dt.float().detach()) * weight

    def _contact_tactile_loss(self, pred, target):
        weight = getattr(self.config, "tactile_contact_loss_weight", 0.0)
        if weight <= 0:
            return pred.new_tensor(0.0)
        threshold = getattr(self.config, "tactile_contact_threshold", 0.05)
        target_contact = (target.float() > threshold).float()
        pred_contact_logit = pred.float() * 4.0
        return F.binary_cross_entropy_with_logits(pred_contact_logit, target_contact) * weight

    def _frame_loss(self, pred, target, frame_weight, mask=None):
        loss = F.mse_loss(pred.float(), target.float().detach(), reduction="none")
        loss = loss * frame_weight[:, None, :, None, None]
        if mask is None:
            mask = torch.ones_like(loss, dtype=torch.bool)
        loss = loss * mask.float()
        loss = loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        mask = mask.float().permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        return (loss.sum(dim=1) / (mask.sum(dim=1) + 1e-6)).mean()

    def _train_step(self, batch, batch_idx):
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0

        self.transformer.set_requires_gradient_sync(should_sync)
        output = self.transformer(input_dict, train_mode=True)
        latent_loss, action_loss, tactile_loss = self.compute_loss(input_dict, output)
        loss = latent_loss + action_loss + tactile_loss
        loss.backward()

        losses = {
            "latent_loss": latent_loss.detach(),
            "action_loss": action_loss.detach(),
            "tactile_loss": tactile_loss.detach(),
            "should_log": False,
        }
        if should_sync:
            total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            losses["total_norm"] = total_norm
            losses["should_log"] = True
        return losses

    def save_checkpoint(self):
        try:
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
            if self.config.rank == 0:
                if getattr(self.config, "overwrite_checkpoint", False):
                    checkpoint_dir = self.save_dir / "checkpoint_latest"
                else:
                    checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)
                save_file(state_dict_bf16, transformer_dir / "diffusion_pytorch_model.safetensors")
                config_dict = dict(self.transformer.config)
                config_dict.pop("_name_or_path", None)
                with open(transformer_dir / "config.json", "w") as f:
                    json.dump(config_dict, f, indent=2)
                metadata = {
                    "step": self.step,
                    "trainable_scope": getattr(self.config, "trainable_scope", "all"),
                    "source_model": getattr(self.config, "wan22_pretrained_model_name_or_path", None),
                }
                with open(checkpoint_dir / "training_state.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Tactile checkpoint saved at {checkpoint_dir}")
            if dist.is_initialized():
                dist.barrier()
        except Exception as exc:
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {exc}")
            if dist.is_initialized():
                dist.barrier()

    def train(self):
        logger.info(f"Starting tactile training for {self.config.num_steps} steps...")
        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Tactile training",
            disable=(self.config.rank != 0),
            dynamic_ncols=True,
            initial=self.step,
        )
        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_tactile_losses = []
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            batch = self._get_next_batch()
            losses = self._train_step(batch, step_in_accumulation)
            accumulated_latent_losses.append(losses["latent_loss"])
            accumulated_action_losses.append(losses["action_loss"])
            accumulated_tactile_losses.append(losses["tactile_loss"])
            step_in_accumulation += 1

            if losses["should_log"]:
                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                tactile_loss_show = dist_mean(torch.stack(accumulated_tactile_losses).sum()).detach().cpu().item()
                max_tactile_loss_show = dist_max(torch.stack(accumulated_tactile_losses).sum()).detach().cpu().item()
                accumulated_latent_losses = []
                accumulated_action_losses = []
                accumulated_tactile_losses = []
                step_in_accumulation = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    progress_bar.n += 1
                    progress_bar.set_postfix(
                        {
                            "video": f"{latent_loss_show:.4f}",
                            "action": f"{action_loss_show:.4f}",
                            "tactile": f"{tactile_loss_show:.4f}",
                            "grad": f"{losses['total_norm'].item():.2f}",
                            "lr": f"{self.lr_scheduler.get_last_lr()[0]:.2e}",
                        }
                    )
                    if self.wandb is not None:
                        self.wandb.log(
                            {
                                "loss_metrics/global_avg_video_loss": latent_loss_show,
                                "loss_metrics/global_avg_action_loss": action_loss_show,
                                "loss_metrics/global_avg_tactile_loss": tactile_loss_show,
                                "loss_metrics/global_max_tactile_loss": max_tactile_loss_show,
                                "grad_norm": losses["total_norm"].item(),
                                "lr": self.lr_scheduler.get_last_lr()[0],
                            },
                            step=self.step,
                        )

                self.step += 1
                if self.step % self.config.save_interval == 0:
                    self.save_checkpoint()

            if dist.is_initialized():
                dist.barrier()

        progress_bar.close()
        logger.info("Tactile training completed.")


def run(args):
    config = apply_stats_json(TACTILE_CONFIGS[args.config_name])
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
        config.empty_emb_path = os.path.join(args.dataset_path, "empty_emb.pt")
    if args.model_path is not None:
        config.wan22_pretrained_model_name_or_path = args.model_path
    if args.stats_json_path is not None:
        config.stats_json_path = args.stats_json_path
        config = apply_stats_json(config)
    if args.num_steps is not None:
        config.num_steps = args.num_steps
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.train_frame_chunk_size is not None:
        config.train_frame_chunk_size = args.train_frame_chunk_size
    if args.train_frame_crop is False:
        config.train_frame_chunk_size = 0
    if args.trainable_scope is not None:
        config.trainable_scope = args.trainable_scope
    if args.load_worker is not None:
        config.load_worker = args.load_worker
    if args.dataset_init_worker is not None:
        config.dataset_init_worker = args.dataset_init_worker
    if args.gc_interval is not None:
        config.gc_interval = args.gc_interval
    if args.overwrite_checkpoint is not None:
        config.overwrite_checkpoint = args.overwrite_checkpoint
    if args.enable_wandb is not None:
        config.enable_wandb = args.enable_wandb
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    if args.save_root is not None:
        config.save_root = args.save_root
    trainer = TactileTrainer(config)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Train LingBot-VA with tactile tokens")
    parser.add_argument("--config-name", type=str, default="robotwin_tactile_train")
    parser.add_argument("--save-root", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--stats-json-path", type=str, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--train-frame-chunk-size", type=int, default=None)
    parser.add_argument("--train-frame-crop", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--trainable-scope",
        type=str,
        choices=[
            "all",
            "video",
            "action",
            "tactile",
            "tactile_io",
            "tactile_head",
            "tactile_action",
            "tactile_video",
            "modality_heads",
            "backbone",
        ],
        default=None,
    )
    parser.add_argument("--load-worker", type=int, default=None)
    parser.add_argument("--dataset-init-worker", type=int, default=None)
    parser.add_argument("--gc-interval", type=int, default=None)
    parser.add_argument("--overwrite-checkpoint", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-wandb", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
