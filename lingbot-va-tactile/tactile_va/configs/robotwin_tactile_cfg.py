import os
from copy import deepcopy

from easydict import EasyDict

from wan_va.configs.shared_config import va_shared_cfg


robotwin_tactile_cfg = EasyDict(deepcopy(va_shared_cfg))
robotwin_tactile_cfg.__name__ = "Config: VA PA-STE tactile"

robotwin_tactile_cfg.wan22_pretrained_model_name_or_path = "/path/to/pretrained/model"
robotwin_tactile_cfg.transformer_path = None
robotwin_tactile_cfg.infer_mode = "server"
robotwin_tactile_cfg.prompt = "make coffee following the demonstrated long-horizon manipulation stages"

robotwin_tactile_cfg.attn_window = 30
robotwin_tactile_cfg.frame_chunk_size = 4
robotwin_tactile_cfg.env_type = "none"

robotwin_tactile_cfg.height = 256
robotwin_tactile_cfg.width = 256
robotwin_tactile_cfg.action_dim = 30
# Default tactile real-machine data is converted at 30Hz and Wan latent
# extraction defaults to 30fps. Wan consumes 4 raw frames per latent frame, so
# each latent frame carries 4 action/tactile control steps.
robotwin_tactile_cfg.action_per_frame = 4
robotwin_tactile_cfg.obs_cam_keys = [
    "observation.images.cam_front",
    "observation.images.cam_side",
    "observation.images.cam_fisheye",
]

robotwin_tactile_cfg.guidance_scale = 5
robotwin_tactile_cfg.action_guidance_scale = 1
robotwin_tactile_cfg.num_inference_steps = 50
robotwin_tactile_cfg.video_exec_step = -1
robotwin_tactile_cfg.action_num_inference_steps = 50
robotwin_tactile_cfg.snr_shift = 5.0
robotwin_tactile_cfg.action_snr_shift = 1.0

# convert_to_lerobot.py defaults to joint-only single-arm action:
# [joint1, joint2, ..., joint7, gripper_distance_mm].
# LingBot-VA's 30D convention is:
# left EEF 0:7, right EEF 7:14, left joints 14:21,
# right joints 21:28, left gripper 28, right gripper 29.
# This dataset is single-arm, so we map it to the left-arm slots.
robotwin_tactile_cfg.action_layout_name = "joint_gripper"
robotwin_tactile_cfg.used_action_channel_ids = list(range(14, 21)) + [28]
inverse_used_action_channel_ids = [len(robotwin_tactile_cfg.used_action_channel_ids)] * robotwin_tactile_cfg.action_dim
for i, j in enumerate(robotwin_tactile_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
robotwin_tactile_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids
robotwin_tactile_cfg.action_norm_method = "quantiles"
_default_action_q01 = [0.0] * robotwin_tactile_cfg.action_dim
_default_action_q99 = [0.0] * robotwin_tactile_cfg.action_dim
for _raw_idx, _model_idx in enumerate(robotwin_tactile_cfg.used_action_channel_ids):
    _default_action_q01[_model_idx] = -180.0 if _raw_idx < 7 else 0.0
    _default_action_q99[_model_idx] = 180.0 if _raw_idx < 7 else 100.0
robotwin_tactile_cfg.norm_stat = {
    # Replaced by tactile_stats.json before serious training.
    "q01": _default_action_q01,
    "q99": _default_action_q99,
}

# convert_to_lerobot.py writes one PA-STE tensor for this raw dataset:
# observation.tactile shape = (2, 32, 58), flattened to 3712 taxel channels.
robotwin_tactile_cfg.tactile_keys = ["observation.tactile"]
robotwin_tactile_cfg.tactile_dim = 2 * 32 * 58
robotwin_tactile_cfg.tactile_shape = (2, 32, 58)
robotwin_tactile_cfg.tactile_token_grid = (4, 6)
robotwin_tactile_cfg.tactile_per_frame = robotwin_tactile_cfg.action_per_frame
robotwin_tactile_cfg.tactile_norm_method = "quantiles"
robotwin_tactile_cfg.tactile_norm_stat = {
    "q01": [0.0] * robotwin_tactile_cfg.tactile_dim,
    "q99": [1.0] * robotwin_tactile_cfg.tactile_dim,
}
robotwin_tactile_cfg.tactile_min_range = 1e-4
robotwin_tactile_cfg.tactile_mask_low_range = True
robotwin_tactile_cfg.tactile_snr_shift = robotwin_tactile_cfg.action_snr_shift
robotwin_tactile_cfg.tactile_noisy_cond_prob = 0.5
robotwin_tactile_cfg.tactile_num_inference_steps = robotwin_tactile_cfg.action_num_inference_steps
robotwin_tactile_cfg.tactile_guidance_scale = 1
robotwin_tactile_cfg.predict_tactile_before_action = True
robotwin_tactile_cfg.return_predicted_tactile = False
robotwin_tactile_cfg.tactile_loss_weight = 1.0
robotwin_tactile_cfg.tactile_contact_loss_weight = 0.1
robotwin_tactile_cfg.tactile_temporal_loss_weight = 0.1
robotwin_tactile_cfg.tactile_contact_threshold = 0.05
robotwin_tactile_cfg.stats_json_path = "lingbot-va-tactile/tactile_stats.json"


robotwin_tactile_lowmem_cfg = EasyDict(deepcopy(robotwin_tactile_cfg))
robotwin_tactile_lowmem_cfg.__name__ = "Config: VA PA-STE tactile low-memory inference"
# Low-memory server profile for single 48 GB GPUs. This preserves the tactile
# prediction branch but shortens rollout chunks, diffusion sampling, and cache.
robotwin_tactile_lowmem_cfg.enable_offload = False
robotwin_tactile_lowmem_cfg.attn_window = 30
robotwin_tactile_lowmem_cfg.frame_chunk_size = 4
robotwin_tactile_lowmem_cfg.guidance_scale = 1
robotwin_tactile_lowmem_cfg.action_guidance_scale = 1
robotwin_tactile_lowmem_cfg.tactile_guidance_scale = 1
robotwin_tactile_lowmem_cfg.num_inference_steps = 20
robotwin_tactile_lowmem_cfg.action_num_inference_steps = 20
robotwin_tactile_lowmem_cfg.tactile_num_inference_steps = 20


robotwin_tactile_train_cfg = EasyDict(deepcopy(robotwin_tactile_cfg))
robotwin_tactile_train_cfg.__name__ = "Config: VA robotwin tactile train"
robotwin_tactile_train_cfg.dataset_path = "/path/to/your/tactile/dataset"
robotwin_tactile_train_cfg.empty_emb_path = os.path.join(
    robotwin_tactile_train_cfg.dataset_path,
    "empty_emb.pt",
)
robotwin_tactile_train_cfg.enable_wandb = False
robotwin_tactile_train_cfg.dataset_init_worker = 1
robotwin_tactile_train_cfg.load_worker = 16
robotwin_tactile_train_cfg.trainable_scope = "all"
robotwin_tactile_train_cfg.enable_lora = False
robotwin_tactile_train_cfg.lora_rank = 16
robotwin_tactile_train_cfg.lora_alpha = 32.0
robotwin_tactile_train_cfg.lora_dropout = 0.0
robotwin_tactile_train_cfg.lora_target = "attention_ffn"
robotwin_tactile_train_cfg.train_frame_chunk_size = 0
robotwin_tactile_train_cfg.save_interval = 10
robotwin_tactile_train_cfg.overwrite_checkpoint = False
robotwin_tactile_train_cfg.gc_interval = 50
robotwin_tactile_train_cfg.cfg_prob = 0.1
robotwin_tactile_train_cfg.learning_rate = 1e-5
robotwin_tactile_train_cfg.beta1 = 0.9
robotwin_tactile_train_cfg.beta2 = 0.95
robotwin_tactile_train_cfg.weight_decay = 0.1
robotwin_tactile_train_cfg.warmup_steps = 10
robotwin_tactile_train_cfg.batch_size = 1
robotwin_tactile_train_cfg.gradient_accumulation_steps = 1
robotwin_tactile_train_cfg.num_steps = 50000
