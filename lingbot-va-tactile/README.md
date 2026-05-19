# LingBot-VA Tactile Overlay

This folder is a sidecar implementation for adding a real piezoresistive tactile token stream to LingBot-VA without editing the original `wan_va/` project files.

The design keeps the original video/action streams and adds a third stream:

```text
video tokens   -> Wan VAE visual latents
action tokens  -> robot actions
tactile tokens -> normalized piezoresistive taxel vectors
```

## What Is Implemented

- `tactile_va.dataset.TactileLatentLeRobotDataset`
  - Loads normal video latents and actions through the original LingBot-VA dataset logic.
  - Loads one or more LeRobot tactile columns, aligns them to latent/action timing, normalizes them, and returns `tactile` plus `tactile_mask`.

- `tactile_va.modules.TactileWanTransformer3DModel`
  - Adds a tactile-native CNN tokenizer, spatial decoder, and `condition_embedder_tactile`.
  - Supports tactile tokens in training with `[noisy, clean]` conditioning pairs.
  - Supports tactile-only forward passes for inference/cache updates.

- `tactile_va.train_tactile`
  - Fine-tuning entrypoint that keeps original pretrained video/action/backbone weights and randomly initializes tactile-specific layers.

- `tactile_va.server_tactile`
  - Server-side extension that preprocesses observed tactile arrays and writes them into the transformer KV cache as contact history.

## Expected Tactile Data Format

Add tactile columns to your LeRobot dataset. For example:

```text
observation.tactile -> float tensor [T, 2, 32, 58]
```

Use raw or calibrated piezoresistive values, but compute quantile normalization stats from the training set and place them in config:

```python
tactile_keys = ["observation.tactile"]
tactile_dim = 3712
tactile_per_frame = 16
tactile_norm_stat = {
    "q01": [...],  # length tactile_dim
    "q99": [...],  # length tactile_dim
}
```

If multiple tactile keys are configured, they are concatenated on the channel dimension. In that case `tactile_dim` must equal the total flattened dimension.

The model no longer flattens tactile into a single token. The dataset preserves taxel geometry:

```text
[T, 2, 32, 58]
-> normalize per taxel
-> [2, T, 32, 58]
-> CNN tactile tokenizer
-> adaptive token grid, default 4 x 6 tokens per tactile timestep
-> shared LingBot transformer
-> spatial tactile decoder
-> [2, T, 32, 58]
```

## Recommended Calibration

Before saving tactile values:

1. Subtract per-taxel no-contact baseline.
2. Clip impossible ADC spikes and sensor saturation.
3. Apply a small temporal median or low-pass filter.
4. Normalize by dataset quantiles, not min/max.
5. Preserve contact onset timing; do not over-smooth.

Do not convert tactile into RGB pseudo-images for the main method. That is useful as an ablation, but the main path expects calibrated numeric pressure maps with shape `(2, 32, 58)`.

## Running

The original `script/run_va_posttrain.sh` calls `wan_va.train`, so for tactile training use the new module directly:

```bash
PYTHONPATH="$PWD:$PWD/lingbot-va-tactile" \
torchrun --nproc_per_node=8 -m tactile_va.train_tactile --config-name robotwin_tactile_train
```

For server mode:

```bash
PYTHONPATH="$PWD:$PWD/lingbot-va-tactile" \
torchrun --nproc_per_node=1 -m tactile_va.server_tactile --config-name robotwin_tactile
```

### LoRA Fine-Tuning

For a 48 GB single GPU, avoid `--trainable-scope all`. A practical setting is
to fully train the new tactile/action modules while adding LoRA adapters to the
frozen transformer backbone:

```bash
python -m torch.distributed.run --nproc_per_node=1 \
  lingbot-va-tactile/tactile_va/train_tactile.py \
  --config-name robotwin_tactile_train \
  --model-path /data/lingbot-va-models/lingbot-va-base \
  --dataset-path /data/data_realworld/lerobot_export_dataset/local/insert-peg-cylinder-realmachine \
  --stats-json-path lingbot-va-tactile/tactile_stats.json \
  --save-root /data/lingbot-va-models/lingbot-va-tactile-lora-ft \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --dataset-init-worker 1 \
  --load-worker 0 \
  --gradient-accumulation-steps 1 \
  --train-frame-chunk-size 4 \
  --gc-interval 1 \
  --trainable-scope tactile_action \
  --enable-lora \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-target attention \
  --overwrite-checkpoint
```

Useful LoRA targets:

- `attention`: safest memory setting; adapts self/cross attention.
- `attention_ffn`: stronger adaptation; higher memory and trainable count.
- `self_attention`, `cross_attention`, `ffn`: targeted ablations.
- `all_block_linear`: broadest adapter coverage; use only if memory allows.

Saved LoRA checkpoints are inference-compatible with the normal loader:

```text
checkpoint_latest/
├── transformer/      # full transformer weights with LoRA merged in
└── lora_adapter/     # adapter-only tensors and metadata for bookkeeping
```

Use `checkpoint_latest/transformer` or just `checkpoint_latest` in offline
validation and server commands; no extra adapter loading flag is needed.

### RealMan + Pika Deployment

Start the tactile inference server first. Use the base model path for VAE,
tokenizer, and text encoder, and point `--transformer-path` to the fine-tuned
tactile checkpoint:

```bash
python -m torch.distributed.run --nproc_per_node=1 \
  lingbot-va-tactile/tactile_va/server_tactile.py \
  --config-name robotwin_tactile \
  --model-path /data/lingbot-va-models/lingbot-va-base \
  --transformer-path /data/lingbot-va-models/lingbot-va-tactile-ft/checkpoints/checkpoint_latest \
  --stats-json-path lingbot-va-tactile/tactile_stats.json \
  --port 29536
```

Then run the local hardware client:

```bash
python lingbot-va-tactile/deploy_realman_tactile.py \
  --server-host 127.0.0.1 \
  --server-port 29536 \
  --prompt "insert the peg into the cylinder hole" \
  --pika-project-root /home/tujian/Projects/pika_sdk/PIKA_RM65B_data_acquisition \
  --front-camera-id 12 \
  --side-camera-id -1 \
  --realsense-serial 315122271124 \
  --gripper-port /dev/ttyUSB82 \
  --tactile-port 0 \
  --tactile-splitted-file config_mapping_gripper.json \
  --tactile-calibrate-file calibration_gripper.json \
  --tactile-target-keys 0,1 \
  --action-mode print \
  --visualize
```

`--action-mode print` is the safe dry mode. The current tactile dataset trains
an 8D end-effector target `[x, y, z, qx, qy, qz, qw, gripper]`; use
`--action-mode ee_pose` only after confirming the RealMan Cartesian API method
and orientation convention. Use `--action-mode joint` only for checkpoints
trained with joint-space actions.

The tactile runtime path mirrors
`/home/tujian/Projects/pika_sdk/PIKA_RM65B_data_acquisition/main_teleop.py`:
`TactileDataProvider` is started on `--tactile-port`, zero-calibrated after
warmup, and every inference step reads `get_latest_data()[0]`. Keys `0,1` are
stacked into the model tensor `[2, 32, 58]` in that order. Keep the gripper
unloaded during zero calibration, or pass `--no-tactile-zero-calibration` if you
already calibrated the sensor externally.

## PIKA Real-Machine Pipeline

For `/data/Datasets/PIKA_real_original/insert_peg_cylinder_RealMachine`, the raw data is compatible with the converter in this folder:

```bash
python lingbot-va-tactile/inspect_raw_pika.py \
  --input-root /data/Datasets/PIKA_real_original/insert_peg_cylinder_RealMachine

python lingbot-va-tactile/convert_to_lerobot.py \
  --input-root /data/Datasets/PIKA_real_original/insert_peg_cylinder_RealMachine \
  --repo-id local/insert-peg-cylinder-RM75B-PIKA \
  --output-root lerobot_export_dataset \
  --overwrite
```

The converted LeRobot dataset is not yet directly trainable by LingBot-VA. You still need to:

1. Ensure `meta/episodes.jsonl` has `action_config`.
2. Extract Wan VAE latents for `observation.images.cam_front`, `observation.images.cam_side`, and `observation.images.cam_fisheye`.
3. Create `empty_emb.pt` with the same Wan text encoder used by the base project.
4. Compute action/tactile quantiles.

`convert_to_lerobot.py` now writes full-episode `action_config` automatically. For an existing converted dataset, patch it with:

```bash
python lingbot-va-tactile/add_action_config.py \
  --dataset-root /path/to/converted/lerobot/dataset
```

Stats helper:

```bash
python lingbot-va-tactile/compute_lerobot_tactile_stats.py \
  --repo-id local/insert-peg-cylinder-RM75B-PIKA \
  --root lerobot_export_dataset/local/insert-peg-cylinder-RM75B-PIKA \
  --output-json lingbot-va-tactile/tactile_stats.json
```

`robotwin_tactile_cfg.py` reads `lingbot-va-tactile/tactile_stats.json` automatically at train/server startup.

Latent extraction helper:

```bash
PYTHONPATH="$PWD:$PWD/lingbot-va-tactile" \
python lingbot-va-tactile/extract_wan_latents.py \
  --repo-id local/insert-peg-cylinder-RM75B-PIKA \
  --dataset-root lerobot_export_dataset/local/insert-peg-cylinder-RM75B-PIKA \
  --wan22-pretrained-model-name-or-path /path/to/lingbot-va-or-wan22-model \
  --fps 10 \
  --height 256 \
  --width 256
```

Current inspected raw shape:

```text
observation.tactile = (2, 32, 58)
tactile_dim = 3712
action_dim = 8, padded onto LingBot-VA's 30-channel action canvas
```

## Checkpoint Compatibility

This is not strict checkpoint-compatible with released LingBot-VA weights because the tactile layers are new. The loader intentionally uses partial loading:

- original matching weights are loaded;
- tactile-specific weights are randomly initialized;
- shape-mismatched weights are skipped.

Fine-tuning on synchronized visual/action/tactile demonstrations is required.
