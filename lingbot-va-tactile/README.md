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
