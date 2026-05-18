# Copyright 2024-2025 The Robbyant Team Authors.
# Tactile overlay additions are kept outside the original wan_va package.

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from einops import rearrange

from wan_va.modules.model import FlexAttnFunc, WanTransformer3DModel


class TactileSpatialTokenizer(nn.Module):
    """CNN tactile tokenizer preserving PA-STE spatial geometry.

    Input:  [B, C, F, H, W]
    Output: [B, F * Gh * Gw, D]
    """

    def __init__(self, in_channels, hidden_dim, token_grid=(4, 6), stem_dim=128):
        super().__init__()
        self.token_grid = tuple(token_grid)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, stem_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, stem_dim),
            nn.SiLU(),
            nn.Conv2d(stem_dim, hidden_dim, kernel_size=1),
        )

    def forward(self, tactile):
        batch_size, channels, frames, height, width = tactile.shape
        x = rearrange(tactile, "b c f h w -> (b f) c h w")
        x = self.stem(x)
        x = F.adaptive_avg_pool2d(x, self.token_grid)
        return rearrange(
            x,
            "(b f) d gh gw -> b (f gh gw) d",
            b=batch_size,
            f=frames,
        )


class TactileSpatialDecoder(nn.Module):
    """Decode tactile tokens back to PA-STE pressure maps."""

    def __init__(self, out_channels, hidden_dim, tactile_hw=(32, 58), token_grid=(4, 6), decoder_dim=128):
        super().__init__()
        self.tactile_hw = tuple(tactile_hw)
        self.token_grid = tuple(token_grid)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, decoder_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, decoder_dim),
            nn.SiLU(),
            nn.Conv2d(decoder_dim, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def forward(self, tokens, batch_size, frames):
        gh, gw = self.token_grid
        x = rearrange(tokens, "b (f gh gw) d -> (b f) d gh gw", f=frames, gh=gh, gw=gw)
        x = F.interpolate(x, size=self.tactile_hw, mode="bilinear", align_corners=False)
        x = self.decoder(x)
        return rearrange(x, "(b f) c h w -> b c f h w", b=batch_size, f=frames)


class TactileWanTransformer3DModel(WanTransformer3DModel):
    """WanTransformer3DModel with an explicit tactile token family.

    The original model supports video latent and action tokens. This subclass
    adds tactile tokens while keeping the original public behavior intact.
    """

    _skip_layerwise_casting_patterns = WanTransformer3DModel._skip_layerwise_casting_patterns + [
        "tactile_embedder",
        "tactile_proj_out",
        "condition_embedder_tactile",
    ]
    _keep_in_fp32_modules = WanTransformer3DModel._keep_in_fp32_modules + [
        "tactile_norm1",
        "tactile_norm2",
        "tactile_norm3",
    ]

    @register_to_config
    def __init__(
        self,
        patch_size=[1, 2, 2],
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        action_dim=30,
        tactile_dim=64,
        tactile_shape=(2, 32, 58),
        tactile_token_grid=(4, 6),
        text_dim=4096,
        freq_dim=256,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        eps=1e-06,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        attn_mode="torch",
    ):
        super().__init__(
            patch_size=patch_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            action_dim=action_dim,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            rope_max_seq_len=rope_max_seq_len,
            pos_embed_seq_len=pos_embed_seq_len,
            attn_mode=attn_mode,
        )
        inner_dim = num_attention_heads * attention_head_dim
        self.tactile_dim = tactile_dim
        self.tactile_shape = tuple(tactile_shape)
        self.tactile_token_grid = tuple(tactile_token_grid)
        tactile_channels, tactile_height, tactile_width = self.tactile_shape
        self.tactile_embedder = TactileSpatialTokenizer(
            in_channels=tactile_channels,
            hidden_dim=inner_dim,
            token_grid=self.tactile_token_grid,
        )
        self.tactile_proj_out = TactileSpatialDecoder(
            out_channels=tactile_channels,
            hidden_dim=inner_dim,
            tactile_hw=(tactile_height, tactile_width),
            token_grid=self.tactile_token_grid,
        )
        self.condition_embedder_tactile = deepcopy(self.condition_embedder_action)

    @torch.no_grad()
    def _init_three_stream_mask(
        self,
        latent_shape,
        action_shape,
        tactile_shape,
        text_length,
        padded_length,
        chunk_size,
        window_size,
        device,
    ):
        B, _, L_F, L_H, L_W = latent_shape
        _, _, A_F, A_H, A_W = action_shape
        _, _, T_F, _, _ = tactile_shape
        patch_f, patch_h, patch_w = self.patch_size
        tactile_gh, tactile_gw = self.tactile_token_grid

        latent_seq_id = (
            torch.arange(B)[:, None, None, None]
            .expand(-1, L_F // patch_f, L_H // patch_h, L_W // patch_w)
            .flatten()
        )
        action_seq_id = (
            torch.arange(B)[:, None, None, None]
            .expand(-1, A_F, A_H, A_W)
            .flatten()
        )
        tactile_seq_id = (
            torch.arange(B)[:, None, None, None]
            .expand(-1, T_F, tactile_gh, tactile_gw)
            .flatten()
        )
        seq_ids = torch.cat(
            [latent_seq_id] * 2
            + [action_seq_id] * 2
            + [tactile_seq_id] * 2,
        )

        latent_frame_id = (
            torch.arange(L_F)[None, :, None, None]
            .expand(B, -1, L_H // patch_h, L_W // patch_w)
            .flatten()
        )
        action_frame_id = (
            torch.arange(A_F)[None, :, None, None]
            .expand(B, -1, A_H, A_W)
            .flatten()
        )
        tactile_frame_id = (
            torch.arange(T_F)[None, :, None, None]
            .expand(B, -1, tactile_gh, tactile_gw)
            .flatten()
        )
        tactile_per_action_frame = max(1, T_F // max(1, A_F))
        frame_ids = torch.cat(
            [latent_frame_id // chunk_size * 2] * 2
            + [action_frame_id // chunk_size * 2 + 1] * 2
            + [
                (tactile_frame_id // tactile_per_action_frame) // chunk_size * 2
                + 1
            ]
            * 2,
        )

        noise_ids = torch.cat(
            [
                torch.zeros_like(latent_frame_id),
                torch.ones_like(latent_frame_id),
                torch.zeros_like(action_frame_id),
                torch.ones_like(action_frame_id),
                torch.zeros_like(tactile_frame_id),
                torch.ones_like(tactile_frame_id),
            ]
        )

        seq_ids = F.pad(seq_ids, (0, padded_length), value=-1)
        frame_ids = F.pad(frame_ids, (0, padded_length), value=-1)
        noise_ids = F.pad(noise_ids, (0, padded_length), value=-1)

        mask_mod = FlexAttnFunc._get_mask_mod(
            seq_ids.long().to(device),
            frame_ids.long().to(device),
            noise_ids.long().to(device),
            window_size,
        )
        FlexAttnFunc.attention_mask = FlexAttnFunc.compiled_create_block_mask(
            mask_mod,
            1,
            1,
            len(seq_ids),
            len(seq_ids),
            device=device,
            _compile=True,
        )

        text_seq_ids = (
            torch.arange(B)[:, None]
            .expand(-1, text_length)
            .flatten()
        )
        mask_mod_cross = FlexAttnFunc._get_cross_mask_mod(
            seq_ids.long().to(device),
            text_seq_ids.long().to(device),
        )
        FlexAttnFunc.cross_attention_mask = FlexAttnFunc.compiled_create_block_mask(
            mask_mod_cross,
            1,
            1,
            len(seq_ids),
            len(text_seq_ids),
            device=device,
            _compile=True,
        )

    def create_empty_cache(
        self,
        cache_name,
        attn_window,
        latent_token_per_chunk,
        action_token_per_chunk,
        device,
        dtype,
        batch_size,
        tactile_token_per_chunk=0,
    ):
        total_tolen = (
            (attn_window // 2) * latent_token_per_chunk
            + (attn_window // 2) * action_token_per_chunk
            + (attn_window // 2) * tactile_token_per_chunk
        )
        for block in self.blocks:
            block.attn1.init_kv_cache(
                cache_name,
                total_tolen,
                self.num_attention_heads,
                self.attention_head_dim,
                device,
                dtype,
                batch_size,
            )

    def _input_embed(self, latents, input_type="latent"):
        if input_type == "tactile":
            return self.tactile_embedder(latents)
        return super()._input_embed(latents, input_type=input_type)

    def _time_embed(self, timesteps, H, W, dtype, action_mode=False, tactile_mode=False):
        pach_scale_h, pach_scale_w = (1, 1) if (action_mode or tactile_mode) else (
            self.patch_size[1],
            self.patch_size[2],
        )
        latent_time_steps = torch.repeat_interleave(
            timesteps,
            (H // pach_scale_h) * (W // pach_scale_w),
            dim=1,
        )
        if tactile_mode:
            current_condition_embedder = self.condition_embedder_tactile
        elif action_mode:
            current_condition_embedder = self.condition_embedder_action
        else:
            current_condition_embedder = self.condition_embedder
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps,
            dtype=dtype,
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
        return temb, timestep_proj

    def forward_train(self, input_dict):
        latent_dict = input_dict["latent_dict"]
        action_dict = input_dict["action_dict"]
        tactile_dict = input_dict["tactile_dict"]

        latent_dict["noisy_latents"] = latent_dict["noisy_latents"].to(torch.bfloat16)
        latent_dict["latent"] = latent_dict["latent"].to(torch.bfloat16)
        action_dict["noisy_latents"] = action_dict["noisy_latents"].to(torch.bfloat16)
        action_dict["latent"] = action_dict["latent"].to(torch.bfloat16)
        tactile_dict["noisy_latents"] = tactile_dict["noisy_latents"].to(torch.bfloat16)
        tactile_dict["latent"] = tactile_dict["latent"].to(torch.bfloat16)

        batch_size = latent_dict["noisy_latents"].shape[0]

        latent_hidden_states = self._input_embed(
            latent_dict["noisy_latents"],
            input_type="latent",
        ).flatten(0, 1)[None]
        condition_latent_hidden_states = self._input_embed(
            latent_dict["latent"],
            input_type="latent",
        ).flatten(0, 1)[None]

        action_hidden_states = self._input_embed(
            action_dict["noisy_latents"],
            input_type="action",
        ).flatten(0, 1)[None]
        condition_action_hidden_states = self._input_embed(
            action_dict["latent"],
            input_type="action",
        ).flatten(0, 1)[None]

        tactile_hidden_states = self._input_embed(
            tactile_dict["noisy_latents"],
            input_type="tactile",
        ).flatten(0, 1)[None]
        condition_tactile_hidden_states = self._input_embed(
            tactile_dict["latent"],
            input_type="tactile",
        ).flatten(0, 1)[None]

        text_hidden_states = self._input_embed(latent_dict["text_emb"], input_type="text")
        text_hidden_states = text_hidden_states.flatten(0, 1)[None]

        hidden_states = torch.cat(
            [
                latent_hidden_states,
                condition_latent_hidden_states,
                action_hidden_states,
                condition_action_hidden_states,
                tactile_hidden_states,
                condition_tactile_hidden_states,
            ],
            dim=1,
        )

        latent_grid_id = latent_dict["grid_id"].permute(1, 0, 2).flatten(1)[None]
        action_grid_id = action_dict["grid_id"].permute(1, 0, 2).flatten(1)[None]
        tactile_grid_id = tactile_dict["grid_id"].permute(1, 0, 2).flatten(1)[None]
        full_grid_id = torch.cat(
            [latent_grid_id] * 2 + [action_grid_id] * 2 + [tactile_grid_id] * 2,
            dim=2,
        )
        rotary_emb = self.rope(full_grid_id)[:, :, None]

        latent_time_steps = torch.cat(
            [
                latent_dict["timesteps"].flatten(0, 1),
                latent_dict["cond_timesteps"].flatten(0, 1),
            ]
        )[None]
        action_time_steps = torch.cat(
            [
                action_dict["timesteps"].flatten(0, 1),
                action_dict["cond_timesteps"].flatten(0, 1),
            ]
        )[None]
        tactile_time_steps = torch.cat(
            [
                tactile_dict["timesteps"].flatten(0, 1),
                tactile_dict["cond_timesteps"].flatten(0, 1),
            ]
        )[None]

        latent_temb, latent_timestep_proj = self._time_embed(
            latent_time_steps,
            latent_dict["noisy_latents"].shape[-2],
            latent_dict["noisy_latents"].shape[-1],
            dtype=hidden_states.dtype,
        )
        action_temb, action_timestep_proj = self._time_embed(
            action_time_steps,
            action_dict["noisy_latents"].shape[-2],
            action_dict["noisy_latents"].shape[-1],
            dtype=hidden_states.dtype,
            action_mode=True,
        )
        tactile_temb, tactile_timestep_proj = self._time_embed(
            tactile_time_steps,
            self.tactile_token_grid[0],
            self.tactile_token_grid[1],
            dtype=hidden_states.dtype,
            tactile_mode=True,
        )
        temb = torch.cat([latent_temb, action_temb, tactile_temb], dim=1)
        timestep_proj = torch.cat(
            [latent_timestep_proj, action_timestep_proj, tactile_timestep_proj],
            dim=1,
        )

        total_length = hidden_states.shape[1]
        padded_length = (128 - total_length % 128) % 128
        hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length))
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        temb = F.pad(temb, (0, 0, 0, padded_length))
        timestep_proj = F.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))

        split_list = [
            latent_hidden_states.shape[1],
            condition_latent_hidden_states.shape[1],
            action_hidden_states.shape[1],
            condition_action_hidden_states.shape[1],
            tactile_hidden_states.shape[1],
            condition_tactile_hidden_states.shape[1],
            padded_length,
        ]

        if getattr(self.blocks[0], "attn_mode", None) == "flex":
            self._init_three_stream_mask(
                latent_shape=latent_dict["noisy_latents"].shape,
                action_shape=action_dict["noisy_latents"].shape,
                tactile_shape=tactile_dict["noisy_latents"].shape,
                text_length=latent_dict["text_emb"].shape[1],
                padded_length=padded_length,
                chunk_size=input_dict["chunk_size"],
                window_size=input_dict["window_size"],
                device=hidden_states.device,
            )

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                text_hidden_states,
                timestep_proj,
                rotary_emb,
                update_cache=False,
            )

        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(
            2,
            dim=1,
        )
        shift = shift.to(hidden_states.device).squeeze(1)
        scale = scale.to(hidden_states.device).squeeze(1)
        hidden_states = (
            self.norm_out(hidden_states.float()) * (1.0 + scale) + shift
        ).type_as(hidden_states)

        (
            latent_hidden_states,
            _,
            action_hidden_states,
            _,
            tactile_hidden_states,
            _,
            _,
        ) = torch.split(hidden_states, split_list, dim=1)

        latent_hidden_states = self.proj_out(latent_hidden_states)
        latent_hidden_states = rearrange(
            latent_hidden_states,
            "1 (b l) (n c) -> b (l n) c",
            n=math.prod(self.patch_size),
            b=batch_size,
        )
        action_hidden_states = self.action_proj_out(action_hidden_states)
        action_hidden_states = rearrange(
            action_hidden_states,
            "1 (b l) c -> b l c",
            b=batch_size,
        )
        tactile_hidden_states = rearrange(tactile_hidden_states, "1 (b l) c -> b l c", b=batch_size)
        tactile_hidden_states = self.tactile_proj_out(
            tactile_hidden_states,
            batch_size=batch_size,
            frames=tactile_dict["targets"].shape[-3],
        )
        return latent_hidden_states, action_hidden_states, tactile_hidden_states

    def forward(
        self,
        input_dict,
        update_cache=0,
        cache_name="pos",
        action_mode=False,
        tactile_mode=False,
        train_mode=False,
    ):
        if train_mode:
            return self.forward_train(input_dict)
        if not tactile_mode:
            return super().forward(
                input_dict,
                update_cache=update_cache,
                cache_name=cache_name,
                action_mode=action_mode,
                train_mode=False,
            )

        batch_size = input_dict["noisy_latents"].shape[0]
        frames = input_dict["noisy_latents"].shape[2]
        hidden_states = self.tactile_embedder(input_dict["noisy_latents"])
        text_hidden_states = self.condition_embedder.text_embedder(input_dict["text_emb"])
        rotary_emb = self.rope(input_dict["grid_id"])[:, :, None]

        tactile_time_steps = torch.repeat_interleave(
            input_dict["timesteps"],
            self.tactile_token_grid[0] * self.tactile_token_grid[1],
            dim=1,
        )
        temb, timestep_proj = self.condition_embedder_tactile(
            tactile_time_steps,
            dtype=hidden_states.dtype,
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1))

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                text_hidden_states,
                timestep_proj,
                rotary_emb,
                update_cache=update_cache,
                cache_name=cache_name,
            )

        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(
            2,
            dim=1,
        )
        hidden_states = (
            self.norm_out(hidden_states.float()) * (1.0 + scale.squeeze(1))
            + shift.squeeze(1)
        ).type_as(hidden_states)
        return self.tactile_proj_out(hidden_states, batch_size=batch_size, frames=frames)
