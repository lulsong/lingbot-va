"""Small LoRA utilities for LingBot-VA-Tactile.

The implementation intentionally avoids a PEFT dependency so checkpoints can be
merged back into the normal Diffusers transformer format before saving.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target: str = "attention_ffn"


class LoRALinear(nn.Module):
    """Linear layer with a trainable low-rank residual branch."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_layer)!r}")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base_layer.out_features, bias=False)

        self.base_layer.requires_grad_(False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def in_features(self):
        return self.base_layer.in_features

    @property
    def out_features(self):
        return self.base_layer.out_features

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        out = self.base_layer(x)
        lora = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return out + lora.to(dtype=out.dtype)

    def merged_weight(self):
        delta = self.lora_B.weight.float() @ self.lora_A.weight.float()
        delta = delta.reshape_as(self.base_layer.weight.float()) * self.scaling
        return self.base_layer.weight.float() + delta

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4g}"
        )


def _get_submodule(root: nn.Module, path: str) -> nn.Module:
    cur = root
    if not path:
        return cur
    for part in path.split("."):
        cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
    return cur


def _replace_submodule(root: nn.Module, path: str, module: nn.Module) -> None:
    parent_path, _, child_name = path.rpartition(".")
    parent = _get_submodule(root, parent_path)
    if child_name.isdigit():
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def _matches_target(name: str, target: str) -> bool:
    if not name.startswith("blocks."):
        return False

    is_self_attn = ".attn1." in name
    is_cross_attn = ".attn2." in name
    is_attention = is_self_attn or is_cross_attn
    is_ffn = ".ffn." in name

    target = target.lower()
    if target == "attention":
        return is_attention
    if target == "self_attention":
        return is_self_attn
    if target == "cross_attention":
        return is_cross_attn
    if target == "ffn":
        return is_ffn
    if target == "attention_ffn":
        return is_attention or is_ffn
    if target == "all_block_linear":
        return True
    raise ValueError(
        f"Unknown LoRA target={target!r}; choose from attention, self_attention, "
        "cross_attention, ffn, attention_ffn, all_block_linear"
    )


def apply_lora(model: nn.Module, config: LoRAConfig) -> list[str]:
    """Replace selected block Linear layers with LoRALinear modules."""
    replacements: list[str] = []
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            continue
        if not isinstance(module, nn.Linear):
            continue
        if not _matches_target(name, config.target):
            continue

        lora_module = LoRALinear(
            module,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
        )
        _replace_submodule(model, name, lora_module)
        replacements.append(name)

    if not replacements:
        raise RuntimeError(f"No Linear modules matched LoRA target={config.target!r}")
    return replacements


def lora_trainable_parameter_count(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            total += module.lora_A.weight.numel()
            total += module.lora_B.weight.numel()
    return total


def lora_adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    out = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        out[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
        out[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
    return out


def lora_metadata(config: LoRAConfig, module_names: list[str]) -> dict:
    return {
        "config": asdict(config),
        "num_lora_modules": len(module_names),
        "module_names": module_names,
        "format": "lingbot_va_tactile_lora_adapter_v1",
    }


def merge_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    lora_config: LoRAConfig | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return a normal transformer state dict plus adapter-only tensors.

    Input state dict is expected to come from a model containing LoRALinear.
    Keys like ``blocks.0.attn1.to_q.base_layer.weight`` are converted back to
    ``blocks.0.attn1.to_q.weight`` with the LoRA delta merged in.
    """
    merged: dict[str, torch.Tensor] = {}
    adapter: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    suffix = ".base_layer.weight"
    for key, weight in state_dict.items():
        if not key.endswith(suffix):
            continue
        prefix = key[: -len(suffix)]
        bias_key = f"{prefix}.base_layer.bias"
        a_key = f"{prefix}.lora_A.weight"
        b_key = f"{prefix}.lora_B.weight"
        if a_key not in state_dict or b_key not in state_dict:
            continue

        a = state_dict[a_key].float()
        b = state_dict[b_key].float()
        rank = max(1, a.shape[0])
        scaling = (float(lora_config.alpha) / rank) if lora_config is not None else 1.0
        delta = b @ a
        merged[f"{prefix}.weight"] = (weight.float() + delta * scaling).to(dtype=weight.dtype)
        consumed.update({key, a_key, b_key})
        adapter[a_key] = state_dict[a_key].detach().cpu()
        adapter[b_key] = state_dict[b_key].detach().cpu()
        if bias_key in state_dict:
            merged[f"{prefix}.bias"] = state_dict[bias_key]
            consumed.add(bias_key)

    for key, value in state_dict.items():
        if key in consumed:
            continue
        if ".base_layer." in key or ".lora_A." in key or ".lora_B." in key:
            continue
        merged[key] = value

    return merged, adapter


def merge_lora_model_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    lora_config: LoRAConfig | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Merge LoRA tensors using module scaling and restore original key names."""
    scaling_by_name = {
        name: module.scaling
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }
    if not scaling_by_name and lora_config is None:
        return state_dict, {}

    merged: dict[str, torch.Tensor] = {}
    adapter: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    suffix = ".base_layer.weight"
    for base_w_key, weight in state_dict.items():
        if not base_w_key.endswith(suffix):
            continue
        name = base_w_key[: -len(suffix)]
        base_b_key = f"{name}.base_layer.bias"
        a_key = f"{name}.lora_A.weight"
        b_key = f"{name}.lora_B.weight"
        if base_w_key not in state_dict or a_key not in state_dict or b_key not in state_dict:
            continue

        a = state_dict[a_key].float()
        b = state_dict[b_key].float()
        scaling = scaling_by_name.get(name)
        if scaling is None and lora_config is not None:
            scaling = float(lora_config.alpha) / max(1, int(lora_config.rank))
        if scaling is None:
            raise KeyError(f"Cannot determine LoRA scaling for {name}")
        delta = (b @ a).reshape_as(weight.float()) * scaling
        merged[f"{name}.weight"] = (weight.float() + delta).to(dtype=weight.dtype)
        adapter[a_key] = state_dict[a_key].detach().cpu()
        adapter[b_key] = state_dict[b_key].detach().cpu()
        consumed.update({base_w_key, a_key, b_key})
        if base_b_key in state_dict:
            merged[f"{name}.bias"] = state_dict[base_b_key]
            consumed.add(base_b_key)

    for key, value in state_dict.items():
        if key in consumed:
            continue
        if ".base_layer." in key or ".lora_A." in key or ".lora_B." in key:
            continue
        merged[key] = value

    return merged, adapter
