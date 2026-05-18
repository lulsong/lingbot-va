import os
import logging

import torch
import torch.nn as nn

from .tactile_model import TactileWanTransformer3DModel


def _copy_meta_params_from_matching_module(dst_module, src_module, device):
    copied = 0
    if dst_module is None or src_module is None:
        return copied

    src_modules = dict(src_module.named_modules())
    for module_name, dst_child in dst_module.named_modules():
        src_child = src_modules.get(module_name)
        if src_child is None:
            continue

        for param_name, dst_param in list(dst_child._parameters.items()):
            if dst_param is None or not getattr(dst_param, "is_meta", False):
                continue
            src_param = src_child._parameters.get(param_name)
            if src_param is None or getattr(src_param, "is_meta", False):
                continue
            if tuple(src_param.shape) != tuple(dst_param.shape):
                continue
            dst_child._parameters[param_name] = nn.Parameter(
                src_param.detach().clone().to(device=device, dtype=dst_param.dtype),
                requires_grad=dst_param.requires_grad,
            )
            copied += 1

        for buffer_name, dst_buffer in list(dst_child._buffers.items()):
            if dst_buffer is None or not getattr(dst_buffer, "is_meta", False):
                continue
            src_buffer = src_child._buffers.get(buffer_name)
            if src_buffer is None or getattr(src_buffer, "is_meta", False):
                continue
            if tuple(src_buffer.shape) != tuple(dst_buffer.shape):
                continue
            dst_child._buffers[buffer_name] = src_buffer.detach().clone().to(
                device=device,
                dtype=dst_buffer.dtype,
            )
            copied += 1

    return copied


def _materialize_remaining_meta_tensors(model, device):
    changed_modules = set()
    materialized = 0

    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is None or not getattr(param, "is_meta", False):
                continue
            tensor = torch.empty(tuple(param.shape), device=device, dtype=param.dtype)
            module._parameters[name] = nn.Parameter(tensor, requires_grad=param.requires_grad)
            changed_modules.add(module)
            materialized += 1

        for name, buffer in list(module._buffers.items()):
            if buffer is None or not getattr(buffer, "is_meta", False):
                continue
            module._buffers[name] = torch.zeros(tuple(buffer.shape), device=device, dtype=buffer.dtype)
            changed_modules.add(module)
            materialized += 1

    for module in changed_modules:
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
            continue

        for param in module.parameters(recurse=False):
            if param.ndim > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    return materialized


def _materialize_tactile_meta_tensors(model, device):
    copied = _copy_meta_params_from_matching_module(
        getattr(model, "condition_embedder_tactile", None),
        getattr(model, "condition_embedder_action", None),
        device,
    )
    materialized = _materialize_remaining_meta_tensors(model, device)
    if copied or materialized:
        logging.info(
            "Materialized tactile meta tensors: copied=%s randomly_initialized=%s",
            copied,
            materialized,
        )


def load_tactile_transformer(transformer_path, torch_dtype, torch_device, **kwargs):
    """Load a tactile transformer with partial checkpoint compatibility."""
    model = TactileWanTransformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
        **kwargs,
    )
    _materialize_tactile_meta_tensors(model, torch_device)
    return model.to(torch_device)


def save_tactile_transformer_config(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_config(output_dir)
