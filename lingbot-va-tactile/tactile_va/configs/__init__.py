from .robotwin_tactile_cfg import (
    robotwin_tactile_cfg,
    robotwin_tactile_lowmem_cfg,
    robotwin_tactile_train_cfg,
)

TACTILE_CONFIGS = {
    "robotwin_tactile": robotwin_tactile_cfg,
    "robotwin_tactile_lowmem": robotwin_tactile_lowmem_cfg,
    "robotwin_tactile_train": robotwin_tactile_train_cfg,
}

__all__ = [
    "TACTILE_CONFIGS",
    "robotwin_tactile_cfg",
    "robotwin_tactile_lowmem_cfg",
    "robotwin_tactile_train_cfg",
]
