import json
from pathlib import Path


def _validate_action_layout(config, stats):
    expected_layout = getattr(config, "action_layout_name", None)
    if expected_layout is None:
        return

    stats_layout = stats.get("action_layout_name")
    if stats_layout is not None and stats_layout != expected_layout:
        raise ValueError(
            f"stats action_layout_name={stats_layout!r} does not match "
            f"config action_layout_name={expected_layout!r}"
        )

    expected_ids = list(getattr(config, "used_action_channel_ids", []))
    stats_ids = stats.get("used_action_channel_ids")
    if expected_layout == "joint_gripper":
        joint_ids = list(range(14, 21)) + [28]
        action_dim = int(stats.get("action_dim", len(stats_ids or expected_ids)))
        ids_to_check = stats_ids if stats_ids is not None else expected_ids
        if action_dim != 8 or list(ids_to_check) != joint_ids:
            raise ValueError(
                "joint_gripper stats must have action_dim=8 and "
                f"used_action_channel_ids={joint_ids}; got action_dim={action_dim}, "
                f"used_action_channel_ids={ids_to_check}"
            )


def apply_stats_json(config):
    stats_path = getattr(config, "stats_json_path", None)
    if not stats_path:
        return config

    path = Path(stats_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"stats_json_path does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)

    _validate_action_layout(config, stats)

    if "norm_stat" in stats:
        config.norm_stat = stats["norm_stat"]
    if "tactile_norm_stat" in stats:
        config.tactile_norm_stat = stats["tactile_norm_stat"]
    if "used_action_channel_ids" in stats:
        config.used_action_channel_ids = stats["used_action_channel_ids"]
        inverse = [len(config.used_action_channel_ids)] * config.action_dim
        for i, channel_id in enumerate(config.used_action_channel_ids):
            inverse[channel_id] = i
        config.inverse_used_action_channel_ids = inverse
    if "tactile_dim" in stats:
        config.tactile_dim = int(stats["tactile_dim"])
    if "tactile_min_range" in stats:
        config.tactile_min_range = float(stats["tactile_min_range"])
    return config
