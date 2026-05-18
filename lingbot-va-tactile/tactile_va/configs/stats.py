import json
from pathlib import Path


def apply_stats_json(config):
    stats_path = getattr(config, "stats_json_path", None)
    if not stats_path:
        return config

    path = Path(stats_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"stats_json_path does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)

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
    return config

