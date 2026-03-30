"""Config loader for v3."""

from pathlib import Path

import yaml


def load_config(path: str = "v3/config.yaml") -> dict:
    """Load YAML config from *path*, searching upward if needed."""
    config_path = Path(path)
    if not config_path.exists():
        alt = Path(__file__).resolve().parents[3] / path
        if alt.exists():
            config_path = alt
        else:
            raise FileNotFoundError(f"Config not found at {path} or {alt}")

    with open(config_path) as handle:
        return yaml.safe_load(handle)
