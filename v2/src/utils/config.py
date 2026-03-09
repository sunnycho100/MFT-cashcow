"""Config loader."""

from pathlib import Path
import yaml


def load_config(path: str = "v2/config.yaml") -> dict:
    """Load YAML config from *path*, searching upward if needed."""
    p = Path(path)
    if not p.exists():
        # Try from repo root
        alt = Path(__file__).resolve().parents[3] / path
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(f"Config not found at {path} or {alt}")
    with open(p) as f:
        return yaml.safe_load(f)
