"""Config loader for v3."""

from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:
    _load_dotenv = None


def _load_repo_dotenv() -> None:
    """Load repo-root `.env` so Kraken env vars match v3/config.yaml without `source`."""
    if _load_dotenv is None:
        return
    root = Path(__file__).resolve().parents[3]
    env_path = root / ".env"
    if env_path.is_file():
        _load_dotenv(env_path)


def load_config(path: str = "v3/config.yaml") -> dict:
    """Load YAML config from *path*, searching upward if needed."""
    _load_repo_dotenv()
    config_path = Path(path)
    if not config_path.exists():
        alt = Path(__file__).resolve().parents[3] / path
        if alt.exists():
            config_path = alt
        else:
            raise FileNotFoundError(f"Config not found at {path} or {alt}")

    with open(config_path) as handle:
        return yaml.safe_load(handle)
