"""Validate env + config, then run a short paper loop to prove the server path.

Use on a fresh host after copying secrets (see deploy/README.md).

Runs N successful `IntegratedPaperRuntime.run_once()` calls (fail-fast on error).

Example:
  cd /path/to/MFT-cashcow/repo/root
  source .venv/bin/activate
  set -a && source /etc/mft-cashcow.env && set +a
  python3 v3/scripts/smoke_paper_deploy.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require_env(names: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for name in names:
        val = os.environ.get(name, "").strip()
        if not val:
            missing.append(name)
    return missing


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Successful paper cycles to run (default: 3)",
    )
    p.add_argument(
        "--skip-env-check",
        action="store_true",
        help="Do not require Kraken env vars (only for local debugging)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    from v3.src.server.paper_runtime import IntegratedPaperRuntime
    from v3.src.utils.config import load_config
    from v3.src.utils.logger import get_logger

    logger = get_logger("v3.scripts.smoke_paper_deploy")
    root = _repo_root()
    os.chdir(root)

    if not (root / "v3" / "config.yaml").is_file():
        print(f"error: expected v3/config.yaml under repo root: {root}", file=sys.stderr)
        sys.exit(2)

    config = load_config(str(root / "v3" / "config.yaml"))
    kraken = config.get("kraken", {})
    key_env = str(kraken.get("api_key_env", "KRAKEN_API_KEY"))
    secret_env = str(kraken.get("api_secret_env", "KRAKEN_API_SECRET"))

    if not args.skip_env_check:
        missing = _require_env((key_env, secret_env))
        if missing:
            print(
                "error: missing or empty environment variables (set these before running):\n  "
                + "\n  ".join(missing),
                file=sys.stderr,
            )
            sys.exit(2)
        print(f"ok: {key_env} and {secret_env} are set")

    runtime = IntegratedPaperRuntime(config)
    for i in range(args.iterations):
        records = runtime.run_once()
        logger.info("smoke cycle={} decisions={}", i + 1, len(records))

    cycle_log = Path(
        config.get("paper", {}).get("cycle_log_path", "v3/data/paper/paper_cycles.jsonl")
    )
    artifact = Path(
        config.get("paper", {}).get("artifact_path", "v3/data/paper/latest_paper_cycle.json")
    )
    if not cycle_log.is_file() or cycle_log.stat().st_size == 0:
        print(
            f"error: expected non-empty cycle log at {cycle_log}",
            file=sys.stderr,
        )
        sys.exit(1)
    if not artifact.is_file():
        print(f"error: expected artifact at {artifact}", file=sys.stderr)
        sys.exit(1)

    print(f"ok: smoke finished; see {cycle_log} and {artifact}")


if __name__ == "__main__":
    main()
