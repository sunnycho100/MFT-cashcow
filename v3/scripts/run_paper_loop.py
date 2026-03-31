"""Run the v3 paper cycle continuously at a fixed interval."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v3.src.server.paper_runtime import IntegratedPaperRuntime
from v3.src.utils.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=0, help="0 means run forever")
    parser.add_argument("--sleep-sec", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config()
    runtime = IntegratedPaperRuntime(config)
    sleep_sec = int(args.sleep_sec or config.get("paper", {}).get("loop_interval_sec", 300))

    cycles = 0
    while True:
        records = runtime.run_once()
        cycles += 1
        print(f"cycle={cycles} decisions={len(records)}")

        if args.iterations and cycles >= args.iterations:
            break
        time.sleep(sleep_sec)


if __name__ == "__main__":
    main()
