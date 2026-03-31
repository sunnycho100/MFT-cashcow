"""Run the standard v3 evaluation stack: hybrid walk-forward + return-max (+ optional stress).

Usage:
  python3 v3/scripts/run_evaluation_suite.py
  python3 v3/scripts/run_evaluation_suite.py --skip-hybrid
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-hybrid", action="store_true", help="Skip run_walkforward_hybrid_comparison.py")
    parser.add_argument("--skip-return-max", action="store_true", help="Skip return-max walk-forward")
    parser.add_argument("--skip-stress", action="store_true", help="Skip execution stress pass on return-max")
    args = parser.parse_args()

    py = sys.executable
    code = 0

    if not args.skip_hybrid:
        code = run(
            [
                py,
                str(REPO_ROOT / "v3/scripts/run_walkforward_hybrid_comparison.py"),
                "--days",
                "1095",
                "--train-days",
                "365",
                "--test-days",
                "60",
                "--step-days",
                "60",
            ]
        )
        if code != 0:
            return code

    if not args.skip_return_max:
        rm_cmd = [
            py,
            str(REPO_ROOT / "v3/scripts/run_walkforward_return_max.py"),
        ]
        if not args.skip_stress:
            rm_cmd.append("--stress-execution")
        code = run(rm_cmd)
        if code != 0:
            return code

    print("\nEvaluation suite finished.")
    print("Artifacts:")
    print(f"  - {REPO_ROOT / 'v3/data/walkforward/hybrid_overlay_walkforward_summary.json'}")
    print(f"  - {REPO_ROOT / 'v3/data/walkforward/return_max_walkforward_summary.json'}")
    if not args.skip_stress and not args.skip_return_max:
        print(f"  - {REPO_ROOT / 'v3/data/walkforward/return_max_walkforward_execution_stress.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
