"""Walk-forward comparison for base, funding, and funding+premium hybrid variants."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v3.src.strategy.hybrid_stack import (
    build_funding_overlay_frames,
    build_funding_premium_overlay_frames,
    build_walkforward_windows,
    choose_champion_variant,
    evaluate_variant_walkforward,
    load_base_feature_frames,
    load_v2_runtime_config,
    write_summary_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=1095)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--step-days", type=int, default=60)
    parser.add_argument("--chunk-days", type=int, default=31)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_v2_runtime_config()
    base_frames = load_base_feature_frames(config, days=args.days)
    funding_frames, funding_coverage = build_funding_overlay_frames(base_frames, chunk_days=args.chunk_days)
    funding_premium_frames, overlay_coverage = build_funding_premium_overlay_frames(
        base_frames,
        chunk_days=args.chunk_days,
        timeframe="1h",
    )

    train_bars = args.train_days * 24
    test_bars = args.test_days * 24
    step_bars = args.step_days * 24
    windows = build_walkforward_windows(base_frames, train_bars=train_bars, test_bars=test_bars, step_bars=step_bars)

    base_summary = evaluate_variant_walkforward(config, base_frames, windows)
    funding_summary = evaluate_variant_walkforward(config, funding_frames, windows)
    funding_premium_summary = evaluate_variant_walkforward(config, funding_premium_frames, windows)

    summary = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "settings": {
            "days": args.days,
            "train_days": args.train_days,
            "test_days": args.test_days,
            "step_days": args.step_days,
            "chunk_days": args.chunk_days,
            "window_count": len(windows),
        },
        "coverage": {
            "funding": funding_coverage,
            "funding_premium": overlay_coverage,
        },
        "variants": {
            "base": base_summary,
            "funding": funding_summary,
            "funding_premium": funding_premium_summary,
        },
    }
    summary["champion_variant"] = choose_champion_variant(summary)

    out_path = Path("v3/data/walkforward/hybrid_overlay_walkforward_summary.json")
    write_summary_json(summary, out_path)

    for name, row in summary["variants"].items():
        print(
            f"{name:>16}: avg_monthly={row['avg_monthly_return_pct']:+.2f}% "
            f"avg_auc={row['avg_auc']:.4f} "
            f"positive_windows={row['positive_windows']}/{row['window_count']} "
            f"best={row['best_window_return_pct']:+.2f}% "
            f"worst={row['worst_window_return_pct']:+.2f}%"
        )
    print(f"\nChampion variant: {summary['champion_variant']}")
    print(f"Saved summary to {out_path.resolve()}")


if __name__ == "__main__":
    main()
