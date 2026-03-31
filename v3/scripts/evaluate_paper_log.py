"""Evaluate logged paper decisions against future realized returns."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v2.src.data.store import DataStore
from v3.src.server.paper_metrics import evaluate_decisions, evaluate_round_trip_trades, flatten_decisions, load_cycle_log
from v3.src.strategy.hybrid_stack import load_v2_runtime_config, write_summary_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", default="v3/data/paper/paper_cycles.jsonl")
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--out", default="v3/data/paper/paper_cycle_evaluation.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cycles = load_cycle_log(args.log_path)
    decisions = flatten_decisions(cycles)

    pairs = sorted({decision["pair"].replace("/USD", "/USDT") for decision in decisions if "pair" in decision})
    config = load_v2_runtime_config()
    store = DataStore(config)
    try:
        price_frames = {pair: store.load_ohlcv(pair, "1h", last_n_days=120) for pair in pairs}
    finally:
        store.close()

    summary = evaluate_decisions(decisions, price_frames, horizon_hours=args.horizon_hours)
    summary["round_trip_evaluation"] = evaluate_round_trip_trades(decisions, price_frames)
    summary["source_log_path"] = args.log_path
    summary["decision_count"] = len(decisions)
    out_path = Path(args.out)
    write_summary_json(summary, out_path)

    print(f"Decision count: {len(decisions)}")
    print(f"Matured trade decisions: {summary['matured_trade_decisions']}")
    print(f"Overall: {summary['overall']}")
    print(f"Round trips: {summary['round_trip_evaluation']['closed_trades']}")
    print(f"Saved summary to {out_path.resolve()}")


if __name__ == "__main__":
    main()
