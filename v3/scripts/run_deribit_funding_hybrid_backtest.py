"""Compare the positive v2 hybrid system with and without Deribit funding overlays."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

import polars as pl

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v2.src.data.store import DataStore
from v2.src.features.pipeline import build_features, get_feature_names
from v2.src.models.lgbm_model import LGBMModel
from v2.src.utils.config import load_config as load_v2_config
from v2.src.validation.trend_backtest import run_trend_backtest
from v3.src.data.deribit import DeribitClient, fetch_funding_history_paginated, write_funding_csv, write_metadata_json
from v3.src.data.deribit_overlay import add_cross_pair_funding_features, join_pair_funding, load_funding_csv, summarize_funding_history
from v3.src.utils.logger import get_logger

logger = get_logger("v3.scripts.run_deribit_funding_hybrid_backtest")
REPO_ROOT = Path(__file__).resolve().parents[2]

PAIR_TO_INSTRUMENT = {
    "BTC/USDT": ("btc", "BTC-PERPETUAL"),
    "ETH/USDT": ("eth", "ETH-PERPETUAL"),
    "SOL/USDT": ("sol", "SOL_USDC-PERPETUAL"),
}

BEST_CONFIG = {
    "entry_period": 480,
    "exit_period": 240,
    "atr_period": 48,
    "atr_stop_mult": 4.0,
    "risk_per_trade": 0.05,
    "max_position_pct": 0.60,
    "ml_filter": True,
    "ml_long_threshold": 0.30,
    "ml_short_threshold": 0.30,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=1095)
    parser.add_argument("--chunk-days", type=int, default=31)
    return parser


def normalize_v2_paths(config: dict) -> dict:
    data_cfg = config.setdefault("data", {})
    for key in ["db_path", "storage_path"]:
        value = data_cfg.get(key)
        if isinstance(value, str) and value.startswith("./"):
            data_cfg[key] = str((REPO_ROOT / "v2" / value[2:]).resolve())
    return config


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def ensure_funding_cache(
    pair_name: str,
    instrument_name: str,
    start_ms: int,
    end_ms: int,
    chunk_days: int,
) -> tuple[pl.DataFrame, dict]:
    out_dir = REPO_ROOT / "v3/data/deribit"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = instrument_name.replace("/", "_")
    csv_path = out_dir / f"funding_{safe_name}.csv"
    meta_path = out_dir / f"funding_{safe_name}.metadata.json"

    if csv_path.exists():
        frame = load_funding_csv(csv_path)
        summary = summarize_funding_history(frame)
        if summary["rows"] > 0 and summary["first_ts"] <= start_ms and summary["last_ts"] >= end_ms - 3_600_000:
            return frame, summary

    client = DeribitClient()
    rows = fetch_funding_history_paginated(
        client=client,
        instrument_name=instrument_name,
        start_timestamp_ms=start_ms,
        end_timestamp_ms=end_ms,
        chunk_days=chunk_days,
    )
    write_funding_csv(rows, csv_path)
    metadata = {
        "pair": pair_name,
        "instrument_name": instrument_name,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "rows": len(rows),
        "fetched_at_utc": datetime.now(tz=UTC).isoformat(),
    }
    write_metadata_json(metadata, meta_path)
    frame = load_funding_csv(csv_path)
    return frame, summarize_funding_history(frame)


def load_pair_frames(store: DataStore, pairs: list[str], days: int) -> dict[str, pl.DataFrame]:
    pair_frames = {}
    for pair in pairs:
        raw = store.load_ohlcv(pair, "1h", last_n_days=days)
        if len(raw) > 500:
            pair_frames[pair] = build_features(raw)
    return pair_frames


def make_integrated_frames(
    base_frames: dict[str, pl.DataFrame],
    funding_frames: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    integrated = {}
    aliases = [PAIR_TO_INSTRUMENT[pair][0] for pair in funding_frames.keys()]
    for pair_name, frame in base_frames.items():
        joined = frame
        for funding_pair_name, funding_frame in funding_frames.items():
            alias = PAIR_TO_INSTRUMENT[funding_pair_name][0]
            joined = join_pair_funding(joined, funding_frame, alias)
        joined = add_cross_pair_funding_features(joined, aliases)
        integrated[pair_name] = joined
    return integrated


def train_model(config: dict, pair_frames: dict[str, pl.DataFrame]):
    names = get_feature_names(next(iter(pair_frames.values())))
    model = LGBMModel(config)
    train_parts = []
    test_parts = []
    for frame in pair_frames.values():
        labeled = model.create_labels(frame)
        split_idx = int(len(labeled) * 0.8)
        train_parts.append(labeled[:split_idx])
        test_parts.append(labeled[split_idx:])
    metrics = model.train(pl.concat(train_parts), pl.concat(test_parts), names)
    return model, names, metrics


def run_hybrid_portfolio(pair_frames: dict[str, pl.DataFrame], model: LGBMModel, config_params: dict) -> dict:
    pair_results = {}
    portfolio_return = 0.0
    for pair_name, frame in pair_frames.items():
        test_start = int(len(frame) * 0.8)
        test_df = model.predict(frame[test_start:])
        result = run_trend_backtest(test_df, **config_params)
        pair_results[pair_name] = result["metrics"]
        portfolio_return += result["metrics"]["total_return_pct"]
    first_frame = next(iter(pair_frames.values()))
    test_len = len(first_frame) - int(len(first_frame) * 0.8)
    months = test_len / (24 * 30.44)
    return {
        "pair_results": pair_results,
        "portfolio_return_pct": round(portfolio_return, 2),
        "monthly_return_pct": round(portfolio_return / months, 2),
        "test_months": round(months, 2),
    }


def print_result(title: str, metrics: dict, portfolio: dict, feature_count: int) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    print(
        f"Features: {feature_count} | AUC: {metrics['auc']:.4f} | "
        f"Portfolio return: {portfolio['portfolio_return_pct']:+.2f}% | "
        f"Monthly: {portfolio['monthly_return_pct']:+.2f}%"
    )
    for pair_name, row in portfolio["pair_results"].items():
        print(
            f"  {pair_name}: return={row['total_return_pct']:+.2f}% "
            f"trades={row['total_trades']} sharpe={row['sharpe_ratio']:.2f} "
            f"maxdd={row['max_drawdown_pct']:.2f}%"
        )


def main() -> None:
    args = build_parser().parse_args()
    config = normalize_v2_paths(load_v2_config(str((REPO_ROOT / "v2/config.yaml").resolve())))

    store = DataStore(config)
    try:
        pairs = config.get("trading", {}).get("pairs", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        base_frames = load_pair_frames(store, pairs, args.days)
        start_ms = min(int(frame["timestamp"].min().timestamp() * 1000) for frame in base_frames.values())
        end_ms = max(int(frame["timestamp"].max().timestamp() * 1000) for frame in base_frames.values()) + 3_600_000
    finally:
        store.close()

    funding_frames = {}
    coverage = []
    for pair_name in pairs:
        alias, instrument = PAIR_TO_INSTRUMENT[pair_name]
        frame, summary = ensure_funding_cache(pair_name, instrument, start_ms, end_ms, args.chunk_days)
        funding_frames[pair_name] = frame
        coverage.append({"pair": pair_name, "alias": alias, "instrument_name": instrument, **summary})

    integrated_frames = make_integrated_frames(base_frames, funding_frames)

    baseline_model, baseline_features, baseline_metrics = train_model(config, base_frames)
    integrated_model, integrated_features, integrated_metrics = train_model(config, integrated_frames)

    baseline_portfolio = run_hybrid_portfolio(base_frames, baseline_model, BEST_CONFIG)
    integrated_portfolio = run_hybrid_portfolio(integrated_frames, integrated_model, BEST_CONFIG)

    print("Deribit funding coverage:")
    for row in coverage:
        print(f"  {row['pair']}: {row['instrument_name']} rows={row['rows']}")

    print_result("Baseline Hybrid Trend+ML", baseline_metrics, baseline_portfolio, len(baseline_features))
    print_result("Integrated Hybrid + Deribit Funding", integrated_metrics, integrated_portfolio, len(integrated_features))

    delta = integrated_portfolio["portfolio_return_pct"] - baseline_portfolio["portfolio_return_pct"]
    print(f"\nDelta return: {delta:+.2f}%")

    summary = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "coverage": coverage,
        "baseline": {
            "feature_count": len(baseline_features),
            "train_metrics": baseline_metrics,
            "portfolio": baseline_portfolio,
        },
        "integrated": {
            "feature_count": len(integrated_features),
            "train_metrics": integrated_metrics,
            "portfolio": integrated_portfolio,
        },
        "delta_return_pct": round(delta, 2),
    }
    out_path = REPO_ROOT / "v3/data/deribit/funding_hybrid_backtest_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(to_jsonable(summary), handle, indent=2)
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
