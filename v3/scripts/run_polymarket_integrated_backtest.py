"""Train and backtest a candle+Polymarket integrated model."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

import numpy as np
import polars as pl

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v2.src.data.store import DataStore
from v2.src.features.pipeline import build_features
from v2.src.models.lgbm_model import LGBMModel
from v2.src.utils.config import load_config as load_v2_config
from v2.src.validation.backtest import run_backtest
from v3.src.data.polymarket import (
    PolymarketClient,
    download_market_history,
    make_metadata,
    write_history_csv,
    write_metadata_json,
)
from v3.src.data.polymarket_overlay import (
    OverlayMarketSpec,
    join_overlay_markets,
    load_history_csv,
    summarize_history,
)
from v3.src.utils.config import load_config as load_v3_config
from v3.src.utils.logger import get_logger

logger = get_logger("v3.scripts.run_polymarket_integrated_backtest")
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MARKETS = [
    OverlayMarketSpec(alias="mstr_sell_mar26", market_slug="microstrategy-sells-any-bitcoin-by-march-31-2026"),
    OverlayMarketSpec(alias="mstr_sell_jun26", market_slug="microstrategy-sells-any-bitcoin-by-june-30-2026"),
    OverlayMarketSpec(alias="mstr_sell_dec26", market_slug="microstrategy-sells-any-bitcoin-by-december-31-2026"),
    OverlayMarketSpec(alias="kraken_ipo_mar26", market_slug="kraken-ipo-by-march-31-2026"),
    OverlayMarketSpec(alias="kraken_ipo_dec26", market_slug="kraken-ipo-by-december-31-2026-513"),
]

SKIP_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "atr_14",
    "target",
    "pred_class",
    "pred_long_prob",
    "pred_prob_up",
    "pred_prob_down",
    "pred_prob_flat",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days",
        type=int,
        default=1095,
        help="How many days of candles to load from the local DuckDB store",
    )
    parser.add_argument(
        "--min-market-points",
        type=int,
        default=100,
        help="Minimum raw Polymarket points required to keep a market in the overlay basket",
    )
    parser.add_argument(
        "--fidelity",
        type=int,
        default=60,
        help="Polymarket history fidelity in minutes when downloading raw history",
    )
    parser.add_argument(
        "--min-pair-rows",
        type=int,
        default=120,
        help="Minimum rows required per pair after overlap filtering",
    )
    return parser


def normalize_v2_paths(config: dict) -> dict:
    """Resolve relative v2 paths from the repository root."""
    data_cfg = config.setdefault("data", {})
    for key in ["db_path", "storage_path"]:
        value = data_cfg.get(key)
        if isinstance(value, str) and value.startswith("./"):
            data_cfg[key] = str((REPO_ROOT / "v2" / value[2:]).resolve())
    return config


def get_feature_names(frame: pl.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in SKIP_COLUMNS]


def ensure_history_file(
    client: PolymarketClient,
    spec: OverlayMarketSpec,
    output_dir: Path,
    fidelity: int,
) -> Path:
    """Fetch and cache one market history if we do not already have a non-empty file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"market_{spec.alias}.csv"
    metadata_path = output_dir / f"market_{spec.alias}.metadata.json"

    if csv_path.exists():
        try:
            current = load_history_csv(csv_path)
            if not current.is_empty():
                return csv_path
        except Exception:
            pass

    market, rows = download_market_history(
        client=client,
        market_slug=spec.market_slug,
        interval="max",
        fidelity=fidelity,
        allowed_outcomes={spec.outcome_label.casefold()},
    )
    write_history_csv(rows, csv_path)
    metadata = make_metadata(
        scope="market",
        interval="max",
        fidelity=fidelity,
        start_ts=None,
        end_ts=None,
        requested_outcomes=[spec.outcome_label],
        row_count=len(rows),
        subject=market,
    )
    write_metadata_json(metadata, metadata_path)
    return csv_path


def load_overlay_histories(v3_config: dict, fidelity: int, min_market_points: int) -> tuple[dict[str, pl.DataFrame], list[dict]]:
    """Download/load all configured markets and keep only those with enough data."""
    settings = v3_config.get("polymarket", {})
    client = PolymarketClient(
        gamma_base_url=settings.get("gamma_base_url", "https://gamma-api.polymarket.com"),
        clob_base_url=settings.get("clob_base_url", "https://clob.polymarket.com"),
        user_agent=settings.get("user_agent", "MFT-Cashcow-v3/0.1"),
        timeout_sec=settings.get("timeout_sec", 30),
    )
    output_dir = Path(settings.get("output_dir", "v3/data/polymarket")) / "overlay_cache"

    histories: dict[str, pl.DataFrame] = {}
    coverage_rows: list[dict] = []

    for spec in DEFAULT_MARKETS:
        csv_path = ensure_history_file(client, spec, output_dir, fidelity=fidelity)
        history = load_history_csv(csv_path)
        summary = summarize_history(history)
        coverage_rows.append(
            {
                "alias": spec.alias,
                "market_slug": spec.market_slug,
                "rows": summary["rows"],
                "first_ts": summary["first_ts"],
                "last_ts": summary["last_ts"],
                "path": str(csv_path),
            }
        )
        if summary["rows"] >= min_market_points:
            histories[spec.alias] = history
        else:
            logger.warning(
                "Dropping market={} because it only has {} points",
                spec.market_slug,
                summary["rows"],
            )

    return histories, coverage_rows


def compute_overlap_start(histories: dict[str, pl.DataFrame]) -> int:
    """Use the earliest available overlay timestamp across the retained markets."""
    starts = [int(frame["timestamp"].min()) for frame in histories.values() if not frame.is_empty()]
    if not starts:
        raise RuntimeError("No usable Polymarket histories were loaded")
    return min(starts)


def prepare_pair_data(
    store: DataStore,
    pairs: list[str],
    last_n_days: int,
    histories: dict[str, pl.DataFrame] | None,
    overlap_start_ts: int | None,
    min_pair_rows: int,
) -> dict[str, pl.DataFrame]:
    """Load candles, build v2 features, optionally join Polymarket overlays."""
    pair_frames: dict[str, pl.DataFrame] = {}

    for pair in pairs:
        raw = store.load_ohlcv(pair, "1h", last_n_days=last_n_days)
        if len(raw) <= 500:
            logger.warning("Skipping pair={} because only {} rows were available", pair, len(raw))
            continue

        feat = build_features(raw)
        if histories:
            feat = join_overlay_markets(feat, histories)
        if overlap_start_ts is not None:
            feat = feat.filter(pl.col("timestamp").dt.epoch(time_unit="s") >= overlap_start_ts)
        if len(feat) >= min_pair_rows:
            pair_frames[pair] = feat
        else:
            logger.warning(
                "Skipping pair={} after filtering because only {} feature rows remained",
                pair,
                len(feat),
            )

    return pair_frames


def train_and_backtest(config: dict, pair_frames: dict[str, pl.DataFrame], feature_names: list[str]) -> dict:
    """Run the existing v2 train/predict/backtest flow on the provided features."""
    model = LGBMModel(config)

    train_parts = []
    test_parts = {}
    for pair_name, frame in pair_frames.items():
        labeled = model.create_labels(frame)
        split_idx = int(len(labeled) * 0.8)
        train_parts.append(labeled[:split_idx])
        test_parts[pair_name] = labeled[split_idx:]

    combined_train = pl.concat(train_parts)
    combined_test = pl.concat(list(test_parts.values()))
    metrics = model.train(combined_train, combined_test, feature_names)

    bt_cfg = config.get("backtest", {})
    risk_cfg = config.get("risk", {})
    confidence_threshold = float(metrics.get("optimal_threshold", bt_cfg.get("confidence_threshold", 0.45)))
    pair_results = {}
    portfolio_returns = []

    for pair_name, frame in pair_frames.items():
        predicted = model.predict(frame)
        test_start = int(len(predicted) * 0.8)
        test_df = predicted[test_start:]
        result = run_backtest(
            test_df,
            confidence_threshold=confidence_threshold,
            tp_atr_mult=bt_cfg.get("tp_atr_mult", 2.5),
            sl_atr_mult=bt_cfg.get("sl_atr_mult", 1.0),
            max_hold_bars=bt_cfg.get("max_hold_bars", 24),
            trailing_activate_atr=bt_cfg.get("trailing_activate_atr", 1.5),
            trailing_distance_atr=bt_cfg.get("trailing_distance_atr", 0.8),
            kelly_fraction=risk_cfg.get("kelly_fraction", 0.50),
            long_only=bt_cfg.get("long_only", True),
            max_position_pct=bt_cfg.get("max_position_pct", risk_cfg.get("max_position_pct", 0.30)),
        )
        pair_results[pair_name] = result["metrics"]
        portfolio_returns.append(result["metrics"]["total_return_pct"])

    compound_return = 1.0
    for value in portfolio_returns:
        compound_return *= (1 + value / 100.0)
    compound_return = (compound_return - 1) * 100

    first_pair = next(iter(pair_frames.values()))
    test_len = len(first_pair) - int(len(first_pair) * 0.8)
    months = test_len / (24 * 30.44)

    return {
        "train_metrics": metrics,
        "pair_results": pair_results,
        "compound_return_pct": round(compound_return, 2),
        "avg_pair_return_pct": round(sum(portfolio_returns) / max(len(portfolio_returns), 1), 2),
        "test_bars": test_len,
        "test_months": round(months, 2),
        "feature_count": len(feature_names),
        "confidence_threshold": round(confidence_threshold, 3),
        "top_features": model.feature_importance(20),
    }


def print_experiment_result(name: str, result: dict) -> None:
    print(f"\n{name}")
    print("=" * len(name))
    print(
        f"Features: {result['feature_count']} | "
        f"Combined return: {result['compound_return_pct']:+.2f}% | "
        f"Avg pair return: {result['avg_pair_return_pct']:+.2f}% | "
        f"Test months: {result['test_months']:.2f}"
    )
    print(
        f"Train AUC: {result['train_metrics']['auc']:.4f} | "
        f"Accuracy: {result['train_metrics']['accuracy']:.4f} | "
        f"Best iter: {result['train_metrics']['best_iteration']} | "
        f"BT threshold: {result['confidence_threshold']:.3f}"
    )
    print("Pair results:")
    for pair_name, metrics in result["pair_results"].items():
        print(
            f"  {pair_name}: return={metrics['total_return_pct']:+.2f}% "
            f"trades={metrics['total_trades']} sharpe={metrics['sharpe_ratio']:.2f} "
            f"maxdd={metrics['max_drawdown_pct']:.2f}%"
        )
    print("Top features:")
    for feature_name, importance in result["top_features"][:10]:
        print(f"  {feature_name:<35} {importance:.0f}")


def to_jsonable(value):
    """Recursively convert numpy / non-JSON types into plain Python values."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def main() -> None:
    args = build_parser().parse_args()
    v2_config = normalize_v2_paths(load_v2_config(str((REPO_ROOT / "v2/config.yaml").resolve())))
    v3_config = load_v3_config(str((REPO_ROOT / "v3/config.yaml").resolve()))

    histories, coverage = load_overlay_histories(v3_config, fidelity=args.fidelity, min_market_points=args.min_market_points)
    if not histories:
        raise RuntimeError("No Polymarket markets passed the minimum history threshold")

    overlap_start_ts = compute_overlap_start(histories)
    print("Retained Polymarket markets:")
    for row in coverage:
        status = "KEEP" if row["alias"] in histories else "DROP"
        print(
            f"  {status:4s} {row['alias']:<18} rows={row['rows']:<6d} "
            f"slug={row['market_slug']}"
        )

    store = DataStore(v2_config)
    try:
        pairs = v2_config.get("trading", {}).get("pairs", ["BTC/USDT"])

        baseline_frames = prepare_pair_data(
            store=store,
            pairs=pairs,
            last_n_days=args.days,
            histories=None,
            overlap_start_ts=overlap_start_ts,
            min_pair_rows=args.min_pair_rows,
        )
        integrated_frames = prepare_pair_data(
            store=store,
            pairs=pairs,
            last_n_days=args.days,
            histories=histories,
            overlap_start_ts=overlap_start_ts,
            min_pair_rows=args.min_pair_rows,
        )
    finally:
        store.close()

    baseline_feature_names = get_feature_names(next(iter(baseline_frames.values())))
    integrated_feature_names = get_feature_names(next(iter(integrated_frames.values())))

    baseline_result = train_and_backtest(v2_config, baseline_frames, baseline_feature_names)
    integrated_result = train_and_backtest(v2_config, integrated_frames, integrated_feature_names)

    print_experiment_result("Baseline Candle-Only Model", baseline_result)
    print_experiment_result("Integrated Candle + Polymarket Model", integrated_result)

    delta = integrated_result["compound_return_pct"] - baseline_result["compound_return_pct"]
    print("\nDelta")
    print("=====")
    print(f"Integrated minus baseline return: {delta:+.2f}%")
    print(f"Overlap start timestamp: {overlap_start_ts}")

    summary_path = REPO_ROOT / "v3/data/polymarket/integrated_backtest_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "overlap_start_ts": overlap_start_ts,
        "overlap_start_iso": datetime.fromtimestamp(overlap_start_ts, tz=UTC).isoformat(),
        "coverage": coverage,
        "baseline": baseline_result,
        "integrated": integrated_result,
        "delta_return_pct": round(delta, 2),
    }
    with open(summary_path, "w") as handle:
        json.dump(to_jsonable(summary), handle, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
