"""Shared helpers for the v3 hybrid backtests and paper runtime."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from v2.src.data.fetcher import DataFetcher
from v2.src.data.store import DataStore
from v2.src.features.pipeline import build_features, get_feature_names
from v2.src.models.lgbm_model import LGBMModel
from v2.src.utils.config import load_config as load_v2_config
from v2.src.validation.trend_backtest import run_trend_backtest
from v3.src.data.deribit import DeribitClient, DeribitFundingPoint, fetch_funding_history_paginated, write_funding_csv, write_metadata_json as write_deribit_metadata_json
from v3.src.data.deribit_overlay import add_cross_pair_funding_features, join_pair_funding, load_funding_csv, summarize_funding_history
from v3.src.data.premium_overlay import add_cross_pair_premium_features, join_pair_premium, summarize_premium_candles
from v3.src.data.venue_candles import VenueCandle, VenueCandleClient, load_candles_csv, summarize_candles, write_candles_csv, write_metadata_json as write_venue_metadata_json

REPO_ROOT = Path(__file__).resolve().parents[3]
HISTORY_TOLERANCE_MS = 3_600_000

PAIR_TO_DERIBIT_INSTRUMENT = {
    "BTC/USDT": ("btc", "BTC-PERPETUAL"),
    "ETH/USDT": ("eth", "ETH-PERPETUAL"),
    "SOL/USDT": ("sol", "SOL_USDC-PERPETUAL"),
}

PAIR_TO_COINBASE = {
    "BTC/USDT": ("btc", "BTC/USD"),
    "ETH/USDT": ("eth", "ETH/USD"),
    "SOL/USDT": ("sol", "SOL/USD"),
}

BEST_TREND_CONFIG = {
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


@dataclass(frozen=True)
class WindowSpec:
    index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_start_ts: str
    train_end_ts: str
    test_start_ts: str
    test_end_ts: str


def normalize_v2_paths(config: dict) -> dict:
    data_cfg = config.setdefault("data", {})
    for key in ["db_path", "storage_path"]:
        value = data_cfg.get(key)
        if isinstance(value, str) and value.startswith("./"):
            data_cfg[key] = str((REPO_ROOT / "v2" / value[2:]).resolve())
    return config


def load_v2_runtime_config() -> dict:
    return normalize_v2_paths(load_v2_config(str((REPO_ROOT / "v2/config.yaml").resolve())))


def align_pair_frames(pair_frames: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    min_len = min(len(frame) for frame in pair_frames.values())
    return {pair: frame.tail(min_len) for pair, frame in pair_frames.items()}


def load_base_feature_frames(config: dict, days: int) -> dict[str, pl.DataFrame]:
    return load_base_feature_frames_with_refresh(config, days=days, refresh_latest=False)


def refresh_latest_candles(config: dict, timeframe: str = "1h") -> dict[str, int]:
    pairs = config.get("trading", {}).get("pairs", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    store = DataStore(config)
    fetcher = DataFetcher(config)
    written = {}
    try:
        for pair in pairs:
            try:
                fresh = fetcher.fetch_incremental(pair, store, timeframe=timeframe)
                if len(fresh) > 0:
                    written[pair] = store.save_ohlcv(pair, timeframe, fresh)
                else:
                    written[pair] = 0
            except Exception:
                written[pair] = 0
        return written
    finally:
        store.close()


def load_latest_candle_timestamps(config: dict, timeframe: str = "1h") -> dict[str, str]:
    pairs = config.get("trading", {}).get("pairs", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    store = DataStore(config)
    try:
        latest = {}
        for pair in pairs:
            timestamp = store.get_latest_timestamp(pair, timeframe)
            if timestamp is not None:
                latest[pair] = str(timestamp)
        return latest
    finally:
        store.close()


def load_base_feature_frames_with_refresh(config: dict, days: int, refresh_latest: bool = False) -> dict[str, pl.DataFrame]:
    if refresh_latest:
        refresh_latest_candles(config, timeframe="1h")
    store = DataStore(config)
    try:
        pairs = config.get("trading", {}).get("pairs", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        pair_frames = {}
        for pair in pairs:
            raw = store.load_ohlcv(pair, "1h", last_n_days=days)
            if len(raw) > 500:
                pair_frames[pair] = build_features(raw)
        return align_pair_frames(pair_frames)
    finally:
        store.close()


def get_frame_bounds(pair_frames: dict[str, pl.DataFrame]) -> tuple[int, int]:
    start_ms = min(int(frame["timestamp"].min().timestamp() * 1000) for frame in pair_frames.values())
    end_ms = max(int(frame["timestamp"].max().timestamp() * 1000) for frame in pair_frames.values()) + 3_600_000
    return start_ms, end_ms


def _frame_to_funding_points(frame: pl.DataFrame, instrument_name: str) -> list[DeribitFundingPoint]:
    if frame.is_empty():
        return []
    points = []
    for row in frame.to_dicts():
        points.append(
            DeribitFundingPoint(
                instrument_name=instrument_name,
                timestamp_ms=int(row["timestamp_ms"]),
                index_price=float(row["index_price"]),
                prev_index_price=float(row["prev_index_price"]),
                interest_1h=float(row["interest_1h"]),
                interest_8h=float(row["interest_8h"]),
            )
        )
    return points


def _merge_funding_points(existing: list[DeribitFundingPoint], new_rows: list[DeribitFundingPoint]) -> list[DeribitFundingPoint]:
    dedup = {row.timestamp_ms: row for row in existing}
    for row in new_rows:
        dedup[row.timestamp_ms] = row
    return [dedup[key] for key in sorted(dedup)]


def _frame_to_venue_candles(frame: pl.DataFrame) -> list[VenueCandle]:
    if frame.is_empty():
        return []
    rows = []
    for row in frame.to_dicts():
        rows.append(
            VenueCandle(
                timestamp_ms=int(row["timestamp_ms"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
        )
    return rows


def _merge_venue_candles(existing: list[VenueCandle], new_rows: list[VenueCandle]) -> list[VenueCandle]:
    dedup = {row.timestamp_ms: row for row in existing}
    for row in new_rows:
        dedup[row.timestamp_ms] = row
    return [dedup[key] for key in sorted(dedup)]


def ensure_deribit_cache(pair_name: str, instrument_name: str, start_ms: int, end_ms: int, chunk_days: int) -> tuple[pl.DataFrame, dict]:
    out_dir = REPO_ROOT / "v3/data/deribit"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = instrument_name.replace("/", "_")
    csv_path = out_dir / f"funding_{safe_name}.csv"
    meta_path = out_dir / f"funding_{safe_name}.metadata.json"

    if csv_path.exists():
        frame = load_funding_csv(csv_path)
        summary = summarize_funding_history(frame)
        if (
            summary["rows"] > 0
            and summary["first_ts"] is not None
            and summary["first_ts"] <= start_ms + HISTORY_TOLERANCE_MS
            and summary["last_ts"] is not None
            and summary["last_ts"] >= end_ms - HISTORY_TOLERANCE_MS
        ):
            return frame, summary
        existing_rows = _frame_to_funding_points(frame, instrument_name)
    else:
        frame = pl.DataFrame()
        summary = {"rows": 0, "first_ts": None, "last_ts": None}
        existing_rows = []

    client = DeribitClient()
    fetched_rows = []
    if summary["rows"] == 0:
        fetched_rows.extend(
            fetch_funding_history_paginated(
                client=client,
                instrument_name=instrument_name,
                start_timestamp_ms=start_ms,
                end_timestamp_ms=end_ms,
                chunk_days=chunk_days,
            )
        )
    else:
        if summary["first_ts"] is not None and summary["first_ts"] > start_ms + HISTORY_TOLERANCE_MS:
            fetched_rows.extend(
                fetch_funding_history_paginated(
                    client=client,
                    instrument_name=instrument_name,
                    start_timestamp_ms=start_ms,
                    end_timestamp_ms=int(summary["first_ts"]),
                    chunk_days=chunk_days,
                )
            )
        if summary["last_ts"] is not None and summary["last_ts"] < end_ms - HISTORY_TOLERANCE_MS:
            fetched_rows.extend(
                fetch_funding_history_paginated(
                    client=client,
                    instrument_name=instrument_name,
                    start_timestamp_ms=int(summary["last_ts"]) + 3_600_000,
                    end_timestamp_ms=end_ms,
                    chunk_days=chunk_days,
                )
            )

    rows = _merge_funding_points(existing_rows, fetched_rows)
    write_funding_csv(rows, csv_path)
    metadata = {
        "pair": pair_name,
        "instrument_name": instrument_name,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "rows": len(rows),
        "fetched_at_utc": datetime.now(tz=UTC).isoformat(),
    }
    write_deribit_metadata_json(metadata, meta_path)
    merged = load_funding_csv(csv_path)
    return merged, summarize_funding_history(merged)


def ensure_coinbase_cache(pair_name: str, venue_pair: str, start_ms: int, end_ms: int, timeframe: str) -> tuple[pl.DataFrame, dict]:
    out_dir = REPO_ROOT / "v3/data/coinbase"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = venue_pair.replace("/", "_")
    csv_path = out_dir / f"ohlcv_{safe_name}_{timeframe}.csv"
    meta_path = out_dir / f"ohlcv_{safe_name}_{timeframe}.metadata.json"

    if csv_path.exists():
        frame = load_candles_csv(csv_path)
        summary = summarize_candles(frame)
        if (
            summary["rows"] > 0
            and summary["first_ts"] is not None
            and summary["first_ts"] <= start_ms + HISTORY_TOLERANCE_MS
            and summary["last_ts"] is not None
            and summary["last_ts"] >= end_ms - HISTORY_TOLERANCE_MS
        ):
            return frame, summary
        existing_rows = _frame_to_venue_candles(frame)
    else:
        frame = pl.DataFrame()
        summary = {"rows": 0, "first_ts": None, "last_ts": None}
        existing_rows = []

    client = VenueCandleClient(exchange_id="coinbase")
    fetched_rows = []
    if summary["rows"] == 0:
        fetched_rows.extend(
            client.fetch_ohlcv_history(
                pair=venue_pair,
                timeframe=timeframe,
                start_timestamp_ms=start_ms,
                end_timestamp_ms=end_ms,
            )
        )
    else:
        if summary["first_ts"] is not None and summary["first_ts"] > start_ms + HISTORY_TOLERANCE_MS:
            fetched_rows.extend(
                client.fetch_ohlcv_history(
                    pair=venue_pair,
                    timeframe=timeframe,
                    start_timestamp_ms=start_ms,
                    end_timestamp_ms=int(summary["first_ts"]),
                )
            )
        if summary["last_ts"] is not None and summary["last_ts"] < end_ms - HISTORY_TOLERANCE_MS:
            fetched_rows.extend(
                client.fetch_ohlcv_history(
                    pair=venue_pair,
                    timeframe=timeframe,
                    start_timestamp_ms=int(summary["last_ts"]) + 3_600_000,
                    end_timestamp_ms=end_ms,
                )
            )

    rows = _merge_venue_candles(existing_rows, fetched_rows)
    write_candles_csv(rows, csv_path)
    metadata = {
        "pair": pair_name,
        "venue_pair": venue_pair,
        "venue": "coinbase",
        "timeframe": timeframe,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "rows": len(rows),
        "fetched_at_utc": datetime.now(tz=UTC).isoformat(),
    }
    write_venue_metadata_json(metadata, meta_path)
    merged = load_candles_csv(csv_path)
    return merged, summarize_premium_candles(merged)


def build_funding_overlay_frames(base_frames: dict[str, pl.DataFrame], chunk_days: int = 31) -> tuple[dict[str, pl.DataFrame], list[dict[str, Any]]]:
    start_ms, end_ms = get_frame_bounds(base_frames)
    funding_frames = {}
    coverage = []
    for pair_name in base_frames:
        alias, instrument = PAIR_TO_DERIBIT_INSTRUMENT[pair_name]
        frame, summary = ensure_deribit_cache(pair_name, instrument, start_ms, end_ms, chunk_days)
        funding_frames[pair_name] = frame
        coverage.append({"pair": pair_name, "alias": alias, "instrument_name": instrument, **summary})

    aliases = [PAIR_TO_DERIBIT_INSTRUMENT[pair][0] for pair in funding_frames]
    integrated = {}
    for pair_name, frame in base_frames.items():
        joined = frame
        for funding_pair_name, funding_frame in funding_frames.items():
            alias = PAIR_TO_DERIBIT_INSTRUMENT[funding_pair_name][0]
            joined = join_pair_funding(joined, funding_frame, alias)
        joined = add_cross_pair_funding_features(joined, aliases)
        integrated[pair_name] = joined
    return align_pair_frames(integrated), coverage


def build_funding_premium_overlay_frames(
    base_frames: dict[str, pl.DataFrame],
    chunk_days: int = 31,
    timeframe: str = "1h",
) -> tuple[dict[str, pl.DataFrame], dict[str, list[dict[str, Any]]]]:
    funding_frames, funding_coverage = build_funding_overlay_frames(base_frames, chunk_days=chunk_days)
    start_ms, end_ms = get_frame_bounds(base_frames)

    premium_frames = {}
    premium_coverage = []
    for pair_name in base_frames:
        alias, venue_pair = PAIR_TO_COINBASE[pair_name]
        frame, summary = ensure_coinbase_cache(pair_name, venue_pair, start_ms, end_ms, timeframe)
        premium_frames[pair_name] = frame
        premium_coverage.append({"pair": pair_name, "alias": alias, "venue": "coinbase", "venue_pair": venue_pair, **summary})

    aliases = [PAIR_TO_COINBASE[pair][0] for pair in premium_frames]
    integrated = {}
    for pair_name, frame in funding_frames.items():
        joined = frame
        for premium_pair_name, premium_frame in premium_frames.items():
            alias = PAIR_TO_COINBASE[premium_pair_name][0]
            joined = join_pair_premium(joined, premium_frame, alias)
        joined = add_cross_pair_premium_features(joined, aliases)
        integrated[pair_name] = joined
    return align_pair_frames(integrated), {"funding": funding_coverage, "premium": premium_coverage}


def build_walkforward_windows(pair_frames: dict[str, pl.DataFrame], train_bars: int, test_bars: int, step_bars: int) -> list[WindowSpec]:
    reference = next(iter(pair_frames.values()))
    total = len(reference)
    windows = []
    cursor = train_bars
    idx = 0
    while cursor + test_bars <= total:
        train_start = cursor - train_bars
        train_end = cursor
        test_start = cursor
        test_end = cursor + test_bars
        windows.append(
            WindowSpec(
                index=idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_start_ts=str(reference["timestamp"][train_start]),
                train_end_ts=str(reference["timestamp"][train_end - 1]),
                test_start_ts=str(reference["timestamp"][test_start]),
                test_end_ts=str(reference["timestamp"][test_end - 1]),
            )
        )
        idx += 1
        cursor += step_bars
    return windows


def _slice_pair_frames(pair_frames: dict[str, pl.DataFrame], start: int, end: int) -> dict[str, pl.DataFrame]:
    return {pair: frame.slice(start, end - start) for pair, frame in pair_frames.items()}


def train_model_for_window(config: dict, pair_frames: dict[str, pl.DataFrame], window: WindowSpec) -> tuple[LGBMModel, list[str], dict]:
    names = get_feature_names(next(iter(pair_frames.values())))
    model = LGBMModel(config)
    train_parts = []
    test_parts = []
    for frame in pair_frames.values():
        train_slice = frame.slice(window.train_start, window.train_end - window.train_start)
        test_slice = frame.slice(window.test_start, window.test_end - window.test_start)
        train_parts.append(model.create_labels(train_slice))
        test_parts.append(model.create_labels(test_slice))
    metrics = model.train(pl.concat(train_parts), pl.concat(test_parts), names)
    return model, names, metrics


def run_portfolio_for_window(pair_frames: dict[str, pl.DataFrame], model: LGBMModel, window: WindowSpec, config_params: dict | None = None) -> dict:
    params = dict(BEST_TREND_CONFIG if config_params is None else config_params)
    pair_results = {}
    portfolio_return = 0.0
    for pair_name, frame in pair_frames.items():
        test_df = model.predict(frame.slice(window.test_start, window.test_end - window.test_start))
        result = run_trend_backtest(test_df, **params)
        pair_results[pair_name] = result["metrics"]
        portfolio_return += result["metrics"]["total_return_pct"]
    months = (window.test_end - window.test_start) / (24 * 30.44)
    return {
        "pair_results": pair_results,
        "portfolio_return_pct": round(portfolio_return, 2),
        "monthly_return_pct": round(portfolio_return / months, 2),
        "test_months": round(months, 2),
    }


def evaluate_variant_walkforward(config: dict, pair_frames: dict[str, pl.DataFrame], windows: list[WindowSpec]) -> dict[str, Any]:
    window_results = []
    for window in windows:
        model, names, metrics = train_model_for_window(config, pair_frames, window)
        portfolio = run_portfolio_for_window(pair_frames, model, window)
        window_results.append(
            {
                "window": asdict(window),
                "feature_count": len(names),
                "train_metrics": metrics,
                "portfolio": portfolio,
            }
        )

    portfolio_returns = [row["portfolio"]["portfolio_return_pct"] for row in window_results]
    monthly_returns = [row["portfolio"]["monthly_return_pct"] for row in window_results]
    aucs = [row["train_metrics"]["auc"] for row in window_results]
    positive = sum(1 for value in portfolio_returns if value > 0)

    return {
        "window_count": len(window_results),
        "positive_windows": positive,
        "positive_window_ratio": round(positive / max(len(window_results), 1), 3),
        "avg_portfolio_return_pct": round(sum(portfolio_returns) / max(len(portfolio_returns), 1), 2),
        "avg_monthly_return_pct": round(sum(monthly_returns) / max(len(monthly_returns), 1), 2),
        "avg_auc": round(sum(aucs) / max(len(aucs), 1), 4),
        "best_window_return_pct": round(max(portfolio_returns) if portfolio_returns else 0.0, 2),
        "worst_window_return_pct": round(min(portfolio_returns) if portfolio_returns else 0.0, 2),
        "total_window_return_sum_pct": round(sum(portfolio_returns), 2),
        "windows": window_results,
    }


def choose_champion_variant(summary: dict[str, Any]) -> str:
    candidates = []
    for variant in ["base", "funding", "funding_premium"]:
        row = summary["variants"].get(variant)
        if row is None:
            continue
        score = (
            row["avg_monthly_return_pct"],
            row["positive_window_ratio"],
            row["avg_auc"],
        )
        candidates.append((score, variant))
    return max(candidates)[1]


def to_jsonable(value: Any) -> Any:
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


def write_summary_json(summary: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(to_jsonable(summary), handle, indent=2)
    return path
