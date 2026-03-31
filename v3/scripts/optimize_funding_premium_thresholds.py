"""Optimize calibration method and execution thresholds for the funding_premium champion."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys

import polars as pl

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v2.src.features.pipeline import get_feature_names
from v2.src.models.lgbm_model import LGBMModel
from v2.src.validation.trend_backtest import run_trend_backtest
from v3.src.strategy.calibration import (
    apply_probability_calibration,
    fit_probability_calibrator,
    predict_frame_with_calibration,
    predict_raw_probabilities,
)
from v3.src.strategy.hybrid_stack import (
    BEST_TREND_CONFIG,
    build_funding_premium_overlay_frames,
    build_walkforward_windows,
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
    parser.add_argument("--long-thresholds", nargs="*", type=float, default=[0.30, 0.35, 0.40, 0.45, 0.50])
    parser.add_argument("--short-confidence-thresholds", nargs="*", type=float, default=[0.55, 0.60, 0.65, 0.70])
    parser.add_argument("--calibration-methods", nargs="*", default=["raw", "isotonic", "sigmoid"])
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    return parser


def detect_regime(row: dict) -> str:
    adx_56 = float(row.get("adx_56", 0.0))
    dist_ema_168 = float(row.get("dist_ema_168", 0.0))
    dist_ema_504 = float(row.get("dist_ema_504", 0.0))
    if adx_56 >= 18.0 and dist_ema_168 > 0 and dist_ema_504 > 0:
        return "bull"
    if adx_56 >= 18.0 and dist_ema_168 < 0 and dist_ema_504 < 0:
        return "bear"
    return "range"


def train_window_bundle(config: dict, pair_frames: dict[str, pl.DataFrame], window, calibration_fraction: float, calibration_method: str):
    names = get_feature_names(next(iter(pair_frames.values())))
    model = LGBMModel(config)

    train_parts = []
    calib_parts = []
    test_label_parts = []
    test_frames = {}

    for pair_name, frame in pair_frames.items():
        train_slice = frame.slice(window.train_start, window.train_end - window.train_start)
        test_slice = frame.slice(window.test_start, window.test_end - window.test_start)

        labeled_train = model.create_labels(train_slice)
        split_idx = int(len(labeled_train) * (1.0 - calibration_fraction))
        split_idx = min(max(split_idx, 1), len(labeled_train) - 1)

        train_parts.append(labeled_train[:split_idx])
        calib_parts.append(labeled_train[split_idx:])
        test_label_parts.append(model.create_labels(test_slice))
        test_frames[pair_name] = test_slice

    train_df = pl.concat(train_parts)
    calib_df = pl.concat(calib_parts)
    test_label_df = pl.concat(test_label_parts)

    train_metrics = model.train(train_df, calib_df, names)

    raw_calib = predict_raw_probabilities(model, calib_df, feature_names=names)
    calibration = fit_probability_calibrator(raw_calib, calib_df["target"].to_numpy(), method=calibration_method)

    raw_test = predict_raw_probabilities(model, test_label_df, feature_names=names)
    calibrated_test = apply_probability_calibration(raw_test, calibration)

    train_metrics["probability_test"] = {
        "raw_auc": calibration.metrics.get("raw_auc"),
        "raw_brier": calibration.metrics.get("raw_brier"),
        "calibrated_auc": calibration.metrics.get("calibrated_auc"),
        "calibrated_brier": calibration.metrics.get("calibrated_brier"),
    }

    predicted_frames = {
        pair_name: predict_frame_with_calibration(model, frame, calibration, feature_names=names)
        for pair_name, frame in test_frames.items()
    }

    decision_snapshot = {}
    for pair_name, frame in predicted_frames.items():
        latest = frame.tail(1).to_dicts()[0]
        decision_snapshot[pair_name] = {
            "signal_timestamp": str(latest["timestamp"]),
            "regime": detect_regime(latest),
            "edge_score": round(float(latest["pred_long_prob"]), 6),
        }

    return predicted_frames, calibration, train_metrics, decision_snapshot, calibrated_test


def summarize_combo(row: dict) -> dict:
    window_count = len(row["monthly_returns"])
    avg_monthly = sum(row["monthly_returns"]) / max(window_count, 1)
    avg_portfolio = sum(row["portfolio_returns"]) / max(window_count, 1)
    avg_auc = sum(value for value in row["auc_values"] if value is not None) / max(sum(1 for value in row["auc_values"] if value is not None), 1)
    avg_brier = sum(value for value in row["brier_values"] if value is not None) / max(sum(1 for value in row["brier_values"] if value is not None), 1)
    positive = sum(1 for value in row["portfolio_returns"] if value > 0)
    return {
        "calibration_method": row["calibration_method"],
        "ml_long_threshold": row["ml_long_threshold"],
        "ml_short_confidence_threshold": row["ml_short_confidence_threshold"],
        "decision_long_threshold": row["ml_long_threshold"],
        "decision_short_score_threshold": round(1.0 - row["ml_short_confidence_threshold"], 4),
        "window_count": window_count,
        "positive_windows": positive,
        "positive_window_ratio": round(positive / max(window_count, 1), 4),
        "avg_portfolio_return_pct": round(avg_portfolio, 2),
        "avg_monthly_return_pct": round(avg_monthly, 2),
        "avg_auc": round(avg_auc, 4) if row["auc_values"] else None,
        "avg_brier": round(avg_brier, 6) if row["brier_values"] else None,
        "worst_window_return_pct": round(min(row["portfolio_returns"]) if row["portfolio_returns"] else 0.0, 2),
        "best_window_return_pct": round(max(row["portfolio_returns"]) if row["portfolio_returns"] else 0.0, 2),
    }


def main() -> None:
    args = build_parser().parse_args()
    config = load_v2_runtime_config()
    base_frames = load_base_feature_frames(config, days=args.days)
    champion_frames, coverage = build_funding_premium_overlay_frames(base_frames, timeframe="1h")

    windows = build_walkforward_windows(
        champion_frames,
        train_bars=args.train_days * 24,
        test_bars=args.test_days * 24,
        step_bars=args.step_days * 24,
    )

    combo_rows = {}
    window_details = []

    for window in windows:
        method_predictions = {}
        method_metrics = {}
        method_snapshot = {}
        for method in args.calibration_methods:
            predicted_frames, calibration, train_metrics, decision_snapshot, _ = train_window_bundle(
                config,
                champion_frames,
                window,
                calibration_fraction=args.calibration_fraction,
                calibration_method=method,
            )
            method_predictions[method] = predicted_frames
            method_metrics[method] = train_metrics
            method_snapshot[method] = {
                "decision_snapshot": decision_snapshot,
                "calibration": calibration.metrics,
            }

            for long_threshold in args.long_thresholds:
                for short_confidence_threshold in args.short_confidence_thresholds:
                    combo_key = (method, round(long_threshold, 4), round(short_confidence_threshold, 4))
                    row = combo_rows.setdefault(
                        combo_key,
                        {
                            "calibration_method": method,
                            "ml_long_threshold": round(long_threshold, 4),
                            "ml_short_confidence_threshold": round(short_confidence_threshold, 4),
                            "portfolio_returns": [],
                            "monthly_returns": [],
                            "auc_values": [],
                            "brier_values": [],
                        },
                    )

                    params = dict(BEST_TREND_CONFIG)
                    params["ml_long_threshold"] = float(long_threshold)
                    params["ml_short_threshold"] = float(short_confidence_threshold)

                    portfolio_return = 0.0
                    for pair_name, frame in method_predictions[method].items():
                        result = run_trend_backtest(frame, **params)
                        portfolio_return += result["metrics"]["total_return_pct"]

                    months = (window.test_end - window.test_start) / (24 * 30.44)
                    row["portfolio_returns"].append(round(portfolio_return, 2))
                    row["monthly_returns"].append(round(portfolio_return / months, 2))
                    row["auc_values"].append(method_metrics[method]["probability_test"]["calibrated_auc"])
                    row["brier_values"].append(method_metrics[method]["probability_test"]["calibrated_brier"])

        window_details.append(
            {
                "window": {
                    "index": window.index,
                    "train_start_ts": window.train_start_ts,
                    "train_end_ts": window.train_end_ts,
                    "test_start_ts": window.test_start_ts,
                    "test_end_ts": window.test_end_ts,
                },
                "methods": method_snapshot,
            }
        )

    ranked = sorted(
        (summarize_combo(row) for row in combo_rows.values()),
        key=lambda row: (
            row["avg_monthly_return_pct"],
            row["positive_window_ratio"],
            row["avg_auc"] if row["avg_auc"] is not None else -999,
        ),
        reverse=True,
    )
    best = ranked[0]

    summary = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "variant": "funding_premium",
        "coverage": coverage,
        "settings": {
            "days": args.days,
            "train_days": args.train_days,
            "test_days": args.test_days,
            "step_days": args.step_days,
            "calibration_fraction": args.calibration_fraction,
            "long_thresholds": args.long_thresholds,
            "short_confidence_thresholds": args.short_confidence_thresholds,
            "calibration_methods": args.calibration_methods,
            "window_count": len(windows),
        },
        "best": best,
        "ranked": ranked,
        "window_details": window_details,
    }

    out_path = Path("v3/data/optimization/funding_premium_threshold_optimization.json")
    write_summary_json(summary, out_path)

    print("Top threshold profiles:")
    for row in ranked[:5]:
        print(
            f"  method={row['calibration_method']:>8} "
            f"long={row['ml_long_threshold']:.2f} "
            f"short_conf={row['ml_short_confidence_threshold']:.2f} "
            f"avg_monthly={row['avg_monthly_return_pct']:+.2f}% "
            f"positive={row['positive_windows']}/{row['window_count']} "
            f"avg_auc={row['avg_auc']:.4f}"
        )

    print(f"\nBest profile: {best}")
    print(f"Saved summary to {out_path.resolve()}")


if __name__ == "__main__":
    main()
