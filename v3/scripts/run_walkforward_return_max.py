"""Return-max walk-forward evaluation for funding_premium vs aggressive v2 ports."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v2.src.validation.trend_backtest import run_trend_backtest
from v3.src.strategy.hybrid_stack import (
    BEST_TREND_CONFIG,
    build_funding_premium_overlay_frames,
    build_walkforward_windows,
    load_base_feature_frames,
    load_v2_runtime_config,
    train_model_for_window,
    write_summary_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=1095)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--step-days", type=int, default=60)
    parser.add_argument("--chunk-days", type=int, default=31)
    parser.add_argument("--optimization-path", default="v3/data/optimization/funding_premium_threshold_optimization.json")
    parser.add_argument("--output-path", default="v3/data/walkforward/return_max_walkforward_summary.json")
    parser.add_argument(
        "--fee-rate",
        type=float,
        default=0.001,
        help="Per-side fee as fraction of notional (default 0.001 = 10 bps per side).",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=5.0,
        help="Slippage in basis points applied on entry/exit (default 5).",
    )
    parser.add_argument(
        "--stress-execution",
        action="store_true",
        help="Also write a second summary with higher fee/slippage (1.5x fee, 2x slippage) to stress-test execution assumptions.",
    )
    parser.add_argument(
        "--stress-output-path",
        default="v3/data/walkforward/return_max_walkforward_execution_stress.json",
        help="Output path when --stress-execution is set.",
    )
    return parser


def load_optimized_thresholds(path: str | Path) -> tuple[float, float]:
    import json

    default_long = float(BEST_TREND_CONFIG["ml_long_threshold"])
    default_short = float(BEST_TREND_CONFIG["ml_short_threshold"])
    file_path = Path(path)
    if not file_path.exists():
        return default_long, default_short
    try:
        with open(file_path, "r") as handle:
            payload = json.load(handle)
        best = payload.get("best", {})
        long_threshold = float(best.get("ml_long_threshold", default_long))
        short_conf = float(best.get("ml_short_confidence_threshold", default_short))
        return long_threshold, short_conf
    except Exception:
        return default_long, default_short


def build_profiles(long_threshold: float, short_threshold: float) -> dict[str, dict[str, Any]]:
    champion = dict(BEST_TREND_CONFIG)
    champion["ml_long_threshold"] = long_threshold
    champion["ml_short_threshold"] = short_threshold

    aggressive_v2_hybrid = dict(champion)
    aggressive_v2_hybrid.update(
        {
            "risk_per_trade": 0.08,
            "max_position_pct": 0.70,
        }
    )

    aggressive_v2_trend = dict(champion)
    aggressive_v2_trend.update(
        {
            "ml_filter": False,
            "risk_per_trade": 0.08,
            "max_position_pct": 0.80,
        }
    )

    aggressive_v2_trend_mid = dict(champion)
    aggressive_v2_trend_mid.update(
        {
            "ml_filter": False,
            "risk_per_trade": 0.07,
            "max_position_pct": 0.75,
        }
    )

    aggressive_v2_trend_balanced = dict(champion)
    aggressive_v2_trend_balanced.update(
        {
            "ml_filter": False,
            "risk_per_trade": 0.06,
            "max_position_pct": 0.70,
        }
    )

    fast_aggressive_hybrid = dict(champion)
    fast_aggressive_hybrid.update(
        {
            "entry_period": 360,
            "exit_period": 120,
            "atr_period": 36,
            "atr_stop_mult": 3.5,
            "risk_per_trade": 0.06,
            "max_position_pct": 0.70,
        }
    )

    fast_confidence_hybrid = dict(champion)
    fast_confidence_hybrid.update(
        {
            "entry_period": 336,
            "exit_period": 144,
            "atr_period": 36,
            "atr_stop_mult": 3.5,
            "risk_per_trade": 0.055,
            "max_position_pct": 0.68,
            "confidence_weighted_sizing": True,
            "min_confidence_risk_mult": 0.60,
            "max_confidence_risk_mult": 1.80,
        }
    )

    swing_confidence_hybrid = dict(champion)
    swing_confidence_hybrid.update(
        {
            "entry_period": 240,
            "exit_period": 96,
            "atr_period": 24,
            "atr_stop_mult": 3.2,
            "risk_per_trade": 0.045,
            "max_position_pct": 0.62,
            "confidence_weighted_sizing": True,
            "min_confidence_risk_mult": 0.55,
            "max_confidence_risk_mult": 1.70,
        }
    )

    trend_16d_balanced = dict(champion)
    trend_16d_balanced.update(
        {
            "ml_filter": False,
            "entry_period": 384,
            "exit_period": 192,
            "atr_period": 36,
            "atr_stop_mult": 3.6,
            "risk_per_trade": 0.065,
            "max_position_pct": 0.72,
        }
    )

    trend_14d_balanced = dict(champion)
    trend_14d_balanced.update(
        {
            "ml_filter": False,
            "entry_period": 336,
            "exit_period": 168,
            "atr_period": 36,
            "atr_stop_mult": 3.5,
            "risk_per_trade": 0.06,
            "max_position_pct": 0.70,
        }
    )

    trend_12d_conservative = dict(champion)
    trend_12d_conservative.update(
        {
            "ml_filter": False,
            "entry_period": 288,
            "exit_period": 144,
            "atr_period": 24,
            "atr_stop_mult": 3.3,
            "risk_per_trade": 0.05,
            "max_position_pct": 0.65,
        }
    )

    aggressive_v2_trend_balanced_top2 = dict(aggressive_v2_trend_balanced)
    aggressive_v2_trend_balanced_top2.update(
        {
            "pair_selection": "top2_trend_strength",
        }
    )

    aggressive_v2_trend_mid_top2 = dict(aggressive_v2_trend_mid)
    aggressive_v2_trend_mid_top2.update(
        {
            "pair_selection": "top2_trend_strength",
        }
    )

    aggressive_v2_trend_balanced_top1 = dict(aggressive_v2_trend_balanced)
    aggressive_v2_trend_balanced_top1.update(
        {
            "pair_selection": "top1_trend_strength",
        }
    )

    return {
        "funding_premium": champion,
        "aggressive_v2_hybrid": aggressive_v2_hybrid,
        "aggressive_v2_trend": aggressive_v2_trend,
        "aggressive_v2_trend_mid": aggressive_v2_trend_mid,
        "aggressive_v2_trend_balanced": aggressive_v2_trend_balanced,
        "fast_aggressive_hybrid": fast_aggressive_hybrid,
        "fast_confidence_hybrid": fast_confidence_hybrid,
        "swing_confidence_hybrid": swing_confidence_hybrid,
        "trend_16d_balanced": trend_16d_balanced,
        "trend_14d_balanced": trend_14d_balanced,
        "trend_12d_conservative": trend_12d_conservative,
        "aggressive_v2_trend_balanced_top2": aggressive_v2_trend_balanced_top2,
        "aggressive_v2_trend_mid_top2": aggressive_v2_trend_mid_top2,
        "aggressive_v2_trend_balanced_top1": aggressive_v2_trend_balanced_top1,
    }


def build_ml_off_variants(profiles: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {}
    for name, profile in profiles.items():
        if not bool(profile.get("ml_filter", False)):
            continue
        ml_off = dict(profile)
        ml_off["ml_filter"] = False
        if "confidence_weighted_sizing" in ml_off:
            ml_off["confidence_weighted_sizing"] = False
        variants[f"{name}__ml_off"] = ml_off
    return variants


def summarize_ml_utility_gate(
    leverage_results: dict[str, dict[str, Any]],
    base_profiles: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    gate: dict[str, dict[str, Any]] = {"1x": {}, "2x": {}}
    for leverage_label in ["1x", "2x"]:
        for profile_name, profile in base_profiles.items():
            if not bool(profile.get("ml_filter", False)):
                continue
            off_name = f"{profile_name}__ml_off"
            on_row = leverage_results[leverage_label].get(profile_name)
            off_row = leverage_results[leverage_label].get(off_name)
            if on_row is None or off_row is None:
                continue
            delta_monthly = round(on_row["avg_monthly_return_pct"] - off_row["avg_monthly_return_pct"], 2)
            delta_drawdown = round(on_row["worst_max_drawdown_pct"] - off_row["worst_max_drawdown_pct"], 2)
            delta_trades = round(on_row["avg_trade_count"] - off_row["avg_trade_count"], 2)
            delta_objective = round(on_row["return_max_objective"] - off_row["return_max_objective"], 3)
            keep_ml = delta_objective >= 0.0
            gate[leverage_label][profile_name] = {
                "ml_on_profile": profile_name,
                "ml_off_profile": off_name,
                "ml_on": on_row,
                "ml_off": off_row,
                "delta_avg_monthly_return_pct": delta_monthly,
                "delta_worst_max_drawdown_pct": delta_drawdown,
                "delta_avg_trade_count": delta_trades,
                "delta_return_max_objective": delta_objective,
                "recommendation": "keep_ml" if keep_ml else "drop_ml",
            }
    return gate


def apply_leverage(trades: list[dict[str, Any]], leverage: float, initial_equity: float = 100_000.0) -> dict[str, float]:
    equity = initial_equity
    peak = initial_equity
    max_drawdown = 0.0
    long_pnl = 0.0
    short_pnl = 0.0

    for trade in trades:
        scaled_pct = float(trade.get("pnl_pct", 0.0)) * leverage
        equity *= 1.0 + (scaled_pct / 100.0)
        if equity <= 0:
            equity = 0.0
            max_drawdown = -100.0
            break
        peak = max(peak, equity)
        drawdown = ((equity / peak) - 1.0) * 100.0
        max_drawdown = min(max_drawdown, drawdown)
        if trade.get("side") == "LONG":
            long_pnl += scaled_pct
        elif trade.get("side") == "SHORT":
            short_pnl += scaled_pct

    total_return = ((equity / initial_equity) - 1.0) * 100.0
    return {
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "long_pnl_pct": round(long_pnl, 2),
        "short_pnl_pct": round(short_pnl, 2),
    }


def evaluate_profile_window(
    predicted_frames: dict[str, Any],
    profile: dict[str, Any],
    leverage: float,
    months: float,
    selected_pairs: list[str] | None = None,
    execution: dict[str, float] | None = None,
) -> dict[str, Any]:
    active_pairs = set(selected_pairs or list(predicted_frames.keys()))
    backtest_profile = {k: v for k, v in profile.items() if k != "pair_selection"}
    if execution:
        backtest_profile = {**backtest_profile, **execution}
    pair_results: dict[str, Any] = {}
    portfolio_return = 0.0
    portfolio_trade_count = 0
    portfolio_long = 0.0
    portfolio_short = 0.0
    worst_pair_drawdown = 0.0

    for pair_name, frame in predicted_frames.items():
        if pair_name not in active_pairs:
            continue
        result = run_trend_backtest(frame, **backtest_profile)
        metrics = dict(result["metrics"])
        if leverage != 1.0:
            leveraged = apply_leverage(result["trades"], leverage=leverage)
            metrics["total_return_pct"] = leveraged["total_return_pct"]
            metrics["max_drawdown_pct"] = leveraged["max_drawdown_pct"]
            metrics["long_pnl_pct"] = leveraged["long_pnl_pct"]
            metrics["short_pnl_pct"] = leveraged["short_pnl_pct"]

        pair_results[pair_name] = metrics
        portfolio_return += float(metrics["total_return_pct"])
        portfolio_trade_count += int(metrics["total_trades"])
        portfolio_long += float(metrics.get("long_pnl_pct", 0.0))
        portfolio_short += float(metrics.get("short_pnl_pct", 0.0))
        worst_pair_drawdown = min(worst_pair_drawdown, float(metrics["max_drawdown_pct"]))

    return {
        "pair_results": pair_results,
        "portfolio_return_pct": round(portfolio_return, 2),
        "monthly_return_pct": round(portfolio_return / max(months, 1e-9), 2),
        "max_drawdown_pct": round(worst_pair_drawdown, 2),
        "trade_count": portfolio_trade_count,
        "long_contribution_pct": round(portfolio_long, 2),
        "short_contribution_pct": round(portfolio_short, 2),
        "active_pairs": sorted(active_pairs),
    }


def _train_pair_trend_strength(frame: Any, window: Any) -> float:
    train_slice = frame.slice(window.train_start, window.train_end - window.train_start)
    if len(train_slice) < 2:
        return 0.0
    first_close = float(train_slice["close"][0])
    last_close = float(train_slice["close"][-1])
    if first_close <= 0:
        return 0.0
    return abs((last_close / first_close) - 1.0)


def select_pairs_for_profile(profile: dict[str, Any], full_frames: dict[str, Any], window: Any) -> list[str]:
    mode = str(profile.get("pair_selection", "all"))
    if mode == "all":
        return sorted(full_frames.keys())

    ranked: list[tuple[float, str]] = []
    for pair_name, frame in full_frames.items():
        ranked.append((_train_pair_trend_strength(frame, window), pair_name))
    ranked.sort(reverse=True)

    if mode == "top2_trend_strength":
        return sorted([pair for _, pair in ranked[:2]])
    if mode == "top1_trend_strength":
        return sorted([pair for _, pair in ranked[:1]])
    return sorted(full_frames.keys())


def summarize_profile_windows(rows: list[dict[str, Any]], leverage: float) -> dict[str, Any]:
    returns = [row["portfolio"]["portfolio_return_pct"] for row in rows]
    monthly = [row["portfolio"]["monthly_return_pct"] for row in rows]
    drawdowns = [row["portfolio"]["max_drawdown_pct"] for row in rows]
    trades = [row["portfolio"]["trade_count"] for row in rows]
    longs = [row["portfolio"]["long_contribution_pct"] for row in rows]
    shorts = [row["portfolio"]["short_contribution_pct"] for row in rows]
    positive = sum(1 for value in returns if value > 0)
    window_count = len(rows)
    dd_cap = -12.0 if leverage <= 1.0 else -20.0
    trade_floor = 7.0 if leverage <= 1.0 else 8.0
    worst_dd = min(drawdowns) if drawdowns else 0.0
    avg_trades = sum(trades) / max(window_count, 1)
    dd_penalty = max(0.0, abs(worst_dd) - abs(dd_cap))
    trade_penalty = max(0.0, trade_floor - avg_trades)
    objective = (sum(monthly) / max(window_count, 1)) - 0.15 * dd_penalty - 0.10 * trade_penalty

    return {
        "window_count": window_count,
        "positive_windows": positive,
        "positive_window_ratio": round(positive / max(window_count, 1), 3),
        "avg_portfolio_return_pct": round(sum(returns) / max(window_count, 1), 2),
        "avg_monthly_return_pct": round(sum(monthly) / max(window_count, 1), 2),
        "best_window_return_pct": round(max(returns) if returns else 0.0, 2),
        "worst_window_return_pct": round(min(returns) if returns else 0.0, 2),
        "avg_max_drawdown_pct": round(sum(drawdowns) / max(window_count, 1), 2),
        "worst_max_drawdown_pct": round(worst_dd, 2),
        "avg_trade_count": round(avg_trades, 2),
        "avg_long_contribution_pct": round(sum(longs) / max(window_count, 1), 2),
        "avg_short_contribution_pct": round(sum(shorts) / max(window_count, 1), 2),
        "objective_trade_floor": trade_floor,
        "objective_drawdown_cap": dd_cap,
        "return_max_objective": round(objective, 3),
        "windows": rows,
    }


def run_return_max_pipeline(
    *,
    days: int,
    train_days: int,
    test_days: int,
    step_days: int,
    chunk_days: int,
    optimization_path: str,
    fee_rate: float,
    slippage_bps: float,
    variant_label: str,
) -> dict[str, Any]:
    config = load_v2_runtime_config()
    base_frames = load_base_feature_frames(config, days=days)
    funding_premium_frames, coverage = build_funding_premium_overlay_frames(
        base_frames,
        chunk_days=chunk_days,
        timeframe="1h",
    )

    windows = build_walkforward_windows(
        funding_premium_frames,
        train_bars=train_days * 24,
        test_bars=test_days * 24,
        step_bars=step_days * 24,
    )
    long_threshold, short_threshold = load_optimized_thresholds(optimization_path)
    profiles = build_profiles(long_threshold=long_threshold, short_threshold=short_threshold)
    ml_off_profiles = build_ml_off_variants(profiles)
    evaluation_profiles = {**profiles, **ml_off_profiles}

    execution = {"fee_rate": fee_rate, "slippage_bps": slippage_bps}
    leverage_results: dict[str, dict[str, Any]] = {"1x": {}, "2x": {}}

    for leverage_label, leverage in [("1x", 1.0), ("2x", 2.0)]:
        profile_windows = {name: [] for name in evaluation_profiles}
        for window in windows:
            model, names, metrics = train_model_for_window(config, funding_premium_frames, window)
            predicted_frames = {
                pair_name: model.predict(frame.slice(window.test_start, window.test_end - window.test_start))
                for pair_name, frame in funding_premium_frames.items()
            }
            months = (window.test_end - window.test_start) / (24 * 30.44)

            for profile_name, profile in evaluation_profiles.items():
                selected_pairs = select_pairs_for_profile(profile, funding_premium_frames, window)
                portfolio = evaluate_profile_window(
                    predicted_frames=predicted_frames,
                    profile=profile,
                    leverage=leverage,
                    months=months,
                    selected_pairs=selected_pairs,
                    execution=execution,
                )
                profile_windows[profile_name].append(
                    {
                        "window": asdict(window),
                        "feature_count": len(names),
                        "train_metrics": metrics,
                        "portfolio": portfolio,
                    }
                )

        for profile_name, rows in profile_windows.items():
            leverage_results[leverage_label][profile_name] = summarize_profile_windows(rows, leverage=leverage)

    ml_utility_gate = summarize_ml_utility_gate(leverage_results, profiles)

    comparison = {"1x": {}, "2x": {}}
    for leverage_label in ["1x", "2x"]:
        baseline = leverage_results[leverage_label]["funding_premium"]
        for profile_name, summary in leverage_results[leverage_label].items():
            if profile_name.endswith("__ml_off"):
                continue
            comparison[leverage_label][profile_name] = {
                "delta_avg_monthly_return_pct": round(
                    summary["avg_monthly_return_pct"] - baseline["avg_monthly_return_pct"],
                    2,
                ),
                "delta_positive_window_ratio": round(
                    summary["positive_window_ratio"] - baseline["positive_window_ratio"],
                    3,
                ),
                "delta_worst_max_drawdown_pct": round(
                    summary["worst_max_drawdown_pct"] - baseline["worst_max_drawdown_pct"],
                    2,
                ),
                "delta_avg_trade_count": round(summary["avg_trade_count"] - baseline["avg_trade_count"], 2),
            }

    winners = {}
    for leverage_label in ["1x", "2x"]:
        ranked = sorted(
            (
                (name, row)
                for name, row in leverage_results[leverage_label].items()
                if not name.endswith("__ml_off")
            ),
            key=lambda item: (
                item[1]["return_max_objective"],
                item[1]["avg_monthly_return_pct"],
                item[1]["positive_window_ratio"],
            ),
            reverse=True,
        )
        winners[leverage_label] = {
            "name": ranked[0][0],
            "summary": ranked[0][1],
            "ranking": [name for name, _ in ranked],
        }

    return {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "variant": variant_label,
        "settings": {
            "days": days,
            "train_days": train_days,
            "test_days": test_days,
            "step_days": step_days,
            "chunk_days": chunk_days,
            "window_count": len(windows),
            "optimized_long_threshold": long_threshold,
            "optimized_short_conf_threshold": short_threshold,
            "profile_names": list(profiles.keys()),
            "ml_off_variant_names": list(ml_off_profiles.keys()),
            "fee_rate": fee_rate,
            "slippage_bps": slippage_bps,
        },
        "coverage": coverage,
        "profiles": profiles,
        "ml_off_profiles": ml_off_profiles,
        "results": leverage_results,
        "comparison_vs_funding_premium": comparison,
        "ml_utility_gate": ml_utility_gate,
        "winners": winners,
    }


def _print_summary(leverage_results: dict[str, dict[str, Any]], winners: dict, ml_utility_gate: dict) -> None:
    for leverage_label in ["1x", "2x"]:
        print(f"\n[{leverage_label}]")
        ranked = sorted(
            (
                (name, row)
                for name, row in leverage_results[leverage_label].items()
                if not name.endswith("__ml_off")
            ),
            key=lambda item: item[1]["return_max_objective"],
            reverse=True,
        )
        for name, row in ranked:
            print(
                f"{name:>24}: avg_monthly={row['avg_monthly_return_pct']:+.2f}% "
                f"maxdd={row['worst_max_drawdown_pct']:+.2f}% "
                f"positive={row['positive_windows']}/{row['window_count']} "
                f"trades={row['avg_trade_count']:.1f} "
                f"long={row['avg_long_contribution_pct']:+.2f}% "
                f"short={row['avg_short_contribution_pct']:+.2f}%"
            )
        print(f"Winner: {winners[leverage_label]['name']}")

        gate_rows = ml_utility_gate.get(leverage_label, {})
        if gate_rows:
            print(f"ML utility gate ({leverage_label}):")
            for profile_name, row in gate_rows.items():
                print(
                    f"  {profile_name:>24}: "
                    f"delta_monthly={row['delta_avg_monthly_return_pct']:+.2f}% "
                    f"delta_obj={row['delta_return_max_objective']:+.3f} "
                    f"delta_trades={row['delta_avg_trade_count']:+.1f} "
                    f"delta_maxdd={row['delta_worst_max_drawdown_pct']:+.2f}% "
                    f"-> {row['recommendation']}"
                )


def main() -> None:
    args = build_parser().parse_args()
    summary = run_return_max_pipeline(
        days=args.days,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        chunk_days=args.chunk_days,
        optimization_path=args.optimization_path,
        fee_rate=args.fee_rate,
        slippage_bps=args.slippage_bps,
        variant_label="funding_premium_return_max",
    )

    out_path = Path(args.output_path)
    write_summary_json(summary, out_path)

    print(f"Saved summary to {out_path.resolve()}")
    _print_summary(summary["results"], summary["winners"], summary["ml_utility_gate"])

    if args.stress_execution:
        stress_fee = round(args.fee_rate * 1.5, 6)
        stress_slip = round(args.slippage_bps * 2.0, 2)
        stress_summary = run_return_max_pipeline(
            days=args.days,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            chunk_days=args.chunk_days,
            optimization_path=args.optimization_path,
            fee_rate=stress_fee,
            slippage_bps=stress_slip,
            variant_label="funding_premium_return_max_execution_stress",
        )
        stress_summary["settings"]["execution_stress_note"] = "1.5x fee_rate and 2x slippage_bps vs base CLI defaults for this run"
        stress_path = Path(args.stress_output_path)
        write_summary_json(stress_summary, stress_path)
        print(f"\nExecution stress summary saved to {stress_path.resolve()}")
        _print_summary(stress_summary["results"], stress_summary["winners"], stress_summary["ml_utility_gate"])


if __name__ == "__main__":
    main()
