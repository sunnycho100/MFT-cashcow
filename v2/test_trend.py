#!/usr/bin/env python3
"""Test trend-following strategies — Donchian breakout + ATR trailing stop.

Tests multiple configurations across all 3 pairs.
No ML required — pure price-action trend following.
"""

import sys
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config import load_config
from src.data.store import DataStore
from src.features.pipeline import build_features, get_feature_names
from src.validation.trend_backtest import run_trend_backtest


def main():
    config = load_config(str(Path(__file__).resolve().parent / "config.yaml"))
    store = DataStore(config)

    pairs = config.get("trading", {}).get("pairs", ["BTC/USDT"])
    pair_dfs = {}
    for pair in pairs:
        df = store.load_ohlcv(pair, "1h", last_n_days=1095)
        if len(df) > 500:
            feat_df = build_features(df)
            pair_dfs[pair] = feat_df
            print(f"  {pair}: {len(feat_df)} rows")
    store.close()

    # Configurations to test
    configs = [
        # (entry_days, exit_days, atr_period, atr_stop, risk_pct, max_pos, label)
        (20, 10, 48, 3.0, 0.02, 0.40, "classic_20d"),       # Classic Turtle
        (10, 5,  24, 2.0, 0.02, 0.40, "fast_10d"),          # Faster
        (30, 15, 72, 3.5, 0.02, 0.40, "slow_30d"),          # Slower, more stable
        (20, 10, 48, 2.0, 0.03, 0.50, "tight_stop_aggro"),  # Tighter stop, bigger size
        (15, 7,  36, 2.5, 0.02, 0.40, "medium_15d"),        # Medium speed
        (20, 10, 48, 3.0, 0.04, 0.60, "classic_big_risk"),  # Classic, more risk
        (7,  3,  14, 1.5, 0.02, 0.30, "very_fast_7d"),      # Very fast
        (20, 10, 48, 4.0, 0.02, 0.40, "wide_stop_20d"),     # Wide trailing stop
    ]

    all_results = []

    for entry_days, exit_days, atr_p, atr_s, risk, maxp, label in configs:
        entry_period = entry_days * 24
        exit_period = exit_days * 24

        print(f"\n{'='*70}")
        print(f"  CONFIG: {label} (entry={entry_days}d exit={exit_days}d ATR_stop={atr_s}x risk={risk*100}%)")
        print(f"{'='*70}")

        pair_returns = []
        total_trades = 0
        all_trades = []

        for pair_name, feat_df in pair_dfs.items():
            # Use last 20% as test (same split as ML backtest)
            test_start = int(len(feat_df) * 0.8)
            test_df = feat_df[test_start:]

            result = run_trend_backtest(
                test_df,
                entry_period=entry_period,
                exit_period=exit_period,
                atr_period=atr_p,
                atr_stop_mult=atr_s,
                risk_per_trade=risk,
                max_position_pct=maxp,
            )

            m = result["metrics"]
            pair_returns.append(m["total_return_pct"])
            total_trades += m["total_trades"]
            all_trades.extend(result["trades"])

            print(
                f"  {pair_name:12s}: {m['total_return_pct']:+7.2f}% | "
                f"{m['total_trades']:3d} trades ({m['long_trades']}L/{m['short_trades']}S) | "
                f"WR={m['win_rate_pct']:.0f}% PF={m['profit_factor']:.2f} "
                f"Sharpe={m['sharpe_ratio']:.2f} MaxDD={m['max_drawdown_pct']:.1f}% | "
                f"Long: {m['long_pnl_pct']:+.1f}% Short: {m['short_pnl_pct']:+.1f}%"
            )

        avg_ret = np.mean(pair_returns) if pair_returns else 0
        portfolio_return = sum(pair_returns)  # simple sum (trading all pairs)
        
        winning = [t for t in all_trades if t["pnl_pct"] > 0]
        losing = [t for t in all_trades if t["pnl_pct"] <= 0]
        pf = sum(t["pnl_pct"] for t in winning) / max(abs(sum(t["pnl_pct"] for t in losing)), 0.001)
        wr = len(winning) / max(len(all_trades), 1) * 100

        # Calculate months
        first_df = list(pair_dfs.values())[0]
        test_len = len(first_df) - int(len(first_df) * 0.8)
        months = test_len / (24 * 30.44)
        monthly_ret = portfolio_return / months if months > 0 else 0

        print(f"\n  PORTFOLIO: {portfolio_return:+.2f}% ({months:.1f} months) | "
              f"Monthly avg: {monthly_ret:+.2f}% | "
              f"{total_trades} trades | WR={wr:.0f}% PF={pf:.2f}")

        all_results.append({
            "label": label,
            "portfolio_return": portfolio_return,
            "monthly_return": monthly_ret,
            "total_trades": total_trades,
            "win_rate": wr,
            "profit_factor": pf,
            "pair_returns": pair_returns,
        })

    # Final ranking
    print(f"\n\n{'='*70}")
    print(f"  FINAL RANKING (by portfolio return)")
    print(f"{'='*70}")

    all_results.sort(key=lambda x: x["portfolio_return"], reverse=True)
    for rank, r in enumerate(all_results, 1):
        pr = " | ".join(f"{ret:+.1f}%" for ret in r["pair_returns"])
        print(
            f"  #{rank} {r['label']:25s} "
            f"ret={r['portfolio_return']:+7.2f}% "
            f"monthly={r['monthly_return']:+.2f}% "
            f"trades={r['total_trades']:3d} "
            f"WR={r['win_rate']:.0f}% "
            f"PF={r['profit_factor']:.2f} "
            f"[{pr}]"
        )

    # Monthly breakdown of best
    if all_results:
        best = all_results[0]
        print(f"\n  BEST CONFIG: {best['label']}")
        print(f"  Estimated annualized: {best['monthly_return'] * 12:+.1f}%")


if __name__ == "__main__":
    main()
