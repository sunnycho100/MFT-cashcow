#!/usr/bin/env python3
"""Aggressive trend-following optimization — push for maximum returns.

Tests aggressive risk sizing, pyramiding, and hybrid ML+trend.
"""

import sys
from pathlib import Path
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config import load_config
from src.data.store import DataStore
from src.features.pipeline import build_features, get_feature_names
from src.models.lgbm_model import LGBMModel
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

    # ============================================================
    # Train ML model (for hybrid filter)
    # ============================================================
    names = get_feature_names(list(pair_dfs.values())[0])
    model = LGBMModel(config)

    train_parts = []
    for pair_name, feat_df in pair_dfs.items():
        labeled = model.create_labels(feat_df)
        split_idx = int(len(labeled) * 0.8)
        train_parts.append(labeled[:split_idx])
    
    combined_train = pl.concat(train_parts)
    combined_test = pl.concat([
        model.create_labels(df)[int(len(model.create_labels(df)) * 0.8):]
        for df in pair_dfs.values()
    ])
    
    metrics = model.train(combined_train, combined_test, names)
    print(f"\nML Model: AUC={metrics['auc']:.4f} iter={metrics['best_iteration']}")

    # ============================================================
    # Aggressive configurations
    # ============================================================
    configs = [
        # Pure trend following — push risk higher
        {"entry_period": 480, "exit_period": 240, "atr_period": 48,
         "atr_stop_mult": 4.0, "risk_per_trade": 0.03, "max_position_pct": 0.50,
         "ml_filter": False, "label": "trend_3%_risk"},
        
        {"entry_period": 480, "exit_period": 240, "atr_period": 48,
         "atr_stop_mult": 4.0, "risk_per_trade": 0.05, "max_position_pct": 0.60,
         "ml_filter": False, "label": "trend_5%_risk"},
         
        {"entry_period": 480, "exit_period": 240, "atr_period": 48,
         "atr_stop_mult": 4.0, "risk_per_trade": 0.08, "max_position_pct": 0.80,
         "ml_filter": False, "label": "trend_8%_risk_YOLO"},
         
        {"entry_period": 480, "exit_period": 240, "atr_period": 48,
         "atr_stop_mult": 5.0, "risk_per_trade": 0.05, "max_position_pct": 0.60,
         "ml_filter": False, "label": "trend_5%_wide_5ATR"},
         
        # Hybrid: trend + ML filter
        {"entry_period": 480, "exit_period": 240, "atr_period": 48,
         "atr_stop_mult": 4.0, "risk_per_trade": 0.05, "max_position_pct": 0.60,
         "ml_filter": True, "ml_long_threshold": 0.30, "ml_short_threshold": 0.30,
         "label": "hybrid_5%_filter30"},
         
        {"entry_period": 480, "exit_period": 240, "atr_period": 48,
         "atr_stop_mult": 4.0, "risk_per_trade": 0.05, "max_position_pct": 0.60,
         "ml_filter": True, "ml_long_threshold": 0.28, "ml_short_threshold": 0.28,
         "label": "hybrid_5%_filter28"},
         
        # Faster entries
        {"entry_period": 360, "exit_period": 120, "atr_period": 36,
         "atr_stop_mult": 3.5, "risk_per_trade": 0.04, "max_position_pct": 0.50,
         "ml_filter": False, "label": "fast_15d_3.5ATR"},
         
        {"entry_period": 336, "exit_period": 168, "atr_period": 48,
         "atr_stop_mult": 4.0, "risk_per_trade": 0.05, "max_position_pct": 0.60,
         "ml_filter": False, "label": "14d_entry_4ATR"},
         
        # Very wide stop
        {"entry_period": 480, "exit_period": 240, "atr_period": 72,
         "atr_stop_mult": 6.0, "risk_per_trade": 0.05, "max_position_pct": 0.60,
         "ml_filter": False, "label": "trend_6ATR_wide"},
         
        # 10% risk — extremely aggressive
        {"entry_period": 480, "exit_period": 240, "atr_period": 48,
         "atr_stop_mult": 4.0, "risk_per_trade": 0.10, "max_position_pct": 0.90,
         "ml_filter": False, "label": "trend_10%_EXTREME"},
    ]

    all_results = []

    for cfg in configs:
        label = cfg.pop("label")
        print(f"\n--- {label} ---")

        pair_returns = []
        all_trades = []

        for pair_name, feat_df in pair_dfs.items():
            test_start = int(len(feat_df) * 0.8)
            test_df = feat_df[test_start:]

            # Add ML predictions if needed
            if cfg.get("ml_filter", False):
                test_df = model.predict(test_df)

            result = run_trend_backtest(test_df, **cfg)
            m = result["metrics"]
            pair_returns.append(m["total_return_pct"])
            all_trades.extend(result["trades"])

            print(
                f"  {pair_name:12s}: {m['total_return_pct']:+7.2f}% | "
                f"{m['total_trades']} trades | WR={m['win_rate_pct']:.0f}% "
                f"Sharpe={m['sharpe_ratio']:.2f} MaxDD={m['max_drawdown_pct']:.1f}%"
            )

        portfolio = sum(pair_returns)
        first_df = list(pair_dfs.values())[0]
        test_len = len(first_df) - int(len(first_df) * 0.8)
        months = test_len / (24 * 30.44)
        monthly = portfolio / months if months > 0 else 0

        winning = [t for t in all_trades if t["pnl_pct"] > 0]
        losing = [t for t in all_trades if t["pnl_pct"] <= 0]
        pf = sum(t["pnl_pct"] for t in winning) / max(abs(sum(t["pnl_pct"] for t in losing)), 0.001)

        print(f"  PORTFOLIO: {portfolio:+.2f}% | Monthly: {monthly:+.2f}% | PF={pf:.2f}")

        all_results.append({
            "label": label,
            "portfolio_return": portfolio,
            "monthly_return": monthly,
            "total_trades": len(all_trades),
            "profit_factor": pf,
            "max_dd": max(0, max(-sum(t["pnl_pct"] for t in all_trades[:i+1]) 
                    for i in range(len(all_trades)))) if all_trades else 0,
        })

        cfg["label"] = label  # restore for display

    # Ranking
    print(f"\n\n{'='*70}")
    print(f"  RANKING")
    print(f"{'='*70}")

    all_results.sort(key=lambda x: x["portfolio_return"], reverse=True)
    for rank, r in enumerate(all_results, 1):
        print(
            f"  #{rank:2d} {r['label']:30s} "
            f"ret={r['portfolio_return']:+8.2f}% "
            f"monthly={r['monthly_return']:+6.2f}% "
            f"PF={r['profit_factor']:.2f} "
            f"trades={r['total_trades']:3d}"
        )

    best = all_results[0]
    print(f"\n  BEST: {best['label']}")
    print(f"  Portfolio: {best['portfolio_return']:+.2f}%")
    print(f"  Monthly: {best['monthly_return']:+.2f}%")
    print(f"  Annualized: {best['monthly_return'] * 12:+.1f}%")


if __name__ == "__main__":
    main()
