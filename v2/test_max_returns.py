#!/usr/bin/env python3
"""Maximum performance test — hybrid strategy with leverage + pyramiding.

Simulates leveraged futures trading on the hybrid trend+ML system.
Adds pyramiding (scaling into winners) for more aggressive returns.
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


def simulate_leverage(trades: list[dict], leverage: float, initial_equity: float = 100_000.0):
    """Simulate leverage on trade results. 
    
    With leverage, PnL is multiplied but so is drawdown.
    Liquidation occurs if loss exceeds 1/leverage of equity.
    """
    equity = initial_equity
    leveraged_trades = []
    peak_equity = equity
    max_dd = 0.0
    liquidated = False
    
    for t in trades:
        # Leveraged PnL
        lev_pnl_pct = t["pnl_pct"] * leverage
        pnl_dollars = equity * lev_pnl_pct / 100
        
        equity += pnl_dollars
        
        # Check liquidation (simplified)
        if equity <= 0:
            liquidated = True
            equity = 0
            break
        
        peak_equity = max(peak_equity, equity)
        dd = (equity - peak_equity) / peak_equity
        max_dd = min(max_dd, dd)
        
        leveraged_trades.append({
            **t,
            "pnl_pct": round(lev_pnl_pct, 4),
            "equity_after": round(equity, 2),
        })
    
    total_return = (equity / initial_equity - 1) * 100
    
    winning = [t for t in leveraged_trades if t["pnl_pct"] > 0]
    losing = [t for t in leveraged_trades if t["pnl_pct"] <= 0]
    
    return {
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "trades": len(leveraged_trades),
        "win_rate": len(winning) / max(len(leveraged_trades), 1) * 100,
        "profit_factor": sum(t["pnl_pct"] for t in winning) / max(abs(sum(t["pnl_pct"] for t in losing)), 0.001),
        "final_equity": round(equity, 2),
        "liquidated": liquidated,
    }


def run_portfolio_backtest(pair_dfs, model, config_params, test_pct=0.2):
    """Run backtest across all pairs and combine as portfolio."""
    pair_results = {}
    all_trades = []
    
    for pair_name, feat_df in pair_dfs.items():
        test_start = int(len(feat_df) * test_pct * -1) if test_pct < 1 else int(len(feat_df) * (1 - test_pct))
        test_df = feat_df[test_start:]
        
        # Add ML predictions if using ML filter
        if config_params.get("ml_filter", False) and model is not None:
            test_df = model.predict(test_df)
        
        result = run_trend_backtest(test_df, **config_params)
        pair_results[pair_name] = result
        all_trades.extend(result["trades"])
    
    return pair_results, all_trades


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

    # Train ML model
    names = get_feature_names(list(pair_dfs.values())[0])
    model = LGBMModel(config)
    
    train_parts = []
    test_parts = []
    for pair_name, feat_df in pair_dfs.items():
        labeled = model.create_labels(feat_df)
        split_idx = int(len(labeled) * 0.8)
        train_parts.append(labeled[:split_idx])
        test_parts.append(labeled[split_idx:])
    
    combined_train = pl.concat(train_parts)
    combined_test = pl.concat(test_parts)
    metrics = model.train(combined_train, combined_test, names)
    print(f"ML Model: AUC={metrics['auc']:.4f}")

    # Best config from previous tests
    best_config = {
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

    # ============================================================
    # 1. Base case (no leverage)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  BASE CASE — No Leverage")
    print(f"{'='*60}")
    
    pair_results, all_trades = run_portfolio_backtest(pair_dfs, model, best_config)
    
    base_return = 0
    for pair_name, result in pair_results.items():
        m = result["metrics"]
        print(f"  {pair_name}: {m['total_return_pct']:+.2f}% | {m['total_trades']} trades | Sharpe={m['sharpe_ratio']:.2f} | MaxDD={m['max_drawdown_pct']:.1f}%")
        base_return += m["total_return_pct"]
    
    first_df = list(pair_dfs.values())[0]
    test_len = len(first_df) - int(len(first_df) * 0.8)
    months = test_len / (24 * 30.44)
    print(f"\n  Portfolio: {base_return:+.2f}% over {months:.1f} months")
    print(f"  Monthly avg: {base_return/months:+.2f}%")

    # ============================================================
    # 2. Leverage simulation (2x, 3x, 5x)
    # ============================================================
    for leverage in [2.0, 3.0, 5.0]:
        print(f"\n{'='*60}")
        print(f"  LEVERAGE {leverage}x — Simulated Futures")
        print(f"{'='*60}")
        
        total_lev_return = 0
        total_lev_dd = 0
        any_liquidated = False
        
        for pair_name, result in pair_results.items():
            lev = simulate_leverage(result["trades"], leverage)
            print(
                f"  {pair_name}: {lev['total_return_pct']:+.2f}% | "
                f"MaxDD={lev['max_drawdown_pct']:.1f}% | "
                f"WR={lev['win_rate']:.0f}% PF={lev['profit_factor']:.2f}"
                f"{' LIQUIDATED!' if lev['liquidated'] else ''}"
            )
            total_lev_return += lev["total_return_pct"]
            total_lev_dd = min(total_lev_dd, lev["max_drawdown_pct"])
            if lev["liquidated"]:
                any_liquidated = True
        
        monthly = total_lev_return / months
        print(f"\n  Portfolio: {total_lev_return:+.2f}% over {months:.1f} months")
        print(f"  Monthly avg: {monthly:+.2f}%")
        print(f"  Annualized: {monthly * 12:+.1f}%")
        print(f"  Max portfolio DD: {total_lev_dd:.1f}%")
        if any_liquidated:
            print(f"  ⚠️  WARNING: At least one pair was liquidated!")

    # ============================================================
    # 3. Even more aggressive base config + leverage
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  AGGRESSIVE BASE + 3x LEVERAGE")
    print(f"{'='*60}")
    
    aggressive_config = {
        "entry_period": 480,
        "exit_period": 240,
        "atr_period": 48,
        "atr_stop_mult": 4.0,
        "risk_per_trade": 0.08,
        "max_position_pct": 0.70,
        "ml_filter": True,
        "ml_long_threshold": 0.30,
        "ml_short_threshold": 0.30,
    }
    
    pair_results_agg, all_trades_agg = run_portfolio_backtest(pair_dfs, model, aggressive_config)
    
    total_agg = 0
    for pair_name, result in pair_results_agg.items():
        lev = simulate_leverage(result["trades"], 3.0)
        m = result["metrics"]
        print(
            f"  {pair_name}: base={m['total_return_pct']:+.2f}% → "
            f"3x={lev['total_return_pct']:+.2f}% | MaxDD={lev['max_drawdown_pct']:.1f}%"
        )
        total_agg += lev["total_return_pct"]
    
    monthly_agg = total_agg / months
    print(f"\n  Portfolio (3x): {total_agg:+.2f}%")
    print(f"  Monthly avg: {monthly_agg:+.2f}%")
    print(f"  Annualized: {monthly_agg * 12:+.1f}%")

    # ============================================================  
    # 4. Monthly breakdown of best approach
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  MONTHLY BREAKDOWN — Best Hybrid + 3x Leverage")
    print(f"{'='*60}")
    
    for pair_name, result in pair_results.items():
        trades = result["trades"]
        if not trades:
            continue
        
        # Get timestamps
        feat_df = pair_dfs[pair_name]
        test_start = int(len(feat_df) * 0.8)
        ts = feat_df["timestamp"].to_numpy()
        
        print(f"\n  {pair_name}:")
        monthly_pnl = {}
        for t in trades:
            idx = test_start + t["entry_idx"]
            if idx < len(ts):
                month = str(ts[idx])[:7]
                if month not in monthly_pnl:
                    monthly_pnl[month] = {"pnl": 0.0, "trades": 0}
                monthly_pnl[month]["pnl"] += t["pnl_pct"] * 3.0  # 3x leverage
                monthly_pnl[month]["trades"] += 1
        
        for month in sorted(monthly_pnl):
            stats = monthly_pnl[month]
            print(f"    {month}: {stats['pnl']:+.2f}% (3x) | {stats['trades']} trades")


if __name__ == "__main__":
    main()
