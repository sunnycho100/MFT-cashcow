#!/usr/bin/env python3
"""End-to-end pipeline test — v2.1 aggressive alpha.

Trains on ALL 3 pairs combined, backtests EACH pair independently,
then shows combined portfolio return with monthly breakdown.
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config import load_config
from src.data.store import DataStore
from src.features.pipeline import build_features, get_feature_names
from src.models.lgbm_model import LGBMModel


def monthly_returns(trades: list[dict], equity_curve: list[float], 
                    timestamps: np.ndarray, test_start_idx: int) -> dict:
    """Calculate monthly return breakdown."""
    if not trades:
        return {}
    
    monthly = {}
    for t in trades:
        rel_idx = t["entry_idx"]  # relative to test data
        if rel_idx < len(timestamps):
            ts = timestamps[rel_idx]
            if hasattr(ts, 'strftime'):
                key = ts.strftime("%Y-%m")
            else:
                key = str(ts)[:7]
            if key not in monthly:
                monthly[key] = {"pnl": 0.0, "trades": 0, "wins": 0}
            monthly[key]["pnl"] += t["pnl_pct"]
            monthly[key]["trades"] += 1
            if t["pnl_pct"] > 0:
                monthly[key]["wins"] += 1
    return monthly


def main():
    config = load_config(str(Path(__file__).resolve().parent / "config.yaml"))
    store = DataStore(config)

    # Load ALL pairs and combine (more data → better model)
    pairs = config.get("trading", {}).get("pairs", ["BTC/USDT"])
    all_dfs = []
    pair_dfs = {}
    for pair in pairs:
        df = store.load_ohlcv(pair, "1h", last_n_days=1095)
        if len(df) > 500:
            feat_df = build_features(df)
            all_dfs.append(feat_df)
            pair_dfs[pair] = feat_df
            print(f"  {pair}: {len(df)} raw → {len(feat_df)} with features")
        else:
            print(f"  {pair}: only {len(df)} rows, skipping")
    store.close()

    if not all_dfs:
        print("Not enough data. Run fetch_data.py first.")
        return

    combined = pl.concat(all_dfs)
    names = get_feature_names(combined)
    print(f"\nCombined: {len(combined):,} rows, {len(names)} features")
    print(f"Feature list: {names[:15]}... ({len(names)} total)")

    # Delete old checkpoint
    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
    for f in ckpt_dir.glob("lgbm_*"):
        f.unlink()
        print(f"Deleted old checkpoint: {f.name}")

    # ============================================================
    # PROPER SPLIT: per-pair 80/20 split BEFORE combining
    # This prevents data leakage (test period data from one pair
    # appearing in training set of combined data)
    # ============================================================
    model = LGBMModel(config)
    
    train_parts = []
    test_parts = {}  # pair_name -> test_df (labeled)
    
    for pair_name, feat_df in pair_dfs.items():
        labeled = model.create_labels(feat_df)
        split_idx = int(len(labeled) * 0.8)
        train_part = labeled[:split_idx]
        test_part = labeled[split_idx:]
        train_parts.append(train_part)
        test_parts[pair_name] = test_part
        print(f"  {pair_name}: {len(train_part)} train / {len(test_part)} test (labeled)")
    
    combined_train = pl.concat(train_parts)
    combined_test = pl.concat(list(test_parts.values()))
    print(f"\nTotal train: {len(combined_train):,} rows, Total test: {len(combined_test):,} rows")
    
    metrics = model.train(combined_train, combined_test, names)
    print(f"\nAccuracy: {metrics['accuracy']}")
    print(f"AUC: {metrics['auc']}")
    print(f"Best iter: {metrics['best_iteration']}")
    print(f"Optimal threshold: {metrics['optimal_threshold']}")
    for cls, vals in metrics.get("per_class", {}).items():
        print(f"  {cls}: P={vals['precision']} R={vals['recall']} F1={vals['f1']}")

    # Top features
    print("\nTop 15 features:")
    for name, imp in model.feature_importance(15):
        print(f"  {name:<30} {imp:.0f}")

    # ============================================================
    # Backtest EACH pair independently (last 20% out-of-sample)
    # ============================================================
    from src.validation.backtest import run_backtest

    bt_cfg = config.get("backtest", {})
    risk_cfg = config.get("risk", {})
    
    portfolio_trades = []
    portfolio_returns = []
    
    for pair_name, feat_df in pair_dfs.items():
        print(f"\n{'='*60}")
        print(f"  BACKTEST: {pair_name} (last 20% — truly out-of-sample)")
        print(f"{'='*60}")
        
        # Use the ORIGINAL feature df, not labeled — predict on raw features
        pred_df = model.predict(feat_df)
        test_start = int(len(pred_df) * 0.8)
        test_df = pred_df[test_start:]
        
        # Get timestamps for monthly breakdown
        if "timestamp" in test_df.columns:
            test_timestamps = test_df["timestamp"].to_numpy()
        else:
            test_timestamps = np.arange(len(test_df))
        
        result = run_backtest(
            test_df,
            confidence_threshold=bt_cfg.get("confidence_threshold", 0.45),
            tp_atr_mult=bt_cfg.get("tp_atr_mult", 2.5),
            sl_atr_mult=bt_cfg.get("sl_atr_mult", 1.0),
            max_hold_bars=bt_cfg.get("max_hold_bars", 24),
            trailing_activate_atr=bt_cfg.get("trailing_activate_atr", 1.5),
            trailing_distance_atr=bt_cfg.get("trailing_distance_atr", 0.8),
            kelly_fraction=risk_cfg.get("kelly_fraction", 0.50),
            long_only=bt_cfg.get("long_only", True),
            max_position_pct=bt_cfg.get("max_position_pct", risk_cfg.get("max_position_pct", 0.30)),
        )

        m = result["metrics"]
        print(f"  Return: {m['total_return_pct']:+.2f}%  vs Buy&Hold: {m['buy_hold_return_pct']:+.2f}%")
        print(f"  Sharpe: {m['sharpe_ratio']:.2f}  MaxDD: {m['max_drawdown_pct']:.2f}%")
        print(f"  Trades: {m['total_trades']}  WinRate: {m['win_rate_pct']:.1f}%")
        print(f"  AvgWin: {m['avg_win_pct']:.3f}%  AvgLoss: {m['avg_loss_pct']:.3f}%")
        print(f"  Profit Factor: {m['profit_factor']:.2f}")
        print(f"  Avg Hold: {m['avg_hold_bars']:.1f} bars")
        print(f"  Exit Reasons: {m.get('exit_reasons', {})}")
        
        portfolio_trades.extend(result["trades"])
        portfolio_returns.append(m['total_return_pct'])
        
        # Monthly breakdown
        monthly = monthly_returns(result["trades"], result["equity_curve"], 
                                  test_timestamps, test_start)
        if monthly:
            print(f"\n  Monthly PnL for {pair_name}:")
            for month, stats in sorted(monthly.items()):
                wr = stats['wins'] / max(stats['trades'], 1) * 100
                print(f"    {month}: {stats['pnl']:+.3f}%  ({stats['trades']} trades, {wr:.0f}% WR)")

        # Show last 5 trades
        if result["trades"]:
            print(f"\n  Last 5 trades:")
            for t in result["trades"][-5:]:
                print(
                    f"    {t['side']:5s} entry={t['entry_price']:>10.2f} "
                    f"exit={t['exit_price']:>10.2f} pnl={t['pnl_pct']:+.3f}% "
                    f"hold={t['hold_bars']}bars reason={t['exit_reason']}"
                )

    # ============================================================
    # PORTFOLIO SUMMARY
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  PORTFOLIO SUMMARY (all pairs)")
    print(f"{'='*60}")
    
    total_portfolio_return = sum(portfolio_returns) / len(portfolio_returns) if portfolio_returns else 0
    # More realistic: if trading all pairs simultaneously, returns compound
    compound_return = 1.0
    for r in portfolio_returns:
        compound_return *= (1 + r / 100)
    compound_return = (compound_return - 1) * 100
    
    total_trades = len(portfolio_trades)
    winning_trades = [t for t in portfolio_trades if t["pnl_pct"] > 0]
    losing_trades = [t for t in portfolio_trades if t["pnl_pct"] <= 0]
    
    print(f"  Total trades across all pairs: {total_trades}")
    print(f"  Portfolio win rate: {len(winning_trades)/max(total_trades,1)*100:.1f}%")
    print(f"  Average return per pair: {total_portfolio_return:+.2f}%")
    print(f"  Combined portfolio return: {compound_return:+.2f}%")
    
    if winning_trades and losing_trades:
        avg_win = np.mean([t["pnl_pct"] for t in winning_trades])
        avg_loss = np.mean([t["pnl_pct"] for t in losing_trades])
        total_wins = sum(t["pnl_pct"] for t in winning_trades)
        total_losses = abs(sum(t["pnl_pct"] for t in losing_trades))
        pf = total_wins / max(total_losses, 0.001)
        print(f"  Portfolio avg win: {avg_win:+.3f}%  avg loss: {avg_loss:+.3f}%")
        print(f"  Portfolio profit factor: {pf:.2f}")
    
    # Test period info
    if pair_dfs:
        first_pair = list(pair_dfs.values())[0]
        test_start = int(len(first_pair) * 0.8)
        test_len = len(first_pair) - test_start
        months = test_len / (24 * 30.44)
        if compound_return != 0 and months > 0:
            monthly_avg = compound_return / months
            annualized = ((1 + compound_return/100) ** (12/months) - 1) * 100
            print(f"\n  Test period: ~{months:.1f} months ({test_len} hourly bars)")
            print(f"  Average monthly return: {monthly_avg:+.2f}%")
            print(f"  Annualized return: {annualized:+.1f}%")

    # Probability distribution for first pair
    first_pair_df = list(pair_dfs.values())[0]
    pred_df = model.predict(first_pair_df)
    test_start = int(len(pred_df) * 0.8)
    if "pred_long_prob" in pred_df.columns:
        probs = pred_df[test_start:]["pred_long_prob"].to_numpy()
        print(f"\n--- Probability Distribution (BTC) ---")
        for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
            pct = (probs >= thresh).sum() / len(probs) * 100
            print(f"  P >= {thresh:.2f}: {pct:.1f}% of bars ({int((probs >= thresh).sum())} signals)")


if __name__ == "__main__":
    main()
