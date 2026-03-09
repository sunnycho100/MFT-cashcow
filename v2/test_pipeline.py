#!/usr/bin/env python3
"""End-to-end pipeline test — binary model + triple barrier + ATR exits.

Trains on ALL 3 pairs combined (3× more data) for better model quality.
"""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config import load_config
from src.data.store import DataStore
from src.features.pipeline import build_features, get_feature_names
from src.models.lgbm_model import LGBMModel


def main():
    config = load_config(str(Path(__file__).resolve().parent / "config.yaml"))
    store = DataStore(config)

    # Load ALL pairs and combine (more data → better model)
    pairs = config.get("trading", {}).get("pairs", ["BTC/USDT"])
    all_dfs = []
    for pair in pairs:
        df = store.load_ohlcv(pair, "1h", last_n_days=1095)
        if len(df) > 500:
            feat_df = build_features(df)
            all_dfs.append(feat_df)
            print(f"  {pair}: {len(df)} raw → {len(feat_df)} with features")
        else:
            print(f"  {pair}: only {len(df)} rows, skipping")
    store.close()

    if not all_dfs:
        print("Not enough data. Run fetch_data.py first.")
        return

    # Combine all pairs (each pair sorted by time, then concat)
    # No need to sort globally — the train/test split is chronological within each pair
    combined = pl.concat(all_dfs)
    names = get_feature_names(combined)
    print(f"\nCombined: {len(combined):,} rows, {len(names)} features")
    print(f"Feature list: {names}")

    # Delete old checkpoint
    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
    for f in ckpt_dir.glob("lgbm_*"):
        f.unlink()
        print(f"Deleted old checkpoint: {f.name}")

    # Create binary triple barrier labels and train
    model = LGBMModel(config)
    labeled = model.create_labels(combined)
    print(f"Labeled rows: {len(labeled)}")

    metrics = model.train(labeled, names)
    print(f"\nAccuracy: {metrics['accuracy']}")
    print(f"AUC: {metrics['auc']}")
    print(f"Best iter: {metrics['best_iteration']}")
    print(f"Optimal threshold: {metrics['optimal_threshold']}")
    for cls, vals in metrics.get("per_class", {}).items():
        print(f"  {cls}: P={vals['precision']} R={vals['recall']} F1={vals['f1']}")

    # Top features
    print("\nTop 10 features:")
    for name, imp in model.feature_importance(10):
        print(f"  {name:<25} {imp:.0f}")

    # Backtest on BTC only (out-of-sample last 20%)
    print("\n--- Backtest: BTC/USDT (last 20%) ---")
    from src.validation.backtest import run_backtest

    btc_df = all_dfs[0]  # BTC features
    pred_df = model.predict(btc_df)
    test_start = int(len(pred_df) * 0.8)
    test_df = pred_df[test_start:]

    bt_cfg = config.get("backtest", {})
    result = run_backtest(
        test_df,
        confidence_threshold=bt_cfg.get("confidence_threshold", 0.40),
        tp_atr_mult=bt_cfg.get("tp_atr_mult", 1.5),
        sl_atr_mult=bt_cfg.get("sl_atr_mult", 1.5),
        max_hold_bars=bt_cfg.get("max_hold_bars", 12),
        trailing_activate_atr=bt_cfg.get("trailing_activate_atr", 0.0),
        trailing_distance_atr=bt_cfg.get("trailing_distance_atr", 0.0),
        kelly_fraction=config.get("risk", {}).get("kelly_fraction", 0.25),
        long_only=bt_cfg.get("long_only", False),
    )

    m = result["metrics"]
    print(f"  Return: {m['total_return_pct']:+.2f}%  vs Buy&Hold: {m['buy_hold_return_pct']:+.2f}%")
    print(f"  Sharpe: {m['sharpe_ratio']:.2f}  MaxDD: {m['max_drawdown_pct']:.2f}%")
    print(f"  Trades: {m['total_trades']}  WinRate: {m['win_rate_pct']:.1f}%")
    print(f"  AvgWin: {m['avg_win_pct']:.3f}%  AvgLoss: {m['avg_loss_pct']:.3f}%")
    print(f"  Profit Factor: {m['profit_factor']:.2f}")
    print(f"  Avg Hold: {m['avg_hold_bars']:.1f} bars")
    print(f"  Exit Reasons: {m.get('exit_reasons', {})}")

    # Show recent trades
    if result["trades"]:
        print(f"\n--- Last 10 trades ---")
        for t in result["trades"][-10:]:
            print(
                f"  {t['side']:5s} entry={t['entry_price']:>10.2f} "
                f"exit={t['exit_price']:>10.2f} pnl={t['pnl_pct']:+.3f}% "
                f"hold={t['hold_bars']}bars reason={t['exit_reason']}"
            )

    # Probability distribution
    if "pred_long_prob" in pred_df.columns:
        probs = pred_df[test_start:]["pred_long_prob"].to_numpy()
        print(f"\n--- Probability Distribution ---")
        for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            pct = (probs >= thresh).sum() / len(probs) * 100
            print(f"  P >= {thresh:.2f}: {pct:.1f}% of bars")


if __name__ == "__main__":
    main()
