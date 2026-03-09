#!/usr/bin/env python3
"""Quick end-to-end test of the v2 pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config import load_config
from src.data.store import DataStore
from src.features.pipeline import build_features, get_feature_names
from src.models.lgbm_model import LGBMModel


def main():
    config = load_config(str(Path(__file__).resolve().parent / "config.yaml"))
    store = DataStore(config)

    # Load 1 year of BTC data
    df = store.load_ohlcv("BTC/USDT", "1h", last_n_days=365)
    store.close()
    print(f"Loaded {len(df)} rows")

    if len(df) < 500:
        print("Not enough data. Run fetch_data.py first.")
        return

    # Build features
    feat_df = build_features(df)
    names = get_feature_names(feat_df)
    print(f"Features: {len(names)}, rows after warmup: {len(feat_df)}")

    # Create labels and train
    model = LGBMModel(config)
    labeled = model.create_labels(feat_df)
    print(f"Labeled rows: {len(labeled)}")

    metrics = model.train(labeled, names)
    print(f"\nAccuracy: {metrics['accuracy']}")
    print(f"Best iter: {metrics['best_iteration']}")
    for cls, vals in metrics.get("per_class", {}).items():
        print(f"  {cls}: P={vals['precision']} R={vals['recall']} F1={vals['f1']}")

    # Top features
    print("\nTop 10 features:")
    for name, imp in model.feature_importance(10):
        print(f"  {name:<25} {imp:.0f}")

    # Quick backtest
    print("\n--- Quick Backtest (last 20%) ---")
    from src.validation.backtest import run_backtest

    pred_df = model.predict(feat_df)
    test_start = int(len(pred_df) * 0.8)
    test_df = pred_df[test_start:]
    result = run_backtest(test_df)
    m = result["metrics"]
    print(f"  Return: {m['total_return_pct']:+.2f}%  vs Buy&Hold: {m['buy_hold_return_pct']:+.2f}%")
    print(f"  Sharpe: {m['sharpe_ratio']:.2f}  MaxDD: {m['max_drawdown_pct']:.2f}%")
    print(f"  Trades: {m['total_trades']}  WinRate: {m['win_rate_pct']:.1f}%")


if __name__ == "__main__":
    main()
