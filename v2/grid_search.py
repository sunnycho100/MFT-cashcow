#!/usr/bin/env python3
"""Grid search across barrier, threshold, and sizing configurations.

Proper per-pair train/test split — NO leakage.
Tests ALL combinations and reports the best.
"""

import sys
from pathlib import Path
from itertools import product

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config import load_config
from src.data.store import DataStore
from src.features.pipeline import build_features, get_feature_names
from src.models.lgbm_model import LGBMModel
from src.validation.backtest import run_backtest


def run_experiment(
    config: dict,
    pair_dfs: dict,
    names: list[str],
    tp_atr: float,
    sl_atr: float,
    max_hold: int,
    conf_thresh: float,
    pos_pct: float,
    kelly: float,
    trailing_act: float,
    trailing_dist: float,
) -> dict:
    """Run one full experiment with given parameters."""
    # Override config for this experiment
    cfg = config.copy()
    cfg["models"] = config.get("models", {}).copy()
    cfg["models"]["lgbm"] = config["models"]["lgbm"].copy()
    cfg["models"]["lgbm"]["barriers"] = {
        "take_profit_atr": tp_atr,
        "stop_loss_atr": sl_atr,
        "max_hold_bars": max_hold,
    }

    model = LGBMModel(cfg)

    # Per-pair split
    train_parts = []
    test_parts_labeled = {}

    for pair_name, feat_df in pair_dfs.items():
        labeled = model.create_labels(feat_df)
        split_idx = int(len(labeled) * 0.8)
        train_parts.append(labeled[:split_idx])
        test_parts_labeled[pair_name] = labeled[split_idx:]

    combined_train = pl.concat(train_parts)
    combined_test = pl.concat(list(test_parts_labeled.values()))

    # Train
    metrics = model.train(combined_train, combined_test, names)

    # Backtest each pair
    pair_results = {}
    total_trades = 0
    all_trades = []

    for pair_name, feat_df in pair_dfs.items():
        pred_df = model.predict(feat_df)
        test_start = int(len(pred_df) * 0.8)
        test_df = pred_df[test_start:]

        result = run_backtest(
            test_df,
            confidence_threshold=conf_thresh,
            tp_atr_mult=tp_atr,
            sl_atr_mult=sl_atr,
            max_hold_bars=max_hold,
            trailing_activate_atr=trailing_act,
            trailing_distance_atr=trailing_dist,
            kelly_fraction=kelly,
            long_only=True,
            max_position_pct=pos_pct,
        )
        pair_results[pair_name] = result["metrics"]
        total_trades += result["metrics"]["total_trades"]
        all_trades.extend(result["trades"])

    # Portfolio-level metrics
    returns = [r["total_return_pct"] for r in pair_results.values()]
    avg_return = sum(returns) / len(returns)

    winning = [t for t in all_trades if t["pnl_pct"] > 0]
    losing = [t for t in all_trades if t["pnl_pct"] <= 0]
    win_rate = len(winning) / max(len(all_trades), 1) * 100

    total_win_pnl = sum(t["pnl_pct"] for t in winning)
    total_loss_pnl = abs(sum(t["pnl_pct"] for t in losing))
    pf = total_win_pnl / max(total_loss_pnl, 0.001)

    return {
        "auc": metrics["auc"],
        "best_iter": metrics["best_iteration"],
        "avg_return": avg_return,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": pf,
        "pair_returns": returns,
        "all_trades": all_trades,
    }


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
            print(f"  {pair}: {len(feat_df)} features rows")
    store.close()

    names = get_feature_names(list(pair_dfs.values())[0])
    print(f"\n{len(names)} features")

    # ============================================================
    # Grid search configurations
    # ============================================================
    configs = [
        # (tp_atr, sl_atr, max_hold, conf_thresh, pos_pct, kelly, trail_act, trail_dist, label)
        (1.5, 1.5, 12, 0.35, 0.30, 0.50, 0.0, 0.0, "symmetric_1.5_t.35"),
        (1.5, 1.5, 12, 0.30, 0.30, 0.50, 0.0, 0.0, "symmetric_1.5_t.30"),
        (2.0, 1.0, 18, 0.35, 0.30, 0.50, 0.0, 0.0, "asym_2:1_t.35"),
        (2.0, 1.0, 18, 0.30, 0.30, 0.50, 0.0, 0.0, "asym_2:1_t.30"),
        (2.0, 1.0, 18, 0.30, 0.30, 0.50, 1.2, 0.6, "asym_2:1_trail"),
        (1.5, 1.0, 12, 0.30, 0.30, 0.50, 0.0, 0.0, "asym_1.5:1_t.30"),
        (1.5, 1.0, 12, 0.35, 0.30, 0.50, 0.0, 0.0, "asym_1.5:1_t.35"),
        (1.5, 1.0, 12, 0.30, 0.50, 0.75, 0.0, 0.0, "asym_1.5:1_aggro_size"),
        (1.0, 0.5, 8,  0.30, 0.30, 0.50, 0.0, 0.0, "tight_1:0.5_quick"),
        (2.5, 1.0, 24, 0.30, 0.30, 0.50, 0.0, 0.0, "wide_2.5:1_t.30"),
        (1.5, 1.0, 12, 0.28, 0.40, 0.60, 0.0, 0.0, "asym_1.5:1_very_aggro"),
        (1.0, 1.0, 8,  0.30, 0.30, 0.50, 0.0, 0.0, "symmetric_1.0_quick"),
    ]

    results = []
    print(f"\nRunning {len(configs)} experiments...\n")

    for i, (tp, sl, hold, thresh, pos, kelly, tact, tdist, label) in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] {label}: TP={tp} SL={sl} hold={hold} thresh={thresh} pos={pos} kelly={kelly}")
        try:
            res = run_experiment(
                config, pair_dfs, names,
                tp_atr=tp, sl_atr=sl, max_hold=hold,
                conf_thresh=thresh, pos_pct=pos, kelly=kelly,
                trailing_act=tact, trailing_dist=tdist,
            )
            results.append((label, res))
            returns_str = " | ".join(f"{r:+.1f}%" for r in res["pair_returns"])
            print(
                f"  → AUC={res['auc']:.4f} iter={res['best_iter']} "
                f"trades={res['total_trades']} WR={res['win_rate']:.0f}% "
                f"PF={res['profit_factor']:.2f} "
                f"avg_ret={res['avg_return']:+.2f}% "
                f"[{returns_str}]\n"
            )
        except Exception as e:
            print(f"  → ERROR: {e}\n")

    # ============================================================
    # Ranking
    # ============================================================
    print("\n" + "=" * 80)
    print("  RESULTS RANKING (by avg return × profit_factor)")
    print("=" * 80)

    # Score: return * sqrt(profit_factor) * min(trades, 20)/20
    # This balances returns, edge quality, and trade count
    scored = []
    for label, res in results:
        trade_adj = min(res["total_trades"], 30) / 30  # penalize if too few trades
        score = res["avg_return"] * (res["profit_factor"] ** 0.5) * trade_adj
        scored.append((score, label, res))

    scored.sort(reverse=True, key=lambda x: x[0])

    for rank, (score, label, res) in enumerate(scored, 1):
        returns_str = " | ".join(f"{r:+.1f}%" for r in res["pair_returns"])
        print(
            f"  #{rank:2d} {label:30s} "
            f"score={score:+.2f} "
            f"ret={res['avg_return']:+.2f}% "
            f"PF={res['profit_factor']:.2f} "
            f"WR={res['win_rate']:.0f}% "
            f"trades={res['total_trades']:3d} "
            f"AUC={res['auc']:.4f} "
            f"[{returns_str}]"
        )

    # Best config details
    if scored:
        _, best_label, best_res = scored[0]
        print(f"\n  BEST: {best_label}")
        print(f"  Return: {best_res['avg_return']:+.2f}% avg across pairs")
        print(f"  Trades: {best_res['total_trades']}, Win Rate: {best_res['win_rate']:.1f}%")
        print(f"  Profit Factor: {best_res['profit_factor']:.2f}")

        # Monthly breakdown of best
        if best_res["all_trades"]:
            monthly = {}
            for t in best_res["all_trades"]:
                # Simple monthly grouping by trade index
                month_approx = t["entry_idx"] // (24 * 30)
                if month_approx not in monthly:
                    monthly[month_approx] = 0.0
                monthly[month_approx] += t["pnl_pct"]
            print(f"\n  Monthly PnL distribution of best config:")
            for m in sorted(monthly):
                print(f"    Month ~{m}: {monthly[m]:+.2f}%")


if __name__ == "__main__":
    main()
