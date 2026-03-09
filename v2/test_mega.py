#!/usr/bin/env python3
"""MEGA TEST — 8 pairs × hybrid trend+ML × leverage × pyramiding.

Goal: Push toward 20%/month by 3 key multipliers:
  1. More pairs (8 instead of 3) — more trade opportunities
  2. Leverage (3-5x simulated futures)
  3. Pyramiding — add to winning positions

Also tests a dual-timeframe approach: both 20-day and 10-day Donchian signals.
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


def run_trend_backtest_with_pyramiding(
    df: pl.DataFrame,
    entry_period: int = 480,
    exit_period: int = 240,
    atr_period: int = 48,
    atr_stop_mult: float = 4.0,
    risk_per_trade: float = 0.05,
    max_position_pct: float = 0.60,
    initial_capital: float = 100_000.0,
    fee_rate: float = 0.001,
    slippage_bps: float = 5.0,
    ml_filter: bool = False,
    ml_long_threshold: float = 0.30,
    ml_short_threshold: float = 0.30,
    # Pyramiding params
    pyramid_levels: int = 3,          # max additional entries
    pyramid_atr_interval: float = 1.5,  # add every 1.5 ATR in profit
) -> dict:
    """Donchian breakout with pyramiding (add to winners).
    
    After initial entry, adds a new position every pyramid_atr_interval ATRs
    of profit, up to pyramid_levels additional entries.
    """
    import talib
    
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    
    if ml_filter and "pred_long_prob" in df.columns:
        ml_prob = df["pred_long_prob"].to_numpy()
    else:
        ml_prob = np.full(len(close), 0.5)
    
    atr = talib.ATR(high, low, close, timeperiod=atr_period)
    
    # Donchian channels
    n = len(close)
    entry_upper = np.full(n, np.nan)
    entry_lower = np.full(n, np.nan)
    exit_upper = np.full(n, np.nan)
    exit_lower = np.full(n, np.nan)
    
    for i in range(entry_period, n):
        entry_upper[i] = np.max(high[i - entry_period:i])
        entry_lower[i] = np.min(low[i - entry_period:i])
    for i in range(exit_period, n):
        exit_upper[i] = np.max(high[i - exit_period:i])
        exit_lower[i] = np.min(low[i - exit_period:i])
    
    equity = np.full(n, initial_capital)
    trades = []
    
    # Position state — supports multiple sub-positions (pyramid)
    position_side = 0.0  # +1 long, -1 short, 0 flat
    sub_positions = []   # list of {entry_price, size_units, entry_idx}
    trail_stop = 0.0
    trail_peak = 0.0
    next_pyramid_price = 0.0
    pyramids_used = 0
    
    warmup = max(entry_period, exit_period, atr_period) + 1
    
    for i in range(1, n):
        equity[i] = equity[i - 1]
        
        if i < warmup or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        
        slip = slippage_bps / 10_000
        current_atr = atr[i]
        
        # ---- PYRAMIDING: add to winning position ----
        if position_side != 0.0 and pyramids_used < pyramid_levels:
            should_pyramid = False
            if position_side > 0 and close[i] > next_pyramid_price:
                should_pyramid = True
            elif position_side < 0 and close[i] < next_pyramid_price:
                should_pyramid = True
            
            if should_pyramid:
                risk_dollars = equity[i] * risk_per_trade
                stop_distance = atr_stop_mult * current_atr
                trade_size = risk_dollars / stop_distance
                max_size = equity[i] * max_position_pct / close[i]
                total_existing = sum(sp["size_units"] for sp in sub_positions)
                remaining = max(max_size - total_existing, 0)
                add_size = min(trade_size, remaining)
                
                if add_size > 0 and remaining > 0:
                    add_price = close[i] * (1 + slip * position_side)
                    sub_positions.append({
                        "entry_price": add_price,
                        "size_units": add_size,
                        "entry_idx": i,
                    })
                    pyramids_used += 1
                    
                    # Set next pyramid level
                    if position_side > 0:
                        next_pyramid_price = close[i] + pyramid_atr_interval * current_atr
                    else:
                        next_pyramid_price = close[i] - pyramid_atr_interval * current_atr
                    
                    # Tighten trailing stop to lock in profits
                    if position_side > 0:
                        trail_peak = max(trail_peak, close[i])
                        trail_stop = max(trail_stop, trail_peak - atr_stop_mult * current_atr)
                    else:
                        trail_peak = min(trail_peak, close[i])
                        trail_stop = min(trail_stop, trail_peak + atr_stop_mult * current_atr)
        
        # ---- EXIT LOGIC ----
        if position_side != 0.0:
            should_exit = False
            exit_reason = ""
            
            if position_side > 0:
                trail_peak = max(trail_peak, close[i])
                trail_stop = max(trail_stop, trail_peak - atr_stop_mult * current_atr)
                
                if close[i] <= trail_stop:
                    should_exit, exit_reason = True, "trailing_stop"
                elif close[i] < exit_lower[i]:
                    should_exit, exit_reason = True, "exit_channel"
                elif close[i] < entry_lower[i]:
                    should_exit, exit_reason = True, "reversal_signal"
            else:
                trail_peak = min(trail_peak, close[i])
                trail_stop = min(trail_stop, trail_peak + atr_stop_mult * current_atr)
                
                if close[i] >= trail_stop:
                    should_exit, exit_reason = True, "trailing_stop"
                elif close[i] > exit_upper[i]:
                    should_exit, exit_reason = True, "exit_channel"
                elif close[i] > entry_upper[i]:
                    should_exit, exit_reason = True, "reversal_signal"
            
            if should_exit:
                # Close ALL sub-positions
                total_pnl = 0.0
                total_cost = 0.0
                first_entry_idx = sub_positions[0]["entry_idx"] if sub_positions else i
                avg_entry = np.mean([sp["entry_price"] for sp in sub_positions]) if sub_positions else 0
                total_units = sum(sp["size_units"] for sp in sub_positions)
                
                for sp in sub_positions:
                    if position_side > 0:
                        ep = close[i] * (1 - slip)
                        pnl = sp["size_units"] * (ep - sp["entry_price"])
                    else:
                        ep = close[i] * (1 + slip)
                        pnl = sp["size_units"] * (sp["entry_price"] - ep)
                    fee = fee_rate * sp["size_units"] * (sp["entry_price"] + ep)
                    total_pnl += pnl
                    total_cost += fee
                
                net_pnl = total_pnl - total_cost
                pnl_pct = net_pnl / equity[i - 1] * 100
                equity[i] += net_pnl
                
                trades.append({
                    "entry_idx": first_entry_idx,
                    "exit_idx": i,
                    "side": "LONG" if position_side > 0 else "SHORT",
                    "entry_price": round(avg_entry, 4),
                    "exit_price": round(close[i], 4),
                    "pnl_pct": round(pnl_pct, 4),
                    "equity_after": round(equity[i], 2),
                    "exit_reason": exit_reason,
                    "hold_bars": i - first_entry_idx,
                    "pyramid_count": len(sub_positions),
                })
                
                # Stop-and-reverse
                if exit_reason == "reversal_signal":
                    new_side = -position_side
                    position_side = new_side
                    sub_positions = []
                    pyramids_used = 0
                    
                    risk_dollars = equity[i] * risk_per_trade
                    stop_distance = atr_stop_mult * current_atr
                    trade_size = risk_dollars / stop_distance
                    max_size = equity[i] * max_position_pct / close[i]
                    pos_size = min(trade_size, max_size)
                    
                    entry_price = close[i] * (1 + slip * new_side)
                    sub_positions.append({
                        "entry_price": entry_price,
                        "size_units": pos_size,
                        "entry_idx": i,
                    })
                    
                    if new_side > 0:
                        trail_peak = close[i]
                        trail_stop = trail_peak - atr_stop_mult * current_atr
                        next_pyramid_price = close[i] + pyramid_atr_interval * current_atr
                    else:
                        trail_peak = close[i]
                        trail_stop = trail_peak + atr_stop_mult * current_atr
                        next_pyramid_price = close[i] - pyramid_atr_interval * current_atr
                    continue
                else:
                    position_side = 0.0
                    sub_positions = []
                    pyramids_used = 0
        
        # ---- ENTRY LOGIC ----
        if position_side == 0.0:
            long_signal = close[i] > entry_upper[i] if not np.isnan(entry_upper[i]) else False
            short_signal = close[i] < entry_lower[i] if not np.isnan(entry_lower[i]) else False
            
            if ml_filter:
                if long_signal and ml_prob[i] < ml_long_threshold:
                    long_signal = False
                if short_signal and (1.0 - ml_prob[i]) < ml_short_threshold:
                    short_signal = False
            
            side = 0.0
            if long_signal:
                side = 1.0
            elif short_signal:
                side = -1.0
            
            if side != 0.0:
                risk_dollars = equity[i] * risk_per_trade
                stop_distance = atr_stop_mult * current_atr
                trade_size = risk_dollars / stop_distance
                max_size = equity[i] * max_position_pct / close[i]
                pos_size = min(trade_size, max_size)
                
                entry_price = close[i] * (1 + slip * side)
                sub_positions = [{
                    "entry_price": entry_price,
                    "size_units": pos_size,
                    "entry_idx": i,
                }]
                position_side = side
                pyramids_used = 0
                
                if side > 0:
                    trail_peak = close[i]
                    trail_stop = trail_peak - atr_stop_mult * current_atr
                    next_pyramid_price = close[i] + pyramid_atr_interval * current_atr
                else:
                    trail_peak = close[i]
                    trail_stop = trail_peak + atr_stop_mult * current_atr
                    next_pyramid_price = close[i] - pyramid_atr_interval * current_atr
    
    # Close open position at end
    if position_side != 0.0 and sub_positions:
        total_pnl = 0.0
        total_cost = 0.0
        first_entry_idx = sub_positions[0]["entry_idx"]
        avg_entry = np.mean([sp["entry_price"] for sp in sub_positions])
        
        for sp in sub_positions:
            slip = slippage_bps / 10_000
            if position_side > 0:
                ep = close[-1] * (1 - slip)
                pnl = sp["size_units"] * (ep - sp["entry_price"])
            else:
                ep = close[-1] * (1 + slip)
                pnl = sp["size_units"] * (sp["entry_price"] - ep)
            fee = fee_rate * sp["size_units"] * (sp["entry_price"] + ep)
            total_pnl += pnl
            total_cost += fee
        
        net_pnl = total_pnl - total_cost
        pnl_pct = net_pnl / equity[-2] * 100
        equity[-1] += net_pnl
        
        trades.append({
            "entry_idx": first_entry_idx,
            "exit_idx": n - 1,
            "side": "LONG" if position_side > 0 else "SHORT",
            "entry_price": round(avg_entry, 4),
            "exit_price": round(close[-1], 4),
            "pnl_pct": round(pnl_pct, 4),
            "equity_after": round(equity[-1], 2),
            "exit_reason": "end_of_data",
            "hold_bars": n - 1 - first_entry_idx,
            "pyramid_count": len(sub_positions),
        })
    
    # Metrics
    returns = np.diff(equity) / equity[:-1]
    returns = returns[~np.isnan(returns)]
    
    total_return = (equity[-1] / initial_capital - 1) * 100
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(8760)
    
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min() * 100
    
    winning = [t for t in trades if t["pnl_pct"] > 0]
    losing = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = len(winning) / max(len(trades), 1) * 100
    
    total_wins = sum(t["pnl_pct"] for t in winning)
    total_losses = abs(sum(t["pnl_pct"] for t in losing))
    profit_factor = total_wins / max(total_losses, 0.001)
    
    pyramided_trades = [t for t in trades if t.get("pyramid_count", 1) > 1]
    
    return {
        "metrics": {
            "total_return_pct": round(total_return, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "total_trades": len(trades),
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "pyramided_trades": len(pyramided_trades),
            "long_pnl": round(sum(t["pnl_pct"] for t in trades if t["side"] == "LONG"), 2),
            "short_pnl": round(sum(t["pnl_pct"] for t in trades if t["side"] == "SHORT"), 2),
        },
        "equity_curve": equity.tolist(),
        "trades": trades,
    }


def main():
    config = load_config(str(Path(__file__).resolve().parent / "config.yaml"))
    store = DataStore(config)

    # Load ALL available pairs
    all_pairs = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT",
        "AVAX/USDT", "DOGE/USDT", "LINK/USDT", "ADA/USDT", "DOT/USDT",
    ]
    
    pair_dfs = {}
    for pair in all_pairs:
        try:
            df = store.load_ohlcv(pair, "1h", last_n_days=1095)
            if len(df) > 500:
                feat_df = build_features(df)
                pair_dfs[pair] = feat_df
                print(f"  {pair}: {len(feat_df)} rows")
        except Exception as e:
            print(f"  {pair}: SKIP ({e})")
    store.close()
    
    print(f"\n  Total pairs loaded: {len(pair_dfs)}")

    # Train ML model on original 3 pairs (avoid leaking new pair data)
    original_pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    names = get_feature_names(list(pair_dfs.values())[0])
    model = LGBMModel(config)
    
    train_parts = []
    test_parts = []
    for pair_name in original_pairs:
        if pair_name in pair_dfs:
            labeled = model.create_labels(pair_dfs[pair_name])
            split_idx = int(len(labeled) * 0.8)
            train_parts.append(labeled[:split_idx])
            test_parts.append(labeled[split_idx:])
    
    combined_train = pl.concat(train_parts)
    combined_test = pl.concat(test_parts)
    metrics = model.train(combined_train, combined_test, names)
    print(f"\nML Model: AUC={metrics['auc']:.4f}")

    # ============================================================
    # Configuration matrix
    # ============================================================
    configs = {
        "A_base_3pair": {
            "pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            "use_pyramiding": False,
            "leverage": 1.0,
            "params": {
                "entry_period": 480, "exit_period": 240, "atr_period": 48,
                "atr_stop_mult": 4.0, "risk_per_trade": 0.05,
                "max_position_pct": 0.60,
                "ml_filter": True, "ml_long_threshold": 0.30, "ml_short_threshold": 0.30,
            },
        },
        "B_8pair_1x": {
            "pairs": list(pair_dfs.keys()),
            "use_pyramiding": False,
            "leverage": 1.0,
            "params": {
                "entry_period": 480, "exit_period": 240, "atr_period": 48,
                "atr_stop_mult": 4.0, "risk_per_trade": 0.05,
                "max_position_pct": 0.60,
                "ml_filter": True, "ml_long_threshold": 0.30, "ml_short_threshold": 0.30,
            },
        },
        "C_8pair_3x": {
            "pairs": list(pair_dfs.keys()),
            "use_pyramiding": False,
            "leverage": 3.0,
            "params": {
                "entry_period": 480, "exit_period": 240, "atr_period": 48,
                "atr_stop_mult": 4.0, "risk_per_trade": 0.05,
                "max_position_pct": 0.60,
                "ml_filter": True, "ml_long_threshold": 0.30, "ml_short_threshold": 0.30,
            },
        },
        "D_8pair_pyramid_3x": {
            "pairs": list(pair_dfs.keys()),
            "use_pyramiding": True,
            "leverage": 3.0,
            "pyramid_levels": 3,
            "pyramid_atr_interval": 1.5,
            "params": {
                "entry_period": 480, "exit_period": 240, "atr_period": 48,
                "atr_stop_mult": 4.0, "risk_per_trade": 0.05,
                "max_position_pct": 0.60,
                "ml_filter": True, "ml_long_threshold": 0.30, "ml_short_threshold": 0.30,
            },
        },
        "E_8pair_pyramid_5x": {
            "pairs": list(pair_dfs.keys()),
            "use_pyramiding": True,
            "leverage": 5.0,
            "pyramid_levels": 3,
            "pyramid_atr_interval": 1.5,
            "params": {
                "entry_period": 480, "exit_period": 240, "atr_period": 48,
                "atr_stop_mult": 4.0, "risk_per_trade": 0.05,
                "max_position_pct": 0.60,
                "ml_filter": True, "ml_long_threshold": 0.30, "ml_short_threshold": 0.30,
            },
        },
        "F_8pair_nofilter_pyramid_3x": {
            "pairs": list(pair_dfs.keys()),
            "use_pyramiding": True,
            "leverage": 3.0,
            "pyramid_levels": 3,
            "pyramid_atr_interval": 1.5,
            "params": {
                "entry_period": 480, "exit_period": 240, "atr_period": 48,
                "atr_stop_mult": 4.0, "risk_per_trade": 0.05,
                "max_position_pct": 0.60,
                "ml_filter": False,
            },
        },
        "G_8pair_fast_donchian_3x": {
            "pairs": list(pair_dfs.keys()),
            "use_pyramiding": True,
            "leverage": 3.0,
            "pyramid_levels": 2,
            "pyramid_atr_interval": 2.0,
            "params": {
                "entry_period": 240, "exit_period": 120, "atr_period": 48,  # 10-day breakout
                "atr_stop_mult": 3.0, "risk_per_trade": 0.05,
                "max_position_pct": 0.50,
                "ml_filter": True, "ml_long_threshold": 0.30, "ml_short_threshold": 0.30,
            },
        },
        "H_8pair_YOLO_5x": {
            "pairs": list(pair_dfs.keys()),
            "use_pyramiding": True,
            "leverage": 5.0,
            "pyramid_levels": 4,
            "pyramid_atr_interval": 1.0,
            "params": {
                "entry_period": 480, "exit_period": 240, "atr_period": 48,
                "atr_stop_mult": 4.0, "risk_per_trade": 0.08,
                "max_position_pct": 0.70,
                "ml_filter": True, "ml_long_threshold": 0.30, "ml_short_threshold": 0.30,
            },
        },
    }

    # ============================================================
    # Run all configs
    # ============================================================
    results_summary = []
    
    for cfg_name, cfg in configs.items():
        leverage = cfg["leverage"]
        use_pyramid = cfg["use_pyramiding"]
        pairs_to_use = cfg["params"].copy()
        
        total_return = 0.0
        total_trades = 0
        worst_dd = 0.0
        pair_details = []
        
        for pair_name in cfg["pairs"]:
            if pair_name not in pair_dfs:
                continue
            
            feat_df = pair_dfs[pair_name]
            test_start = int(len(feat_df) * 0.8)
            test_df = feat_df[test_start:]
            
            # Add ML predictions
            if cfg["params"].get("ml_filter", False):
                test_df = model.predict(test_df)
            
            if use_pyramid:
                result = run_trend_backtest_with_pyramiding(
                    test_df,
                    pyramid_levels=cfg.get("pyramid_levels", 3),
                    pyramid_atr_interval=cfg.get("pyramid_atr_interval", 1.5),
                    **cfg["params"],
                )
            else:
                result = run_trend_backtest(test_df, **cfg["params"])
            
            m = result["metrics"]
            base_return = m["total_return_pct"]
            
            # Apply leverage
            if leverage > 1.0:
                lev_return = base_return * leverage
                lev_dd = m["max_drawdown_pct"] * leverage
            else:
                lev_return = base_return
                lev_dd = m["max_drawdown_pct"]
            
            total_return += lev_return
            total_trades += m["total_trades"]
            worst_dd = min(worst_dd, lev_dd)
            pair_details.append((pair_name, lev_return, m["total_trades"], m["profit_factor"]))
        
        # Calculate months
        first_df = list(pair_dfs.values())[0]
        test_len = len(first_df) - int(len(first_df) * 0.8)
        months = test_len / (24 * 30.44)
        monthly = total_return / months
        
        results_summary.append({
            "name": cfg_name,
            "total_return": total_return,
            "monthly": monthly,
            "trades": total_trades,
            "worst_dd": worst_dd,
            "pairs_used": len(cfg["pairs"]),
            "leverage": leverage,
            "details": pair_details,
        })
    
    # ============================================================
    # Print results
    # ============================================================
    print(f"\n{'='*80}")
    print(f"  MEGA RESULTS — Ranked by monthly return")
    print(f"{'='*80}")
    
    results_summary.sort(key=lambda x: x["monthly"], reverse=True)
    
    for i, r in enumerate(results_summary):
        print(
            f"\n  #{i+1} {r['name']}"
            f"\n      Return: {r['total_return']:+.1f}% | "
            f"Monthly: {r['monthly']:+.2f}% | "
            f"Annualized: {r['monthly']*12:+.1f}% | "
            f"Trades: {r['trades']} | "
            f"MaxDD: {r['worst_dd']:.1f}% | "
            f"Pairs: {r['pairs_used']} | "
            f"Leverage: {r['leverage']:.0f}x"
        )
        for pair_name, ret, trades, pf in r["details"]:
            sym = pair_name.split("/")[0]
            print(f"        {sym}: {ret:+.1f}% ({trades} trades, PF={pf:.2f})")
    
    # Monthly breakdown for top config
    print(f"\n{'='*80}")
    print(f"  MONTHLY BREAKDOWN — {results_summary[0]['name']}")
    print(f"{'='*80}")
    
    best_cfg = configs[results_summary[0]["name"]]
    leverage = best_cfg["leverage"]
    
    all_monthly = {}
    for pair_name in best_cfg["pairs"]:
        if pair_name not in pair_dfs:
            continue
        feat_df = pair_dfs[pair_name]
        test_start = int(len(feat_df) * 0.8)
        test_df = feat_df[test_start:]
        
        if best_cfg["params"].get("ml_filter", False):
            test_df = model.predict(test_df)
        
        if best_cfg["use_pyramiding"]:
            result = run_trend_backtest_with_pyramiding(
                test_df,
                pyramid_levels=best_cfg.get("pyramid_levels", 3),
                pyramid_atr_interval=best_cfg.get("pyramid_atr_interval", 1.5),
                **best_cfg["params"],
            )
        else:
            result = run_trend_backtest(test_df, **best_cfg["params"])
        
        ts = feat_df["timestamp"].to_numpy()
        for t in result["trades"]:
            idx = test_start + t["entry_idx"]
            if idx < len(ts):
                month = str(ts[idx])[:7]
                if month not in all_monthly:
                    all_monthly[month] = 0.0
                all_monthly[month] += t["pnl_pct"] * leverage
    
    total_positive_months = 0
    total_months = 0
    for month in sorted(all_monthly):
        pnl = all_monthly[month]
        marker = "+++" if pnl >= 20 else "++" if pnl >= 10 else "+" if pnl > 0 else "---" if pnl < -10 else "--" if pnl < -5 else "-"
        print(f"    {month}: {pnl:+.1f}% {marker}")
        total_months += 1
        if pnl > 0:
            total_positive_months += 1
    
    print(f"\n    Win months: {total_positive_months}/{total_months} ({total_positive_months/max(total_months,1)*100:.0f}%)")


if __name__ == "__main__":
    main()
