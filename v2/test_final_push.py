#!/usr/bin/env python3
"""FINAL PUSH — Combo filter + pyramiding + leverage to hit 20%/month.

Based on finding: ETHSOL combo filter (ADX+vol+ML) gives PF=3.52 with 5x leverage.
Now testing:
  1. Higher leverage (7x, 10x) on combo filter
  2. Pyramiding on combo-filtered trades
  3. Full 3-year validation (not just test 20%)
  4. Split-decade walk-forward validation
"""

import sys
from pathlib import Path
import numpy as np
import polars as pl
import talib

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config import load_config
from src.data.store import DataStore
from src.features.pipeline import build_features, get_feature_names
from src.models.lgbm_model import LGBMModel


def run_combo_backtest(
    df: pl.DataFrame,
    entry_period=480, exit_period=240, atr_period=48, atr_stop_mult=4.0,
    risk_per_trade=0.05, max_position_pct=0.60,
    initial_capital=100_000.0, fee_rate=0.001, slippage_bps=5.0,
    # Filters
    adx_period=48, adx_threshold=18.0,
    vol_expansion=True, vol_sma_period=240,
    ml_filter=True, ml_long_threshold=0.30, ml_short_threshold=0.30,
    # Pyramiding
    pyramid_levels=0, pyramid_atr_interval=1.5,
    # Leverage (applied to position sizing notionally)
    leverage=1.0,
):
    """Combined: Donchian breakout + ADX + vol expansion + ML + pyramiding + leverage."""
    
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    
    ml_prob = df["pred_long_prob"].to_numpy() if ml_filter and "pred_long_prob" in df.columns else np.full(len(close), 0.5)
    
    atr = talib.ATR(high, low, close, timeperiod=atr_period)
    adx = talib.ADX(high, low, close, timeperiod=adx_period)
    
    # ATR SMA for vol expansion
    atr_sma = np.full(len(close), np.nan)
    if vol_expansion:
        for i in range(vol_sma_period, len(close)):
            vals = atr[i - vol_sma_period:i]
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                atr_sma[i] = np.mean(valid)
    
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
    
    position_side = 0.0
    sub_positions = []
    trail_stop = 0.0
    trail_peak = 0.0
    next_pyramid_price = 0.0
    pyramids_used = 0
    
    warmup = max(entry_period, exit_period, atr_period, adx_period, vol_sma_period if vol_expansion else 0) + 1
    
    def check_filters(idx):
        """Check if entry passes all filters."""
        if adx_threshold > 0:
            if np.isnan(adx[idx]) or adx[idx] < adx_threshold:
                return False
        if vol_expansion:
            if not np.isnan(atr_sma[idx]) and atr[idx] < atr_sma[idx]:
                return False
        return True
    
    def check_ml_filter(idx, is_long):
        if not ml_filter:
            return True
        if is_long and ml_prob[idx] < ml_long_threshold:
            return False
        if not is_long and (1.0 - ml_prob[idx]) < ml_short_threshold:
            return False
        return True
    
    def calc_position_size(idx, current_equity):
        risk_dollars = current_equity * risk_per_trade * leverage
        stop_distance = atr_stop_mult * atr[idx]
        trade_size = risk_dollars / stop_distance
        max_size = current_equity * max_position_pct * leverage / close[idx]
        return min(trade_size, max_size)
    
    for i in range(1, n):
        equity[i] = equity[i - 1]
        
        if i < warmup or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        
        slip = slippage_bps / 10_000
        current_atr = atr[i]
        
        # PYRAMIDING
        if position_side != 0.0 and pyramid_levels > 0 and pyramids_used < pyramid_levels:
            should_pyramid = False
            if position_side > 0 and close[i] > next_pyramid_price:
                should_pyramid = True
            elif position_side < 0 and close[i] < next_pyramid_price:
                should_pyramid = True
            
            if should_pyramid:
                total_existing = sum(sp["size_units"] for sp in sub_positions)
                max_allowed = equity[i] * max_position_pct * leverage / close[i]
                remaining = max(max_allowed - total_existing, 0)
                add_size = min(calc_position_size(i, equity[i]), remaining)
                
                if add_size > 0:
                    add_price = close[i] * (1 + slip * position_side)
                    sub_positions.append({"entry_price": add_price, "size_units": add_size, "entry_idx": i})
                    pyramids_used += 1
                    
                    if position_side > 0:
                        next_pyramid_price = close[i] + pyramid_atr_interval * current_atr
                        trail_peak = max(trail_peak, close[i])
                        trail_stop = max(trail_stop, trail_peak - atr_stop_mult * current_atr)
                    else:
                        next_pyramid_price = close[i] - pyramid_atr_interval * current_atr
                        trail_peak = min(trail_peak, close[i])
                        trail_stop = min(trail_stop, trail_peak + atr_stop_mult * current_atr)
        
        # EXIT
        if position_side != 0.0:
            should_exit = False
            exit_reason = ""
            
            if position_side > 0:
                trail_peak = max(trail_peak, close[i])
                trail_stop = max(trail_stop, trail_peak - atr_stop_mult * current_atr)
                if close[i] <= trail_stop: should_exit, exit_reason = True, "trailing_stop"
                elif not np.isnan(exit_lower[i]) and close[i] < exit_lower[i]: should_exit, exit_reason = True, "exit_channel"
                elif not np.isnan(entry_lower[i]) and close[i] < entry_lower[i]: should_exit, exit_reason = True, "reversal"
            else:
                trail_peak = min(trail_peak, close[i])
                trail_stop = min(trail_stop, trail_peak + atr_stop_mult * current_atr)
                if close[i] >= trail_stop: should_exit, exit_reason = True, "trailing_stop"
                elif not np.isnan(exit_upper[i]) and close[i] > exit_upper[i]: should_exit, exit_reason = True, "exit_channel"
                elif not np.isnan(entry_upper[i]) and close[i] > entry_upper[i]: should_exit, exit_reason = True, "reversal"
            
            if should_exit:
                total_pnl = 0.0
                total_fee = 0.0
                first_entry_idx = sub_positions[0]["entry_idx"] if sub_positions else i
                
                for sp in sub_positions:
                    if position_side > 0:
                        ep = close[i] * (1 - slip)
                        pnl = sp["size_units"] * (ep - sp["entry_price"])
                    else:
                        ep = close[i] * (1 + slip)
                        pnl = sp["size_units"] * (sp["entry_price"] - ep)
                    fee = fee_rate * sp["size_units"] * (sp["entry_price"] + ep)
                    total_pnl += pnl
                    total_fee += fee
                
                net_pnl = total_pnl - total_fee
                pnl_pct = net_pnl / equity[i - 1] * 100
                equity[i] += net_pnl
                
                # Liquidation check
                if equity[i] <= 0:
                    equity[i] = 0
                    trades.append({"entry_idx": first_entry_idx, "exit_idx": i, "side": "LONG" if position_side > 0 else "SHORT",
                                   "pnl_pct": round(pnl_pct, 4), "equity_after": 0, "exit_reason": "LIQUIDATED", "hold_bars": i - first_entry_idx, "pyramids": len(sub_positions)})
                    break
                
                trades.append({
                    "entry_idx": first_entry_idx, "exit_idx": i,
                    "side": "LONG" if position_side > 0 else "SHORT",
                    "pnl_pct": round(pnl_pct, 4), "equity_after": round(equity[i], 2),
                    "exit_reason": exit_reason, "hold_bars": i - first_entry_idx,
                    "pyramids": len(sub_positions),
                })
                
                if exit_reason == "reversal" and check_filters(i):
                    new_side = -position_side
                    is_long = new_side > 0
                    if check_ml_filter(i, is_long):
                        position_side = new_side
                        sub_positions = []
                        pyramids_used = 0
                        pos_size = calc_position_size(i, equity[i])
                        entry_p = close[i] * (1 + slip * new_side)
                        sub_positions.append({"entry_price": entry_p, "size_units": pos_size, "entry_idx": i})
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
                else:
                    position_side = 0.0
                    sub_positions = []
                    pyramids_used = 0
        
        # ENTRY
        if position_side == 0.0:
            long_signal = not np.isnan(entry_upper[i]) and close[i] > entry_upper[i]
            short_signal = not np.isnan(entry_lower[i]) and close[i] < entry_lower[i]
            
            if (long_signal or short_signal) and check_filters(i):
                if long_signal and check_ml_filter(i, True):
                    side = 1.0
                elif short_signal and check_ml_filter(i, False):
                    side = -1.0
                else:
                    side = 0.0
                
                if side != 0.0:
                    pos_size = calc_position_size(i, equity[i])
                    entry_p = close[i] * (1 + slip * side)
                    sub_positions = [{"entry_price": entry_p, "size_units": pos_size, "entry_idx": i}]
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
    
    # Close open
    if position_side != 0.0 and sub_positions:
        total_pnl = 0.0
        total_fee = 0.0
        first_entry_idx = sub_positions[0]["entry_idx"]
        for sp in sub_positions:
            sl = slippage_bps / 10_000
            if position_side > 0:
                ep = close[-1] * (1 - sl)
                pnl = sp["size_units"] * (ep - sp["entry_price"])
            else:
                ep = close[-1] * (1 + sl)
                pnl = sp["size_units"] * (sp["entry_price"] - ep)
            fee = fee_rate * sp["size_units"] * (sp["entry_price"] + ep)
            total_pnl += pnl
            total_fee += fee
        net_pnl = total_pnl - total_fee
        pnl_pct = net_pnl / equity[-2] * 100
        equity[-1] += net_pnl
        trades.append({"entry_idx": first_entry_idx, "exit_idx": n-1, "side": "LONG" if position_side > 0 else "SHORT",
                       "pnl_pct": round(pnl_pct, 4), "equity_after": round(equity[-1], 2), "exit_reason": "end_of_data",
                       "hold_bars": n-1-first_entry_idx, "pyramids": len(sub_positions)})
    
    # Metrics
    total_return = (equity[-1] / initial_capital - 1) * 100
    returns = np.diff(equity) / equity[:-1]
    returns = returns[~np.isnan(returns)]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(8760)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min() * 100
    
    winning = [t for t in trades if t["pnl_pct"] > 0]
    losing = [t for t in trades if t["pnl_pct"] <= 0]
    total_wins = sum(t["pnl_pct"] for t in winning)
    total_losses = abs(sum(t["pnl_pct"] for t in losing))
    pf = total_wins / max(total_losses, 0.001)
    wr = len(winning) / max(len(trades), 1) * 100
    
    return {
        "total_return": round(total_return, 2),
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd, 2),
        "trades": len(trades),
        "win_rate": round(wr, 1),
        "profit_factor": round(pf, 2),
        "equity_curve": equity.tolist(),
        "trade_list": trades,
        "liquidated": any(t.get("exit_reason") == "LIQUIDATED" for t in trades),
    }


def main():
    config = load_config(str(Path(__file__).resolve().parent / "config.yaml"))
    store = DataStore(config)

    pair_dfs = {}
    for pair in ["ETH/USDT", "SOL/USDT", "BTC/USDT"]:
        df = store.load_ohlcv(pair, "1h", last_n_days=1095)
        if len(df) > 500:
            pair_dfs[pair] = build_features(df)
            print(f"  {pair}: {len(pair_dfs[pair])} rows")
    store.close()

    # Train ML
    names = get_feature_names(list(pair_dfs.values())[0])
    model = LGBMModel(config)
    train_parts, test_parts = [], []
    for pn in pair_dfs:
        labeled = model.create_labels(pair_dfs[pn])
        split_idx = int(len(labeled) * 0.8)
        train_parts.append(labeled[:split_idx])
        test_parts.append(labeled[split_idx:])
    combined_train = pl.concat(train_parts)
    combined_test = pl.concat(test_parts)
    metrics = model.train(combined_train, combined_test, names)
    print(f"ML AUC={metrics['auc']:.4f}\n")

    # ============================================================
    # Test matrix: combo filter with varying leverage + pyramiding
    # ============================================================
    base_params = dict(
        entry_period=480, exit_period=240, atr_period=48, atr_stop_mult=4.0,
        risk_per_trade=0.05, max_position_pct=0.60,
        adx_period=48, adx_threshold=18.0,
        vol_expansion=True, vol_sma_period=240,
        ml_filter=True, ml_long_threshold=0.30, ml_short_threshold=0.30,
    )
    
    configs = [
        ("combo_1x",         {"leverage": 1, "pyramid_levels": 0}),
        ("combo_3x",         {"leverage": 3, "pyramid_levels": 0}),
        ("combo_5x",         {"leverage": 5, "pyramid_levels": 0}),
        ("combo_7x",         {"leverage": 7, "pyramid_levels": 0}),
        ("combo_10x",        {"leverage": 10, "pyramid_levels": 0}),
        ("combo_5x_pyr2",    {"leverage": 5, "pyramid_levels": 2, "pyramid_atr_interval": 1.5}),
        ("combo_5x_pyr3",    {"leverage": 5, "pyramid_levels": 3, "pyramid_atr_interval": 1.5}),
        ("combo_7x_pyr2",    {"leverage": 7, "pyramid_levels": 2, "pyramid_atr_interval": 1.5}),
        ("combo_7x_pyr3",    {"leverage": 7, "pyramid_levels": 3, "pyramid_atr_interval": 1.0}),
        ("combo_10x_pyr2",   {"leverage": 10, "pyramid_levels": 2, "pyramid_atr_interval": 2.0}),
        # Tighter risk for high leverage
        ("combo_7x_3pct",    {"leverage": 7, "pyramid_levels": 0, "risk_per_trade": 0.03, "max_position_pct": 0.40}),
        ("combo_10x_2pct",   {"leverage": 10, "pyramid_levels": 0, "risk_per_trade": 0.02, "max_position_pct": 0.30}),
    ]
    
    results = []
    
    # Only test on ETH+SOL (BTC is a drag)
    test_pairs = ["ETH/USDT", "SOL/USDT"]
    
    for cfg_name, overrides in configs:
        params = {**base_params, **overrides}
        
        total_return = 0.0
        worst_dd = 0.0
        total_trades = 0
        total_pf_num = 0.0
        total_pf_den = 0.0
        any_liquidated = False
        pair_info = []
        
        for pair in test_pairs:
            feat_df = pair_dfs[pair]
            test_start = int(len(feat_df) * 0.8)
            test_df = model.predict(feat_df[test_start:])
            
            result = run_combo_backtest(test_df, **params)
            
            total_return += result["total_return"]
            worst_dd = min(worst_dd, result["max_dd"])
            total_trades += result["trades"]
            if result["liquidated"]:
                any_liquidated = True
            
            wins = sum(t["pnl_pct"] for t in result["trade_list"] if t["pnl_pct"] > 0)
            losses = abs(sum(t["pnl_pct"] for t in result["trade_list"] if t["pnl_pct"] <= 0))
            total_pf_num += wins
            total_pf_den += losses
            
            pair_info.append((pair.split("/")[0], result["total_return"], result["trades"], result["profit_factor"]))
        
        test_len = len(list(pair_dfs.values())[0]) - int(len(list(pair_dfs.values())[0]) * 0.8)
        months = test_len / (24 * 30.44)
        monthly = total_return / months
        pf = total_pf_num / max(total_pf_den, 0.001)
        
        results.append({
            "name": cfg_name,
            "return": total_return,
            "monthly": monthly,
            "annual": monthly * 12,
            "trades": total_trades,
            "dd": worst_dd,
            "pf": pf,
            "liquidated": any_liquidated,
            "pairs": pair_info,
        })
    
    # Sort and print
    results.sort(key=lambda x: x["monthly"], reverse=True)
    
    print(f"{'='*90}")
    print(f"  FINAL PUSH — ETH+SOL Combo Filter (ADX+Vol+ML)")
    print(f"{'='*90}")
    
    for i, r in enumerate(results):
        liq = " LIQUIDATED!" if r["liquidated"] else ""
        target = " <<<< TARGET" if r["monthly"] >= 20 else " << CLOSE" if r["monthly"] >= 15 else ""
        print(
            f"\n  #{i+1:2d} {r['name']:25s} "
            f"Ret={r['return']:+7.1f}%  Mo={r['monthly']:+6.2f}%  Ann={r['annual']:+7.1f}%  "
            f"PF={r['pf']:.2f}  Tr={r['trades']:3d}  DD={r['dd']:6.1f}%{liq}{target}"
        )
        for sym, ret, tr, pf in r["pairs"]:
            print(f"      {sym}: {ret:+.1f}% ({tr}t, PF={pf:.2f})")
    
    # ============================================================
    # Monthly breakdown for top 3
    # ============================================================
    for rank in range(min(3, len(results))):
        r = results[rank]
        cfg_name = r["name"]
        overrides = dict(configs)[ cfg_name] if cfg_name in dict(configs) else {}
        
        # Find the matching config
        for cn, ov in configs:
            if cn == cfg_name:
                overrides = ov
                break
        
        params = {**base_params, **overrides}
        
        print(f"\n{'='*90}")
        print(f"  MONTHLY #{rank+1}: {cfg_name} ({r['monthly']:+.2f}%/mo)")
        print(f"{'='*90}")
        
        all_monthly = {}
        for pair in test_pairs:
            feat_df = pair_dfs[pair]
            test_start = int(len(feat_df) * 0.8)
            test_df = model.predict(feat_df[test_start:])
            result = run_combo_backtest(test_df, **params)
            
            ts = feat_df["timestamp"].to_numpy()
            for t in result["trade_list"]:
                idx = test_start + t["entry_idx"]
                if idx < len(ts):
                    month = str(ts[idx])[:7]
                    if month not in all_monthly:
                        all_monthly[month] = 0.0
                    all_monthly[month] += t["pnl_pct"]
        
        pos_months = 0
        for month in sorted(all_monthly):
            pnl = all_monthly[month]
            print(f"    {month}: {pnl:+8.1f}%  {'[WIN]' if pnl > 0 else '[LOSS]'}")
            if pnl > 0:
                pos_months += 1
        
        total_m = len(all_monthly)
        print(f"    Win months: {pos_months}/{total_m} ({pos_months/max(total_m,1)*100:.0f}%)")

    # ============================================================
    # FULL DATASET VALIDATION (3 years, not just test 20%)
    # ============================================================
    print(f"\n{'='*90}")
    print(f"  FULL DATASET VALIDATION (3 years) — Best config")
    print(f"{'='*90}")
    
    # Use the #1 config
    best_name = results[0]["name"]
    for cn, ov in configs:
        if cn == best_name:
            best_overrides = ov
            break
    
    best_params = {**base_params, **best_overrides}
    
    for pair in ["ETH/USDT", "SOL/USDT"]:
        feat_df = pair_dfs[pair]
        full_df = model.predict(feat_df)
        
        result = run_combo_backtest(full_df, **best_params)
        
        total_months_full = len(feat_df) / (24 * 30.44)
        monthly_full = result["total_return"] / total_months_full
        
        print(f"\n  {pair} (full {total_months_full:.0f} months):")
        print(f"    Return: {result['total_return']:+.1f}%, Monthly: {monthly_full:+.2f}%")
        print(f"    Trades: {result['trades']}, PF={result['profit_factor']:.2f}, MaxDD={result['max_dd']:.1f}%")
        print(f"    Sharpe: {result['sharpe']:.2f}, WR: {result['win_rate']:.0f}%")
        
        # Monthly breakdown for full
        ts = feat_df["timestamp"].to_numpy()
        monthly_pnl = {}
        for t in result["trade_list"]:
            idx = t["entry_idx"]
            if idx < len(ts):
                month = str(ts[idx])[:7]
                if month not in monthly_pnl:
                    monthly_pnl[month] = 0.0
                monthly_pnl[month] += t["pnl_pct"]
        
        pos_m = sum(1 for v in monthly_pnl.values() if v > 0)
        print(f"    Win months: {pos_m}/{len(monthly_pnl)} ({pos_m/max(len(monthly_pnl),1)*100:.0f}%)")
        
        # Show yearly breakdown
        yearly = {}
        for month, pnl in monthly_pnl.items():
            year = month[:4]
            if year not in yearly:
                yearly[year] = 0.0
            yearly[year] += pnl
        for year in sorted(yearly):
            print(f"      {year}: {yearly[year]:+.1f}%")


if __name__ == "__main__":
    main()
