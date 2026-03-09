#!/usr/bin/env python3
"""Strategic optimizer — regime filtering + selective pairs + leverage.

Key findings so far:
  - Original 3 pairs (BTC/ETH/SOL) = +21.4%, +2.98%/month
  - BTC contributes almost nothing (+0.4%), ETH+SOL = +21.0%
  - New pairs (DOGE, LINK, ADA, DOT) all LOSE money
  - Choppy months (Aug, Oct, Dec, Mar) kill returns
  - Trending months (Nov, Jan) make all the money

New approaches:
  1. Drop BTC → just trade ETH+SOL (removes drag)
  2. ADX regime filter → only enter trades in trending markets
  3. Volatility expansion filter → only trade after squeeze
  4. Selective leverage on high-conviction pairs
  5. Dynamic pair selection by ADX ranking
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


def run_regime_filtered_backtest(
    df: pl.DataFrame,
    # Donchian params
    entry_period: int = 480,
    exit_period: int = 240,
    atr_period: int = 48,
    atr_stop_mult: float = 4.0,
    risk_per_trade: float = 0.05,
    max_position_pct: float = 0.60,
    initial_capital: float = 100_000.0,
    fee_rate: float = 0.001,
    slippage_bps: float = 5.0,
    # Regime filter
    adx_period: int = 48,
    adx_threshold: float = 0.0,   # 0 = disabled; 20-25 = moderate filter
    vol_expansion: bool = False,   # require ATR > SMA(ATR)
    vol_sma_period: int = 240,     # 10-day SMA of ATR
    # ML filter
    ml_filter: bool = False,
    ml_long_threshold: float = 0.30,
    ml_short_threshold: float = 0.30,
) -> dict:
    """Donchian breakout with optional regime (ADX + vol expansion) filter."""
    
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    
    if ml_filter and "pred_long_prob" in df.columns:
        ml_prob = df["pred_long_prob"].to_numpy()
    else:
        ml_prob = np.full(len(close), 0.5)
    
    atr = talib.ATR(high, low, close, timeperiod=atr_period)
    
    # ADX for regime detection
    adx = talib.ADX(high, low, close, timeperiod=adx_period)
    
    # ATR SMA for volatility expansion detection
    atr_sma = np.full(len(close), np.nan)
    if vol_expansion:
        for i in range(vol_sma_period, len(close)):
            atr_vals = atr[i - vol_sma_period:i]
            valid = atr_vals[~np.isnan(atr_vals)]
            if len(valid) > 0:
                atr_sma[i] = np.mean(valid)
    
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
    
    position = 0.0
    entry_price = 0.0
    entry_idx = 0
    pos_size_units = 0.0
    trail_stop = 0.0
    trail_peak = 0.0
    
    entries_filtered_adx = 0
    entries_filtered_vol = 0
    entries_filtered_ml = 0
    entries_total = 0
    
    warmup = max(entry_period, exit_period, atr_period, adx_period, vol_sma_period if vol_expansion else 0) + 1
    
    for i in range(1, n):
        equity[i] = equity[i - 1]
        
        if i < warmup or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        
        slip = slippage_bps / 10_000
        current_atr = atr[i]
        
        # ---- EXIT LOGIC ----
        if position != 0.0:
            should_exit = False
            exit_reason = ""
            
            if position > 0:
                trail_peak = max(trail_peak, close[i])
                trail_stop = max(trail_stop, trail_peak - atr_stop_mult * current_atr)
                
                if close[i] <= trail_stop:
                    should_exit, exit_reason = True, "trailing_stop"
                elif close[i] < exit_lower[i] if not np.isnan(exit_lower[i]) else False:
                    should_exit, exit_reason = True, "exit_channel"
                elif close[i] < entry_lower[i] if not np.isnan(entry_lower[i]) else False:
                    should_exit, exit_reason = True, "reversal_signal"
            else:
                trail_peak = min(trail_peak, close[i])
                trail_stop = min(trail_stop, trail_peak + atr_stop_mult * current_atr)
                
                if close[i] >= trail_stop:
                    should_exit, exit_reason = True, "trailing_stop"
                elif close[i] > exit_upper[i] if not np.isnan(exit_upper[i]) else False:
                    should_exit, exit_reason = True, "exit_channel"
                elif close[i] > entry_upper[i] if not np.isnan(entry_upper[i]) else False:
                    should_exit, exit_reason = True, "reversal_signal"
            
            if should_exit:
                if position > 0:
                    exit_price = close[i] * (1 - slip)
                    pnl = pos_size_units * (exit_price - entry_price)
                else:
                    exit_price = close[i] * (1 + slip)
                    pnl = pos_size_units * (entry_price - exit_price)
                
                fee_cost = fee_rate * pos_size_units * (entry_price + exit_price)
                net_pnl = pnl - fee_cost
                pnl_pct = net_pnl / equity[i - 1] * 100
                equity[i] += net_pnl
                
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "side": "LONG" if position > 0 else "SHORT",
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "pnl_pct": round(pnl_pct, 4),
                    "equity_after": round(equity[i], 2),
                    "exit_reason": exit_reason,
                    "hold_bars": i - entry_idx,
                })
                
                if exit_reason == "reversal_signal":
                    new_side = -position
                    position = 0.0
                    
                    # Check regime filters for re-entry
                    regime_ok = True
                    if adx_threshold > 0 and not np.isnan(adx[i]) and adx[i] < adx_threshold:
                        regime_ok = False
                    if vol_expansion and not np.isnan(atr_sma[i]) and atr[i] < atr_sma[i]:
                        regime_ok = False
                    
                    if regime_ok:
                        risk_dollars = equity[i] * risk_per_trade
                        stop_distance = atr_stop_mult * current_atr
                        trade_size = risk_dollars / stop_distance
                        max_size = equity[i] * max_position_pct / close[i]
                        pos_size_units = min(trade_size, max_size)
                        
                        entry_price = close[i] * (1 + slip * new_side)
                        entry_idx = i
                        position = new_side
                        
                        if new_side > 0:
                            trail_peak = close[i]
                            trail_stop = trail_peak - atr_stop_mult * current_atr
                        else:
                            trail_peak = close[i]
                            trail_stop = trail_peak + atr_stop_mult * current_atr
                    continue
                else:
                    position = 0.0
                    pos_size_units = 0.0
        
        # ---- ENTRY LOGIC ----
        if position == 0.0:
            long_signal = close[i] > entry_upper[i] if not np.isnan(entry_upper[i]) else False
            short_signal = close[i] < entry_lower[i] if not np.isnan(entry_lower[i]) else False
            
            if long_signal or short_signal:
                entries_total += 1
                
                # ADX regime filter
                if adx_threshold > 0:
                    if np.isnan(adx[i]) or adx[i] < adx_threshold:
                        entries_filtered_adx += 1
                        long_signal = False
                        short_signal = False
                
                # Volatility expansion filter
                if vol_expansion and (long_signal or short_signal):
                    if not np.isnan(atr_sma[i]) and atr[i] < atr_sma[i]:
                        entries_filtered_vol += 1
                        long_signal = False
                        short_signal = False
                
                # ML filter
                if ml_filter and (long_signal or short_signal):
                    if long_signal and ml_prob[i] < ml_long_threshold:
                        entries_filtered_ml += 1
                        long_signal = False
                    if short_signal and (1.0 - ml_prob[i]) < ml_short_threshold:
                        entries_filtered_ml += 1
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
                pos_size_units = min(trade_size, max_size)
                
                entry_price = close[i] * (1 + slip * side)
                entry_idx = i
                position = side
                
                if side > 0:
                    trail_peak = close[i]
                    trail_stop = trail_peak - atr_stop_mult * current_atr
                else:
                    trail_peak = close[i]
                    trail_stop = trail_peak + atr_stop_mult * current_atr
    
    # Close open position
    if position != 0.0:
        slip = slippage_bps / 10_000
        if position > 0:
            exit_price = close[-1] * (1 - slip)
            pnl = pos_size_units * (exit_price - entry_price)
        else:
            exit_price = close[-1] * (1 + slip)
            pnl = pos_size_units * (entry_price - exit_price)
        
        fee_cost = fee_rate * pos_size_units * (entry_price + exit_price)
        net_pnl = pnl - fee_cost
        pnl_pct = net_pnl / equity[-2] * 100
        equity[-1] += net_pnl
        
        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": n - 1,
            "side": "LONG" if position > 0 else "SHORT",
            "entry_price": round(entry_price, 4),
            "exit_price": round(close[-1], 4),
            "pnl_pct": round(pnl_pct, 4),
            "equity_after": round(equity[-1], 2),
            "exit_reason": "end_of_data",
            "hold_bars": n - 1 - entry_idx,
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
    
    return {
        "metrics": {
            "total_return_pct": round(total_return, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "total_trades": len(trades),
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "long_pnl": round(sum(t["pnl_pct"] for t in trades if t["side"] == "LONG"), 2),
            "short_pnl": round(sum(t["pnl_pct"] for t in trades if t["side"] == "SHORT"), 2),
        },
        "equity_curve": equity.tolist(),
        "trades": trades,
        "filter_stats": {
            "total_signals": entries_total,
            "filtered_adx": entries_filtered_adx,
            "filtered_vol": entries_filtered_vol,
            "filtered_ml": entries_filtered_ml,
        },
    }


def main():
    config = load_config(str(Path(__file__).resolve().parent / "config.yaml"))
    store = DataStore(config)

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
        except:
            pass
    store.close()
    
    print(f"Loaded {len(pair_dfs)} pairs")

    # Train ML on original 3 pairs
    names = get_feature_names(list(pair_dfs.values())[0])
    model = LGBMModel(config)
    
    train_parts, test_parts = [], []
    for pn in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
        if pn in pair_dfs:
            labeled = model.create_labels(pair_dfs[pn])
            split_idx = int(len(labeled) * 0.8)
            train_parts.append(labeled[:split_idx])
            test_parts.append(labeled[split_idx:])
    
    combined_train = pl.concat(train_parts)
    combined_test = pl.concat(test_parts)
    metrics = model.train(combined_train, combined_test, names)
    print(f"ML AUC={metrics['auc']:.4f}")

    # ============================================================
    # Configuration grid
    # ============================================================
    configs = {
        # Baseline: original 3 pairs, ML filter only
        "01_baseline_3pair_ml": {
            "pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            "leverage": 1.0,
            "params": {
                "entry_period": 480, "exit_period": 240, "atr_period": 48,
                "atr_stop_mult": 4.0, "risk_per_trade": 0.05,
                "max_position_pct": 0.60,
                "ml_filter": True, "ml_long_threshold": 0.30, "ml_short_threshold": 0.30,
            },
        },
        # Drop BTC — it's a drag
        "02_ETH_SOL_only_1x": {
            "pairs": ["ETH/USDT", "SOL/USDT"],
            "leverage": 1.0,
            "params": {
                "entry_period": 480, "exit_period": 240, "atr_period": 48,
                "atr_stop_mult": 4.0, "risk_per_trade": 0.05,
                "max_position_pct": 0.60,
                "ml_filter": True, "ml_long_threshold": 0.30, "ml_short_threshold": 0.30,
            },
        },
        "03_ETH_SOL_3x": {
            "pairs": ["ETH/USDT", "SOL/USDT"],
            "leverage": 3.0,
            "params": {
                "entry_period": 480, "exit_period": 240, "atr_period": 48,
                "atr_stop_mult": 4.0, "risk_per_trade": 0.05,
                "max_position_pct": 0.60,
                "ml_filter": True, "ml_long_threshold": 0.30, "ml_short_threshold": 0.30,
            },
        },
        "04_ETH_SOL_5x": {
            "pairs": ["ETH/USDT", "SOL/USDT"],
            "leverage": 5.0,
            "params": {
                "entry_period": 480, "exit_period": 240, "atr_period": 48,
                "atr_stop_mult": 4.0, "risk_per_trade": 0.05,
                "max_position_pct": 0.60,
                "ml_filter": True, "ml_long_threshold": 0.30, "ml_short_threshold": 0.30,
            },
        },
        # ADX filter — only trade when pair is trending (ADX > 20)
        "05_3pair_adx20_ml": {
            "pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            "leverage": 1.0,
            "params": {
                "entry_period": 480, "exit_period": 240, "atr_period": 48,
                "atr_stop_mult": 4.0, "risk_per_trade": 0.05,
                "max_position_pct": 0.60,
                "adx_threshold": 20.0, "adx_period": 48,
                "ml_filter": True, "ml_long_threshold": 0.30, "ml_short_threshold": 0.30,
            },
        },
        "06_3pair_adx20_ml_3x": {
            "pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            "leverage": 3.0,
            "params": {
                "entry_period": 480, "exit_period": 240, "atr_period": 48,
                "atr_stop_mult": 4.0, "risk_per_trade": 0.05,
                "max_position_pct": 0.60,
                "adx_threshold": 20.0, "adx_period": 48,
                "ml_filter": True, "ml_long_threshold": 0.30, "ml_short_threshold": 0.30,
            },
        },
        # Volatility expansion filter
        "07_3pair_volexp_ml": {
            "pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            "leverage": 1.0,
            "params": {
                "entry_period": 480, "exit_period": 240, "atr_period": 48,
                "atr_stop_mult": 4.0, "risk_per_trade": 0.05,
                "max_position_pct": 0.60,
                "vol_expansion": True, "vol_sma_period": 240,
                "ml_filter": True, "ml_long_threshold": 0.30, "ml_short_threshold": 0.30,
            },
        },
        "08_3pair_volexp_ml_3x": {
            "pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            "leverage": 3.0,
            "params": {
                "entry_period": 480, "exit_period": 240, "atr_period": 48,
                "atr_stop_mult": 4.0, "risk_per_trade": 0.05,
                "max_position_pct": 0.60,
                "vol_expansion": True, "vol_sma_period": 240,
                "ml_filter": True, "ml_long_threshold": 0.30, "ml_short_threshold": 0.30,
            },
        },
        # Combo: ADX + vol expansion + ML — maximum filtering
        "09_3pair_combo_filter": {
            "pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            "leverage": 1.0,
            "params": {
                "entry_period": 480, "exit_period": 240, "atr_period": 48,
                "atr_stop_mult": 4.0, "risk_per_trade": 0.05,
                "max_position_pct": 0.60,
                "adx_threshold": 18.0, "adx_period": 48,
                "vol_expansion": True, "vol_sma_period": 240,
                "ml_filter": True, "ml_long_threshold": 0.30, "ml_short_threshold": 0.30,
            },
        },
        "10_3pair_combo_filter_3x": {
            "pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            "leverage": 3.0,
            "params": {
                "entry_period": 480, "exit_period": 240, "atr_period": 48,
                "atr_stop_mult": 4.0, "risk_per_trade": 0.05,
                "max_position_pct": 0.60,
                "adx_threshold": 18.0, "adx_period": 48,
                "vol_expansion": True, "vol_sma_period": 240,
                "ml_filter": True, "ml_long_threshold": 0.30, "ml_short_threshold": 0.30,
            },
        },
        # ETH+SOL combo filters with leverage
        "11_ETHSOL_combo_5x": {
            "pairs": ["ETH/USDT", "SOL/USDT"],
            "leverage": 5.0,
            "params": {
                "entry_period": 480, "exit_period": 240, "atr_period": 48,
                "atr_stop_mult": 4.0, "risk_per_trade": 0.05,
                "max_position_pct": 0.60,
                "adx_threshold": 18.0, "adx_period": 48,
                "vol_expansion": True, "vol_sma_period": 240,
                "ml_filter": True, "ml_long_threshold": 0.30, "ml_short_threshold": 0.30,
            },
        },
        # Best 4 pairs only (exclude DOGE, LINK)
        "12_best4_3x": {
            "pairs": ["ETH/USDT", "SOL/USDT", "AVAX/USDT", "BTC/USDT"],
            "leverage": 3.0,
            "params": {
                "entry_period": 480, "exit_period": 240, "atr_period": 48,
                "atr_stop_mult": 4.0, "risk_per_trade": 0.05,
                "max_position_pct": 0.60,
                "adx_threshold": 18.0, "adx_period": 48,
                "ml_filter": True, "ml_long_threshold": 0.30, "ml_short_threshold": 0.30,
            },
        },
        # Higher risk per trade on ETH+SOL
        "13_ETHSOL_8pct_risk_3x": {
            "pairs": ["ETH/USDT", "SOL/USDT"],
            "leverage": 3.0,
            "params": {
                "entry_period": 480, "exit_period": 240, "atr_period": 48,
                "atr_stop_mult": 4.0, "risk_per_trade": 0.08,
                "max_position_pct": 0.70,
                "ml_filter": True, "ml_long_threshold": 0.30, "ml_short_threshold": 0.30,
            },
        },
        "14_ETHSOL_8pct_risk_5x": {
            "pairs": ["ETH/USDT", "SOL/USDT"],
            "leverage": 5.0,
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
    results = []
    
    for cfg_name, cfg in configs.items():
        leverage = cfg["leverage"]
        total_return = 0.0
        total_trades = 0
        worst_dd = 0.0
        total_pf_num = 0.0
        total_pf_den = 0.0
        pair_details = []
        filter_stats_all = {"total_signals": 0, "filtered_adx": 0, "filtered_vol": 0, "filtered_ml": 0}
        
        for pair_name in cfg["pairs"]:
            if pair_name not in pair_dfs:
                continue
            
            feat_df = pair_dfs[pair_name]
            test_start = int(len(feat_df) * 0.8)
            test_df = feat_df[test_start:]
            
            if cfg["params"].get("ml_filter", False):
                test_df = model.predict(test_df)
            
            result = run_regime_filtered_backtest(test_df, **cfg["params"])
            m = result["metrics"]
            
            lev_return = m["total_return_pct"] * leverage
            lev_dd = m["max_drawdown_pct"] * leverage
            
            total_return += lev_return
            total_trades += m["total_trades"]
            worst_dd = min(worst_dd, lev_dd)
            
            # Aggregate PF
            wins = sum(t["pnl_pct"] for t in result["trades"] if t["pnl_pct"] > 0)
            losses = abs(sum(t["pnl_pct"] for t in result["trades"] if t["pnl_pct"] <= 0))
            total_pf_num += wins
            total_pf_den += losses
            
            pair_details.append((pair_name.split("/")[0], lev_return, m["total_trades"], m["profit_factor"]))
            
            for k in filter_stats_all:
                filter_stats_all[k] += result["filter_stats"].get(k, 0)
        
        first_df = list(pair_dfs.values())[0]
        test_len = len(first_df) - int(len(first_df) * 0.8)
        months = test_len / (24 * 30.44)
        monthly = total_return / months
        portfolio_pf = total_pf_num / max(total_pf_den, 0.001)
        
        results.append({
            "name": cfg_name,
            "total_return": total_return,
            "monthly": monthly,
            "annualized": monthly * 12,
            "trades": total_trades,
            "worst_dd": worst_dd,
            "pf": portfolio_pf,
            "pairs": len(cfg["pairs"]),
            "leverage": leverage,
            "details": pair_details,
            "filters": filter_stats_all,
        })
    
    # ============================================================
    # Print ranked results
    # ============================================================
    results.sort(key=lambda x: x["monthly"], reverse=True)
    
    print(f"\n{'='*90}")
    print(f"  STRATEGIC OPTIMIZER — Ranked by monthly return")
    print(f"{'='*90}")
    
    for i, r in enumerate(results):
        print(
            f"\n  #{i+1:2d} {r['name']}"
            f"\n      Return: {r['total_return']:+.1f}% | Monthly: {r['monthly']:+.2f}% | "
            f"Annual: {r['annualized']:+.1f}% | PF={r['pf']:.2f} | "
            f"Trades: {r['trades']} | DD: {r['worst_dd']:.1f}% | "
            f"{r['pairs']}pairs {r['leverage']:.0f}x"
        )
        for sym, ret, trades, pf in r["details"]:
            print(f"        {sym}: {ret:+.1f}% ({trades}t, PF={pf:.2f})")
        
        fs = r["filters"]
        if fs["total_signals"] > 0:
            total_filtered = fs["filtered_adx"] + fs["filtered_vol"] + fs["filtered_ml"]
            print(f"        Filters: {total_filtered}/{fs['total_signals']} signals blocked "
                  f"(ADX={fs['filtered_adx']}, Vol={fs['filtered_vol']}, ML={fs['filtered_ml']})")

    # Monthly breakdown for top 3
    for rank in range(min(3, len(results))):
        r = results[rank]
        cfg = configs[r["name"]]
        leverage = cfg["leverage"]
        
        print(f"\n{'='*90}")
        print(f"  MONTHLY — #{rank+1} {r['name']} ({r['monthly']:+.2f}%/mo)")
        print(f"{'='*90}")
        
        all_monthly = {}
        for pair_name in cfg["pairs"]:
            if pair_name not in pair_dfs:
                continue
            feat_df = pair_dfs[pair_name]
            test_start = int(len(feat_df) * 0.8)
            test_df = feat_df[test_start:]
            
            if cfg["params"].get("ml_filter", False):
                test_df = model.predict(test_df)
            
            result = run_regime_filtered_backtest(test_df, **cfg["params"])
            
            ts = feat_df["timestamp"].to_numpy()
            for t in result["trades"]:
                idx = test_start + t["entry_idx"]
                if idx < len(ts):
                    month = str(ts[idx])[:7]
                    if month not in all_monthly:
                        all_monthly[month] = 0.0
                    all_monthly[month] += t["pnl_pct"] * leverage
        
        pos_months = 0
        for month in sorted(all_monthly):
            pnl = all_monthly[month]
            bar = "█" * max(int(abs(pnl) / 2), 1)
            sign = "+" if pnl >= 0 else "-"
            print(f"    {month}: {pnl:+7.1f}%  {'🟢' if pnl > 0 else '🔴'} {bar}")
            if pnl > 0:
                pos_months += 1
        
        total_months = len(all_monthly)
        print(f"\n    Win months: {pos_months}/{total_months} ({pos_months/max(total_months,1)*100:.0f}%)")


if __name__ == "__main__":
    main()
