"""Trend-Following Backtest Engine — Donchian Breakout + ATR Trailing Stop.

This is a proven strategy that works in both bull AND bear markets.
Makes money by following trends: long in uptrends, short in downtrends.

Strategy:
  Entry:
    - LONG when close breaks above 20-day high (480 1H bars)
    - SHORT when close breaks below 20-day low (480 1H bars)
  Exit:
    - ATR trailing stop (chandelier exit)
    - Opposite breakout signal (stop-and-reverse)
  Position Sizing:
    - Risk X% of equity per trade (volatility-adjusted)
    - size = risk_equity / (ATR * atr_stop_mult)
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..utils.logger import get_logger

logger = get_logger("v2.validation.trend_backtest")


def compute_donchian(high: np.ndarray, low: np.ndarray, period: int):
    """Compute Donchian channels (rolling highest high / lowest low)."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period, n):
        upper[i] = np.max(high[i - period:i])  # exclude current bar
        lower[i] = np.min(low[i - period:i])   # exclude current bar
    
    return upper, lower


def run_trend_backtest(
    df: pl.DataFrame,
    # Donchian params
    entry_period: int = 480,           # 20 days of 1H bars
    exit_period: int = 240,            # 10 days of 1H bars (tighter exit)
    # Risk management
    atr_period: int = 48,              # 2-day ATR for smoother volatility
    atr_stop_mult: float = 3.0,        # trailing stop distance in ATR units
    risk_per_trade: float = 0.02,      # risk 2% of equity per trade
    max_position_pct: float = 0.40,    # max 40% of equity in one trade
    # Capital
    initial_capital: float = 100_000.0,
    fee_rate: float = 0.001,           # 10 bps per side
    slippage_bps: float = 5.0,         # 5 bps slippage
    # ML filter (optional)
    ml_filter: bool = False,           # if True, use pred_long_prob
    ml_long_threshold: float = 0.30,   # min prob for long confirmation
    ml_short_threshold: float = 0.30,  # max prob (1 - prob) for short confirmation
) -> dict:
    """Run Donchian breakout trend-following backtest.
    
    Position sizing: volatility-adjusted (risk X% of equity per trade)
      trade_size = (equity * risk_per_trade) / (ATR * atr_stop_mult)
      capped at max_position_pct of equity
    """
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    
    # ML filter probabilities
    if ml_filter and "pred_long_prob" in df.columns:
        ml_prob = df["pred_long_prob"].to_numpy()
    else:
        ml_prob = np.full(len(close), 0.5)  # neutral if no ML
    
    # Compute ATR
    import talib
    atr = talib.ATR(high, low, close, timeperiod=atr_period)
    
    # Compute Donchian channels
    entry_upper, entry_lower = compute_donchian(high, low, entry_period)
    exit_upper, exit_lower = compute_donchian(high, low, exit_period)
    
    n = len(close)
    equity = np.full(n, initial_capital)
    trades: list[dict] = []
    
    # Position state
    position = 0.0     # +1 long, -1 short, 0 flat
    entry_price = 0.0
    entry_idx = 0
    pos_size_units = 0.0  # number of units (coins)
    trail_stop = 0.0
    trail_peak = 0.0
    
    warmup = max(entry_period, exit_period, atr_period) + 1
    
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
            
            if position > 0:  # LONG
                # Update trailing stop
                trail_peak = max(trail_peak, close[i])
                trail_stop = trail_peak - atr_stop_mult * current_atr
                
                # Exit conditions
                if close[i] <= trail_stop:
                    should_exit = True
                    exit_reason = "trailing_stop"
                elif close[i] < exit_lower[i]:  # exit channel breakout (faster)
                    should_exit = True
                    exit_reason = "exit_channel"
                elif close[i] < entry_lower[i]:  # opposite signal → stop-and-reverse
                    should_exit = True
                    exit_reason = "reversal_signal"
                    
            elif position < 0:  # SHORT
                trail_peak = min(trail_peak, close[i])
                trail_stop = trail_peak + atr_stop_mult * current_atr
                
                if close[i] >= trail_stop:
                    should_exit = True
                    exit_reason = "trailing_stop"
                elif close[i] > exit_upper[i]:
                    should_exit = True
                    exit_reason = "exit_channel"
                elif close[i] > entry_upper[i]:
                    should_exit = True
                    exit_reason = "reversal_signal"
            
            if should_exit:
                # Close position
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
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "pnl_pct": round(pnl_pct, 4),
                    "equity_after": round(equity[i], 2),
                    "exit_reason": exit_reason,
                    "hold_bars": i - entry_idx,
                })
                
                # Check for stop-and-reverse
                if exit_reason == "reversal_signal":
                    new_side = -position  # reverse direction
                    position = 0.0  # temporarily flat
                    
                    # Open reverse position
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
        
        # ---- ENTRY LOGIC (only if flat) ----
        if position == 0.0:
            # Donchian breakout
            long_signal = close[i] > entry_upper[i]
            short_signal = close[i] < entry_lower[i]
            
            # ML filter
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
                # Position sizing: risk-based
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
    
    # Close any open position at end
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
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl_pct": round(pnl_pct, 4),
            "equity_after": round(equity[-1], 2),
            "exit_reason": "end_of_data",
            "hold_bars": n - 1 - entry_idx,
        })
    
    # ---- METRICS ----
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
    avg_win = np.mean([t["pnl_pct"] for t in winning]) if winning else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losing]) if losing else 0
    
    total_wins = sum(t["pnl_pct"] for t in winning)
    total_losses = abs(sum(t["pnl_pct"] for t in losing))
    profit_factor = total_wins / max(total_losses, 0.001)
    
    exit_reasons = {}
    for t in trades:
        r = t.get("exit_reason", "unknown")
        exit_reasons[r] = exit_reasons.get(r, 0) + 1
    
    avg_hold = np.mean([t["hold_bars"] for t in trades]) if trades else 0
    bnh_return = (close[-1] / close[0] - 1) * 100
    
    long_trades = [t for t in trades if t["side"] == "LONG"]
    short_trades = [t for t in trades if t["side"] == "SHORT"]
    long_pnl = sum(t["pnl_pct"] for t in long_trades)
    short_pnl = sum(t["pnl_pct"] for t in short_trades)
    
    metrics = {
        "total_return_pct": round(total_return, 2),
        "buy_hold_return_pct": round(bnh_return, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "total_trades": len(trades),
        "win_rate_pct": round(win_rate, 1),
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "profit_factor": round(profit_factor, 2),
        "final_equity": round(equity[-1], 2),
        "avg_hold_bars": round(avg_hold, 1),
        "exit_reasons": exit_reasons,
        "long_pnl_pct": round(long_pnl, 2),
        "short_pnl_pct": round(short_pnl, 2),
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
    }
    
    logger.info(
        f"Trend Backtest: {metrics['total_return_pct']:+.1f}% return, "
        f"{metrics['total_trades']} trades ({metrics['long_trades']}L/{metrics['short_trades']}S), "
        f"{metrics['win_rate_pct']:.0f}% WR, "
        f"Sharpe {metrics['sharpe_ratio']:.2f}, "
        f"PF {metrics['profit_factor']:.2f}, "
        f"Long PnL: {long_pnl:+.1f}%, Short PnL: {short_pnl:+.1f}%"
    )
    
    return {
        "metrics": metrics,
        "equity_curve": equity.tolist(),
        "trades": trades,
    }
