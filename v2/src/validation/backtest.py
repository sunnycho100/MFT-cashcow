"""Backtester — event-driven with ATR exits, trailing stop, Kelly sizing."""

from __future__ import annotations

import numpy as np
import polars as pl

from ..utils.logger import get_logger

logger = get_logger("v2.validation.backtest")


def run_backtest(
    df: pl.DataFrame,
    initial_capital: float = 100_000.0,
    fee_rate: float = 0.001,            # 10 bps taker
    slippage_bps: float = 5.0,          # 5 bps slippage
    max_position_pct: float = 0.10,     # base position size
    confidence_threshold: float = 0.60,  # min prob to trade
    # ATR-based exit parameters
    tp_atr_mult: float = 3.0,           # take-profit = 3 × ATR
    sl_atr_mult: float = 2.0,           # stop-loss = 2 × ATR
    max_hold_bars: int = 12,            # time exit after N bars
    trailing_activate_atr: float = 1.5,  # activate trailing stop after +1.5 ATR
    trailing_distance_atr: float = 1.0,  # trail at 1.0 ATR below peak
    kelly_fraction: float = 0.25,       # fraction of Kelly for position sizing
    long_only: bool = False,            # if True, only take LONG signals
) -> dict:
    """Event-driven backtest with proper exit discipline.

    Exit hierarchy (checked every bar while in position):
      1. Stop-loss hit          → immediate exit
      2. Take-profit hit        → immediate exit
      3. Trailing stop hit      → immediate exit (after activation)
      4. Time exit              → exit after max_hold_bars

    Position sizing: Kelly-scaled by confidence
      size = base × kelly_frac × (confidence - threshold) / (1 - threshold)

    Required columns: close, atr_14, pred_class, pred_prob_up, pred_prob_down
    """
    close = df["close"].to_numpy()
    pred = df["pred_class"].to_numpy()
    prob_up = df["pred_prob_up"].to_numpy()
    prob_down = df["pred_prob_down"].to_numpy()

    # ATR for dynamic exits
    if "atr_14" in df.columns:
        atr = df["atr_14"].to_numpy()
    else:
        # Fallback: use a fixed percentage of close
        atr = close * 0.02
        logger.warning("atr_14 not in backtest df, using 2% of close as fallback")

    n = len(close)
    equity = np.full(n, initial_capital)
    position = np.zeros(n)          # +1 long, -1 short, 0 flat
    trades: list[dict] = []

    # Active trade state
    in_trade = False
    trade_side = 0.0
    entry_price = 0.0
    entry_idx = 0
    pos_size_dollars = 0.0
    tp_price = 0.0
    sl_price = 0.0
    trailing_active = False
    trailing_peak = 0.0
    trailing_stop_price = 0.0
    entry_atr = 0.0

    def _close_trade(i: int, exit_reason: str):
        nonlocal in_trade, trade_side, entry_price, pos_size_dollars
        nonlocal trailing_active, trailing_peak, trailing_stop_price

        slip = slippage_bps / 10_000
        if trade_side > 0:
            exit_price = close[i] * (1 - slip)
            pnl_pct = (exit_price / entry_price - 1)
        else:
            exit_price = close[i] * (1 + slip)
            pnl_pct = (entry_price / exit_price - 1)

        fee = fee_rate * 2  # entry + exit
        net_pnl_pct = pnl_pct - fee
        equity[i] += pos_size_dollars * net_pnl_pct

        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": i,
            "side": "LONG" if trade_side > 0 else "SHORT",
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl_pct": round(net_pnl_pct * 100, 4),
            "equity_after": round(equity[i], 2),
            "exit_reason": exit_reason,
            "hold_bars": i - entry_idx,
        })

        in_trade = False
        trade_side = 0.0
        trailing_active = False

    def _open_trade(i: int, side: float, confidence: float):
        nonlocal in_trade, trade_side, entry_price, entry_idx, pos_size_dollars
        nonlocal tp_price, sl_price, trailing_active, trailing_peak, trailing_stop_price
        nonlocal entry_atr

        slip = slippage_bps / 10_000
        entry_price = close[i] * (1 + slip * side)
        entry_idx = i
        trade_side = side
        entry_atr = atr[i] if not np.isnan(atr[i]) else close[i] * 0.02

        # Kelly-scaled position sizing
        edge = (confidence - confidence_threshold) / (1 - confidence_threshold)
        size_mult = kelly_fraction * max(edge, 0.0)
        pos_size_dollars = equity[i] * max_position_pct * (1 + size_mult)

        # Set ATR-based exit levels
        if side > 0:  # long
            tp_price = entry_price + tp_atr_mult * entry_atr
            sl_price = entry_price - sl_atr_mult * entry_atr
        else:         # short
            tp_price = entry_price - tp_atr_mult * entry_atr
            sl_price = entry_price + sl_atr_mult * entry_atr

        trailing_active = False
        trailing_peak = entry_price
        trailing_stop_price = 0.0
        in_trade = True

    for i in range(1, n):
        equity[i] = equity[i - 1]

        # --- If in a trade, check exits first ---
        if in_trade:
            bars_held = i - entry_idx

            if trade_side > 0:  # LONG position
                # 1. Stop-loss
                if close[i] <= sl_price:
                    _close_trade(i, "stop_loss")
                    position[i] = 0.0
                    continue
                # 2. Take-profit
                if close[i] >= tp_price:
                    _close_trade(i, "take_profit")
                    position[i] = 0.0
                    continue
                # 3. Trailing stop (disabled if trailing_activate_atr <= 0)
                if trailing_activate_atr > 0:
                    if close[i] > entry_price + trailing_activate_atr * entry_atr:
                        trailing_active = True
                    if trailing_active:
                        trailing_peak = max(trailing_peak, close[i])
                        trailing_stop_price = trailing_peak - trailing_distance_atr * entry_atr
                        if close[i] <= trailing_stop_price:
                            _close_trade(i, "trailing_stop")
                            position[i] = 0.0
                            continue

            else:  # SHORT position
                # 1. Stop-loss
                if close[i] >= sl_price:
                    _close_trade(i, "stop_loss")
                    position[i] = 0.0
                    continue
                # 2. Take-profit
                if close[i] <= tp_price:
                    _close_trade(i, "take_profit")
                    position[i] = 0.0
                    continue
                # 3. Trailing stop (disabled if trailing_activate_atr <= 0)
                if trailing_activate_atr > 0:
                    if close[i] < entry_price - trailing_activate_atr * entry_atr:
                        trailing_active = True
                    if trailing_active:
                        trailing_peak = min(trailing_peak, close[i])
                        trailing_stop_price = trailing_peak + trailing_distance_atr * entry_atr
                        if close[i] >= trailing_stop_price:
                            _close_trade(i, "trailing_stop")
                            position[i] = 0.0
                            continue

            # 4. Time exit
            if bars_held >= max_hold_bars:
                _close_trade(i, "time_exit")
                position[i] = 0.0
                continue

            # Still in trade — carry position
            position[i] = trade_side

        # --- Entry logic (only if flat) ---
        if not in_trade:
            conf_up = prob_up[i]
            conf_down = prob_down[i]

            if pred[i] == 2 and conf_up >= confidence_threshold:
                _open_trade(i, 1.0, conf_up)
                position[i] = 1.0
            elif not long_only and pred[i] == 0 and conf_down >= confidence_threshold:
                _open_trade(i, -1.0, conf_down)
                position[i] = -1.0
            else:
                position[i] = 0.0

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
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
    profit_factor = abs(sum(t["pnl_pct"] for t in winning)) / max(abs(sum(t["pnl_pct"] for t in losing)), 0.001)

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        reason = t.get("exit_reason", "unknown")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    # Average hold time
    avg_hold = np.mean([t["hold_bars"] for t in trades]) if trades else 0

    bnh_return = (close[-1] / close[0] - 1) * 100

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
    }

    logger.info(
        f"Backtest: {metrics['total_return_pct']:+.1f}% return, "
        f"{metrics['total_trades']} trades, "
        f"{metrics['win_rate_pct']:.0f}% win rate, "
        f"Sharpe {metrics['sharpe_ratio']:.2f}, "
        f"PF {metrics['profit_factor']:.2f}, "
        f"exits={exit_reasons}"
    )

    return {
        "metrics": metrics,
        "equity_curve": equity.tolist(),
        "trades": trades,
    }
