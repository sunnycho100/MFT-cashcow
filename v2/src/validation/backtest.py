"""Backtester — vectorized Polars-based backtest with realistic costs."""

from __future__ import annotations

import numpy as np
import polars as pl

from ..utils.logger import get_logger

logger = get_logger("v2.validation.backtest")


def run_backtest(
    df: pl.DataFrame,
    initial_capital: float = 100_000.0,
    fee_rate: float = 0.001,        # 10 bps taker
    slippage_bps: float = 5.0,      # 5 bps slippage
    max_position_pct: float = 0.10,
    confidence_threshold: float = 0.45,  # min prob to trade
) -> dict:
    """Vectorized backtest on a DataFrame with prediction columns.

    Required columns: close, pred_class, pred_prob_up, pred_prob_down

    Returns dict with equity curve, metrics, and trade log.
    """
    close = df["close"].to_numpy()
    pred = df["pred_class"].to_numpy()
    prob_up = df["pred_prob_up"].to_numpy()
    prob_down = df["pred_prob_down"].to_numpy()

    n = len(close)
    equity = np.full(n, initial_capital)
    position = np.zeros(n)  # +1 long, -1 short, 0 flat
    trades: list[dict] = []
    entry_price = 0.0
    entry_idx = 0

    for i in range(1, n):
        equity[i] = equity[i - 1]

        # Determine desired position
        conf = max(prob_up[i], prob_down[i])
        if pred[i] == 2 and prob_up[i] >= confidence_threshold:
            desired = 1.0   # long
        elif pred[i] == 0 and prob_down[i] >= confidence_threshold:
            desired = -1.0  # short
        else:
            desired = 0.0   # flat

        prev_pos = position[i - 1]

        # If position changed, execute trade
        if desired != prev_pos:
            # Close existing position
            if prev_pos != 0:
                exit_price = close[i] * (1 - slippage_bps / 10000 * abs(prev_pos))
                pnl_pct = prev_pos * (exit_price / entry_price - 1)
                fee = abs(prev_pos) * fee_rate * 2  # entry + exit fee
                net_pnl_pct = pnl_pct - fee
                pos_size = equity[i] * max_position_pct
                equity[i] += pos_size * net_pnl_pct

                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "side": "LONG" if prev_pos > 0 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": round(net_pnl_pct * 100, 4),
                    "equity_after": round(equity[i], 2),
                })

            # Open new position
            if desired != 0:
                entry_price = close[i] * (1 + slippage_bps / 10000 * abs(desired))
                entry_idx = i

        position[i] = desired

    # Calculate metrics
    returns = np.diff(equity) / equity[:-1]
    returns = returns[~np.isnan(returns)]

    total_return = (equity[-1] / initial_capital - 1) * 100
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(8760)  # annualized (hourly data)

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min() * 100

    # Trade stats
    winning = [t for t in trades if t["pnl_pct"] > 0]
    losing = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = len(winning) / max(len(trades), 1) * 100
    avg_win = np.mean([t["pnl_pct"] for t in winning]) if winning else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losing]) if losing else 0
    profit_factor = abs(sum(t["pnl_pct"] for t in winning)) / max(abs(sum(t["pnl_pct"] for t in losing)), 0.001)

    # Buy and hold comparison
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
    }

    logger.info(
        f"Backtest: {metrics['total_return_pct']:+.1f}% return, "
        f"{metrics['total_trades']} trades, "
        f"{metrics['win_rate_pct']:.0f}% win rate, "
        f"Sharpe {metrics['sharpe_ratio']:.2f}"
    )

    return {
        "metrics": metrics,
        "equity_curve": equity.tolist(),
        "trades": trades,
    }
