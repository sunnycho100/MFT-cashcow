"""EMA golden-cross / dead-cross backtest with monthly performance."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class BacktestResult:
    pair: str
    equity_curve: pl.DataFrame
    monthly: pl.DataFrame
    trades: pl.DataFrame
    summary: dict


def _ema(close: pl.Expr, span: int) -> pl.Expr:
    return close.ewm_mean(span=span, adjust=False, min_periods=span)


def compute_ema_columns(
    df: pl.DataFrame,
    fast: int,
    slow: int,
    trend: int,
) -> pl.DataFrame:
    c = pl.col("close")
    return df.with_columns(
        _ema(c, fast).alias("ema_fast"),
        _ema(c, slow).alias("ema_slow"),
        _ema(c, trend).alias("ema_trend"),
    ).drop_nulls()


def run_ema_cross_backtest(
    df: pl.DataFrame,
    pair: str,
    fast: int,
    slow: int,
    trend: int,
    initial_capital: float,
    fee_rate: float,
    slippage_bps: float,
) -> BacktestResult:
    """
    Long-only spot:
    - Enter on golden cross (EMA_fast crosses above EMA_slow) when close > EMA_trend.
    - Exit on dead cross (EMA_fast crosses below EMA_slow).
    """
    d = compute_ema_columns(df, fast, slow, trend)
    if d.is_empty():
        empty_m = pl.DataFrame(
            schema={"ym": pl.Utf8, "month_return_pct": pl.Float64, "bars": pl.UInt32}
        )
        return BacktestResult(
            pair=pair,
            equity_curve=d,
            monthly=empty_m,
            trades=pl.DataFrame(),
            summary={"error": "no rows after EMA warmup"},
        )

    close = d["close"].to_numpy()
    ema_f = d["ema_fast"].to_numpy()
    ema_s = d["ema_slow"].to_numpy()
    ema_t = d["ema_trend"].to_numpy()
    ts_ms = d["timestamp_ms"].to_numpy()

    slip = slippage_bps / 10_000.0
    n = len(close)

    cash = initial_capital
    units = 0.0
    in_pos = False

    eq_list: list[float] = []
    trade_rows: list[dict] = []
    entry_price = 0.0
    entry_i = 0

    for i in range(1, n):
        prev_f, prev_s = ema_f[i - 1], ema_s[i - 1]
        cur_f, cur_s = ema_f[i], ema_s[i]
        c = float(close[i])
        buy_px = c * (1.0 + slip)
        sell_px = c * (1.0 - slip)
        # golden cross: fast crosses above slow
        golden = prev_f <= prev_s and cur_f > cur_s
        # dead cross: fast crosses below slow
        dead = prev_f >= prev_s and cur_f < cur_s

        if not in_pos and golden and c > ema_t[i]:
            spend = cash * (1.0 - fee_rate)
            units = spend / buy_px
            cash = 0.0
            in_pos = True
            entry_price = buy_px
            entry_i = i

        elif in_pos and dead:
            proceeds = units * sell_px * (1.0 - fee_rate)
            cost = units * entry_price
            pnl = proceeds - cost
            cash = proceeds
            trade_rows.append(
                {
                    "entry_ts_ms": int(ts_ms[entry_i]),
                    "exit_ts_ms": int(ts_ms[i]),
                    "entry_price": float(entry_price),
                    "exit_price": float(sell_px),
                    "return_pct": float((proceeds / cost - 1.0) * 100.0) if cost else 0.0,
                    "pnl_abs": float(pnl),
                }
            )
            units = 0.0
            in_pos = False

        mark = cash + units * c
        eq_list.append(mark)

    d_skip = d.slice(1, d.height - 1) if d.height > 1 else d.head(0)
    equity_curve = pl.DataFrame(
        {
            "timestamp": d_skip["timestamp"],
            "timestamp_ms": d_skip["timestamp_ms"],
            "equity": eq_list,
            "close": close[1:],
        }
    )

    equity_curve = equity_curve.with_columns(
        pl.col("timestamp").dt.strftime("%Y-%m").alias("ym"),
    )

    monthly = (
        equity_curve.group_by("ym", maintain_order=True)
        .agg(
            pl.col("equity").first().alias("eq_month_open"),
            pl.col("equity").last().alias("eq_month_close"),
            pl.len().alias("bars"),
        )
        .with_columns(
            ((pl.col("eq_month_close") / pl.col("eq_month_open")) - 1.0).alias("month_return_pct")
        )
        .select(["ym", "month_return_pct", "bars", "eq_month_open", "eq_month_close"])
    )

    trades = pl.DataFrame(trade_rows) if trade_rows else pl.DataFrame()

    final_eq = float(eq_list[-1]) if eq_list else initial_capital
    total_ret = (final_eq / initial_capital - 1.0) * 100.0
    summary = {
        "pair": pair,
        "initial_capital": initial_capital,
        "final_equity": final_eq,
        "total_return_pct": total_ret,
        "n_trades": len(trade_rows),
        "fast": fast,
        "slow": slow,
        "trend": trend,
    }

    return BacktestResult(
        pair=pair,
        equity_curve=equity_curve,
        monthly=monthly,
        trades=trades,
        summary=summary,
    )


def _equity_monthly_from_eq_list(
    d: pl.DataFrame,
    close,
    eq_list: list[float],
    initial_capital: float,
    pair: str,
    extra_summary: dict,
) -> BacktestResult:
    d_skip = d.slice(1, d.height - 1) if d.height > 1 else d.head(0)
    equity_curve = pl.DataFrame(
        {
            "timestamp": d_skip["timestamp"],
            "timestamp_ms": d_skip["timestamp_ms"],
            "equity": eq_list,
            "close": close[1:],
        }
    )
    equity_curve = equity_curve.with_columns(
        pl.col("timestamp").dt.strftime("%Y-%m").alias("ym"),
    )
    monthly = (
        equity_curve.group_by("ym", maintain_order=True)
        .agg(
            pl.col("equity").first().alias("eq_month_open"),
            pl.col("equity").last().alias("eq_month_close"),
            pl.len().alias("bars"),
        )
        .with_columns(
            ((pl.col("eq_month_close") / pl.col("eq_month_open")) - 1.0).alias("month_return_pct")
        )
        .select(["ym", "month_return_pct", "bars", "eq_month_open", "eq_month_close"])
    )
    final_eq = float(eq_list[-1]) if eq_list else initial_capital
    total_ret = (final_eq / initial_capital - 1.0) * 100.0
    trades_df = extra_summary.pop("_trades_df", pl.DataFrame())
    summary = {
        "pair": pair,
        "initial_capital": initial_capital,
        "final_equity": final_eq,
        "total_return_pct": total_ret,
        **extra_summary,
    }
    return BacktestResult(
        pair=pair,
        equity_curve=equity_curve,
        monthly=monthly,
        trades=trades_df,
        summary=summary,
    )


def run_price_vs_ema_backtest(
    df: pl.DataFrame,
    pair: str,
    signal: int,
    trend: int,
    initial_capital: float,
    fee_rate: float,
    slippage_bps: float,
) -> BacktestResult:
    """
    Long-only using **price vs EMA(signal)** crossovers (not EMA-on-EMA):
    - Golden: close crosses **above** EMA(signal) (was at/below, now above).
    - Dead: close crosses **below** EMA(signal).
    - Entry only when close > EMA(trend) at the entry bar (trend filter).
    """
    c = pl.col("close")
    d = (
        df.with_columns(
            _ema(c, signal).alias("ema_signal"),
            _ema(c, trend).alias("ema_trend"),
        )
        .drop_nulls()
    )
    if d.is_empty():
        return BacktestResult(
            pair=pair,
            equity_curve=d,
            monthly=pl.DataFrame(),
            trades=pl.DataFrame(),
            summary={"error": "no rows after EMA warmup"},
        )

    close = d["close"].to_numpy()
    ema_sig = d["ema_signal"].to_numpy()
    ema_t = d["ema_trend"].to_numpy()
    ts_ms = d["timestamp_ms"].to_numpy()
    slip = slippage_bps / 10_000.0
    n = len(close)

    cash = initial_capital
    units = 0.0
    in_pos = False
    eq_list: list[float] = []
    trade_rows: list[dict] = []
    entry_price = 0.0
    entry_i = 0

    for i in range(1, n):
        pc, cc = float(close[i - 1]), float(close[i])
        ps, cs = float(ema_sig[i - 1]), float(ema_sig[i])
        _, ct = float(ema_t[i - 1]), float(ema_t[i])
        c = float(close[i])
        buy_px = c * (1.0 + slip)
        sell_px = c * (1.0 - slip)
        # price golden: close crosses above EMA(signal)
        golden_price = pc <= ps and cc > cs
        # price dead: close crosses below EMA(signal)
        dead_price = pc >= ps and cc < cs

        if not in_pos and golden_price and cc > ct:
            spend = cash * (1.0 - fee_rate)
            units = spend / buy_px
            cash = 0.0
            in_pos = True
            entry_price = buy_px
            entry_i = i
        elif in_pos and dead_price:
            proceeds = units * sell_px * (1.0 - fee_rate)
            cost = units * entry_price
            pnl = proceeds - cost
            cash = proceeds
            trade_rows.append(
                {
                    "entry_ts_ms": int(ts_ms[entry_i]),
                    "exit_ts_ms": int(ts_ms[i]),
                    "entry_price": float(entry_price),
                    "exit_price": float(sell_px),
                    "return_pct": float((proceeds / cost - 1.0) * 100.0) if cost else 0.0,
                    "pnl_abs": float(pnl),
                }
            )
            units = 0.0
            in_pos = False

        mark = cash + units * c
        eq_list.append(mark)

    trades_df = pl.DataFrame(trade_rows) if trade_rows else pl.DataFrame()
    extra: dict = {
        "n_trades": len(trade_rows),
        "mode": "price_vs_ema",
        "signal_ema": signal,
        "trend_ema": trend,
        "_trades_df": trades_df,
    }
    return _equity_monthly_from_eq_list(
        d, close, eq_list, initial_capital, pair, extra
    )


def run_ema50_vs_ema120_backtest(
    df: pl.DataFrame,
    pair: str,
    fast: int,
    mid: int,
    slow: int,
    initial_capital: float,
    fee_rate: float,
    slippage_bps: float,
) -> BacktestResult:
    """
    **Triple EMA cross** (30/50/120 style): golden/dead on **EMA(mid) vs EMA(slow)**,
    with **EMA(fast)** trend filter — enter only when close > EMA(fast) at signal.
    """
    c = pl.col("close")
    d = (
        df.with_columns(
            _ema(c, fast).alias("ema_fast"),
            _ema(c, mid).alias("ema_mid"),
            _ema(c, slow).alias("ema_slow"),
        )
        .drop_nulls()
    )
    if d.is_empty():
        return BacktestResult(
            pair=pair,
            equity_curve=d,
            monthly=pl.DataFrame(),
            trades=pl.DataFrame(),
            summary={"error": "no rows after EMA warmup"},
        )

    close = d["close"].to_numpy()
    ema_m = d["ema_mid"].to_numpy()
    ema_sl = d["ema_slow"].to_numpy()
    ema_f = d["ema_fast"].to_numpy()
    ts_ms = d["timestamp_ms"].to_numpy()
    slip = slippage_bps / 10_000.0
    n = len(close)

    cash = initial_capital
    units = 0.0
    in_pos = False
    eq_list: list[float] = []
    trade_rows: list[dict] = []
    entry_price = 0.0
    entry_i = 0

    for i in range(1, n):
        prev_m, prev_sl = ema_m[i - 1], ema_sl[i - 1]
        cur_m, cur_sl = ema_m[i], ema_sl[i]
        c = float(close[i])
        buy_px = c * (1.0 + slip)
        sell_px = c * (1.0 - slip)
        golden = prev_m <= prev_sl and cur_m > cur_sl
        dead = prev_m >= prev_sl and cur_m < cur_sl

        if not in_pos and golden and c > ema_f[i]:
            spend = cash * (1.0 - fee_rate)
            units = spend / buy_px
            cash = 0.0
            in_pos = True
            entry_price = buy_px
            entry_i = i
        elif in_pos and dead:
            proceeds = units * sell_px * (1.0 - fee_rate)
            cost = units * entry_price
            pnl = proceeds - cost
            cash = proceeds
            trade_rows.append(
                {
                    "entry_ts_ms": int(ts_ms[entry_i]),
                    "exit_ts_ms": int(ts_ms[i]),
                    "entry_price": float(entry_price),
                    "exit_price": float(sell_px),
                    "return_pct": float((proceeds / cost - 1.0) * 100.0) if cost else 0.0,
                    "pnl_abs": float(pnl),
                }
            )
            units = 0.0
            in_pos = False

        mark = cash + units * c
        eq_list.append(mark)

    trades_df = pl.DataFrame(trade_rows) if trade_rows else pl.DataFrame()
    extra: dict = {
        "n_trades": len(trade_rows),
        "mode": "ema_mid_vs_slow",
        "ema_fast": fast,
        "ema_mid": mid,
        "ema_slow": slow,
        "_trades_df": trades_df,
    }
    return _equity_monthly_from_eq_list(
        d, close, eq_list, initial_capital, pair, extra
    )
