"""Runtime helpers to keep paper/live decisions aligned with the hybrid trend backtest."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl
import talib

from .types import TradeAction


@dataclass(slots=True)
class PositionState:
    """Minimal position state for live-style entry/exit gating."""

    side: int = 0
    entry_price: float | None = None
    entry_timestamp: str | None = None
    trail_peak: float | None = None
    trail_stop: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "entry_price": self.entry_price,
            "entry_timestamp": self.entry_timestamp,
            "trail_peak": self.trail_peak,
            "trail_stop": self.trail_stop,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PositionState":
        if not payload:
            return cls()
        return cls(
            side=int(payload.get("side", 0) or 0),
            entry_price=float(payload["entry_price"]) if payload.get("entry_price") is not None else None,
            entry_timestamp=str(payload["entry_timestamp"]) if payload.get("entry_timestamp") else None,
            trail_peak=float(payload["trail_peak"]) if payload.get("trail_peak") is not None else None,
            trail_stop=float(payload["trail_stop"]) if payload.get("trail_stop") is not None else None,
        )


def detect_regime_from_row(row: dict[str, Any]) -> str:
    adx_56 = float(row.get("adx_56", 0.0))
    dist_ema_168 = float(row.get("dist_ema_168", 0.0))
    dist_ema_504 = float(row.get("dist_ema_504", 0.0))
    if adx_56 >= 18.0 and dist_ema_168 > 0 and dist_ema_504 > 0:
        return "bull"
    if adx_56 >= 18.0 and dist_ema_168 < 0 and dist_ema_504 < 0:
        return "bear"
    return "range"


def prepare_runtime_frame(
    frame: pl.DataFrame,
    entry_period: int,
    exit_period: int,
    atr_period: int,
) -> pl.DataFrame:
    """Add runtime-only indicators used by the live paper decision path."""
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    close = frame["close"].to_numpy()
    atr = talib.ATR(high, low, close, timeperiod=atr_period)

    return frame.with_columns(
        pl.col("high").shift(1).rolling_max(window_size=entry_period).alias("runtime_entry_upper"),
        pl.col("low").shift(1).rolling_min(window_size=entry_period).alias("runtime_entry_lower"),
        pl.col("high").shift(1).rolling_max(window_size=exit_period).alias("runtime_exit_upper"),
        pl.col("low").shift(1).rolling_min(window_size=exit_period).alias("runtime_exit_lower"),
        pl.Series("runtime_atr", atr),
    )


def build_market_state_payload(
    row: dict[str, Any],
    position: PositionState,
    atr_stop_mult: float,
) -> dict[str, Any]:
    """Build the latest live-style market snapshot for one pair."""
    close = float(row.get("close", 0.0))
    atr = row.get("runtime_atr")
    entry_upper = row.get("runtime_entry_upper")
    entry_lower = row.get("runtime_entry_lower")
    exit_upper = row.get("runtime_exit_upper")
    exit_lower = row.get("runtime_exit_lower")

    ready = all(value is not None for value in [atr, entry_upper, entry_lower, exit_upper, exit_lower])
    atr_value = float(atr) if ready else 0.0

    long_breakout = ready and close > float(entry_upper)
    short_breakout = ready and close < float(entry_lower)

    trail_peak = position.trail_peak
    trail_stop = position.trail_stop
    exit_long = False
    exit_short = False

    if ready and position.side > 0:
        trail_peak = max(float(position.trail_peak or close), close)
        trail_stop = trail_peak - atr_stop_mult * atr_value
        exit_long = (
            close <= trail_stop
            or close < float(exit_lower)
            or close < float(entry_lower)
        )
    elif ready and position.side < 0:
        trail_peak = min(float(position.trail_peak or close), close)
        trail_stop = trail_peak + atr_stop_mult * atr_value
        exit_short = (
            close >= trail_stop
            or close > float(exit_upper)
            or close > float(entry_upper)
        )

    return {
        "ready": ready,
        "position_side": position.side,
        "entry_price": position.entry_price,
        "entry_timestamp": position.entry_timestamp,
        "long_breakout": bool(long_breakout),
        "short_breakout": bool(short_breakout),
        "exit_long": bool(exit_long),
        "exit_short": bool(exit_short),
        "trail_peak": trail_peak,
        "trail_stop": trail_stop,
        "atr": atr_value,
        "entry_upper": float(entry_upper) if entry_upper is not None else None,
        "entry_lower": float(entry_lower) if entry_lower is not None else None,
        "exit_upper": float(exit_upper) if exit_upper is not None else None,
        "exit_lower": float(exit_lower) if exit_lower is not None else None,
    }


def apply_market_snapshot(position: PositionState, market_state: dict[str, Any]) -> PositionState:
    """Carry forward trailing-stop state even when no trade is executed."""
    if position.side == 0:
        return position
    return PositionState(
        side=position.side,
        entry_price=position.entry_price,
        entry_timestamp=position.entry_timestamp,
        trail_peak=float(market_state["trail_peak"]) if market_state.get("trail_peak") is not None else position.trail_peak,
        trail_stop=float(market_state["trail_stop"]) if market_state.get("trail_stop") is not None else position.trail_stop,
    )


def apply_trade_action(
    position: PositionState,
    action: TradeAction | str,
    price: float,
    timestamp: str | datetime,
    market_state: dict[str, Any],
    atr_stop_mult: float,
) -> PositionState:
    """Update position state after one normalized trade action."""
    normalized = action.value if isinstance(action, TradeAction) else str(action).lower()
    marked = apply_market_snapshot(position, market_state)
    ts_value = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)
    atr_value = float(market_state.get("atr", 0.0) or 0.0)

    if normalized == TradeAction.HOLD.value:
        return marked
    if normalized == TradeAction.CLOSE.value:
        return PositionState()

    if normalized == TradeAction.BUY.value:
        return PositionState(
            side=1,
            entry_price=price,
            entry_timestamp=ts_value,
            trail_peak=price,
            trail_stop=price - atr_stop_mult * atr_value if atr_value > 0 else None,
        )

    if normalized == TradeAction.SELL.value:
        return PositionState(
            side=-1,
            entry_price=price,
            entry_timestamp=ts_value,
            trail_peak=price,
            trail_stop=price + atr_stop_mult * atr_value if atr_value > 0 else None,
        )

    return marked
