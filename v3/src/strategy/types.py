"""Core message and decision types for v3."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class SignalType(str, Enum):
    MARKET_STATE = "market_state"
    REGIME = "regime"
    EDGE = "edge"
    RISK = "risk"
    EVENT = "event"


class TradeAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


@dataclass(slots=True)
class SignalEnvelope:
    """Normalized signal message sent from a collector to the decision server."""

    source: str
    pair: str
    timeframe: str
    signal_type: SignalType
    payload: dict[str, Any]
    confidence: float = 0.0
    observed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signal_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(slots=True)
class TradeDecision:
    """Decision emitted by the server after fusing incoming signals."""

    pair: str
    action: TradeAction
    confidence: float
    reason: str
    order_type: str = "limit"
    size_fraction: float = 0.0
    leverage: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
