"""Kraken execution gateway for v3.

This starts as a thin adapter around the existing v2 Kraken client so we can
move to a server-owned execution boundary without rewriting everything at once.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..strategy.types import TradeAction, TradeDecision
from ..utils.logger import get_logger

logger = get_logger("v3.execution.kraken_gateway")

try:
    from v2.src.execution.kraken_client import KrakenClient
except Exception:  # pragma: no cover - optional until runtime deps are installed
    KrakenClient = None


@dataclass(slots=True)
class KrakenOrderInstruction:
    pair: str
    side: str
    order_type: str
    volume: float
    price: float | None
    leverage: int | None
    validate: bool


class KrakenExecutionGateway:
    """Translate normalized trade decisions into Kraken order instructions."""

    def __init__(self, config: dict) -> None:
        self.config = config
        execution_cfg = config.get("execution", {})
        kraken_cfg = config.get("kraken", {})

        self.validate_orders = bool(execution_cfg.get("validate_orders", True))
        self.default_leverage = int(execution_cfg.get("default_leverage", 1))
        self.use_margin = bool(kraken_cfg.get("use_margin", False))

        if KrakenClient is not None:
            self.client = KrakenClient()
        else:
            self.client = None

    def preview_payload(self, decision: TradeDecision, reference_price: float) -> dict[str, Any]:
        """Return the order payload we would send to Kraken."""
        instruction = self._translate(decision, reference_price=reference_price)
        return asdict(instruction)

    def submit(self, decision: TradeDecision, reference_price: float) -> dict[str, Any]:
        """Submit a trade decision through the underlying Kraken client."""
        instruction = self._translate(decision, reference_price=reference_price)
        if decision.action == TradeAction.HOLD:
            return {"status": "skipped", "reason": decision.reason}

        if self.client is None:
            logger.warning("Kraken client unavailable; returning preview only")
            return {"status": "preview_only", "instruction": asdict(instruction)}

        return self.client.place_order(
            pair=instruction.pair,
            side=instruction.side,
            order_type=instruction.order_type,
            volume=instruction.volume,
            price=instruction.price,
            leverage=instruction.leverage,
            validate=instruction.validate,
        )

    def _translate(self, decision: TradeDecision, reference_price: float) -> KrakenOrderInstruction:
        """Convert a normalized decision into a Kraken order instruction."""
        if decision.action == TradeAction.HOLD:
            return KrakenOrderInstruction(
                pair=decision.pair,
                side="buy",
                order_type=decision.order_type,
                volume=0.0,
                price=None,
                leverage=None,
                validate=True,
            )

        if decision.action == TradeAction.CLOSE:
            close_side = str(decision.metadata.get("close_side", "long")).lower()
            side = "sell" if close_side == "long" else "buy"
        else:
            side = "buy" if decision.action == TradeAction.BUY else "sell"
        volume = 0.0 if reference_price <= 0 else decision.size_fraction / reference_price
        leverage = self.default_leverage if self.use_margin else None

        return KrakenOrderInstruction(
            pair=decision.pair,
            side=side,
            order_type=decision.order_type,
            volume=volume,
            price=None if decision.order_type == "market" else reference_price,
            leverage=decision.leverage or leverage,
            validate=self.validate_orders,
        )
