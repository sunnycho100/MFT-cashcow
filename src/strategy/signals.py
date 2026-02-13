"""Signal generation from model outputs.

Aggregates signals from the ensemble model, applies filters,
and produces actionable trading decisions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import polars as pl

from ..models.base import Signal, SignalDirection
from ..utils.logger import get_logger

logger = get_logger("crypto_trader.strategy.signals")


@dataclass
class TradingDecision:
    """Actionable trading decision derived from model signals.

    Attributes:
        pair: Trading pair.
        action: 'buy', 'sell', 'close_long', 'close_short', or 'hold'.
        direction: Signal direction.
        confidence: Combined confidence level.
        suggested_size: Suggested position size (fraction of portfolio).
        stop_loss: Stop loss price.
        take_profit: Take profit price.
        reason: Human-readable reason.
        signals: Contributing signals.
        timestamp: Decision timestamp.
    """
    pair: str
    action: str
    direction: SignalDirection
    confidence: float
    suggested_size: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""
    signals: list[Signal] = field(default_factory=list)
    timestamp: Optional[datetime] = None


class SignalGenerator:
    """Generate actionable trading decisions from model signals.

    Applies position context (existing positions), signal filtering,
    and converts raw signals into trading decisions.

    Args:
        config: Configuration dictionary.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        risk_cfg = self.config.get("risk", {})
        self.min_confidence = risk_cfg.get("min_confidence", 0.2)
        self.max_position_pct = risk_cfg.get("max_position_pct", 0.10)

        # Track current positions
        self._positions: dict[str, dict] = {}

    def process_signals(
        self,
        signals: list[Signal],
        current_prices: dict[str, float],
        portfolio_value: float,
    ) -> list[TradingDecision]:
        """Process raw model signals into trading decisions.

        Args:
            signals: List of signals from models.
            current_prices: Dict mapping pair to current price.
            portfolio_value: Current total portfolio value.

        Returns:
            List of TradingDecision objects.
        """
        decisions = []

        # Group signals by pair
        pair_signals: dict[str, list[Signal]] = {}
        for sig in signals:
            pair_signals.setdefault(sig.pair, []).append(sig)

        for pair, sigs in pair_signals.items():
            decision = self._process_pair_signals(
                pair, sigs, current_prices.get(pair, 0.0), portfolio_value
            )
            if decision is not None:
                decisions.append(decision)

        # Check for positions that should be closed (no signal = potential exit)
        for pair, pos in self._positions.items():
            if pair not in pair_signals:
                # Check if stop loss or take profit hit
                current_price = current_prices.get(pair, 0.0)
                if current_price > 0:
                    close_decision = self._check_exit_conditions(pair, pos, current_price)
                    if close_decision:
                        decisions.append(close_decision)

        return decisions

    def _process_pair_signals(
        self,
        pair: str,
        signals: list[Signal],
        current_price: float,
        portfolio_value: float,
    ) -> Optional[TradingDecision]:
        """Process signals for a single pair.

        Args:
            pair: Trading pair.
            signals: Signals for this pair.
            current_price: Current market price.
            portfolio_value: Portfolio value.

        Returns:
            TradingDecision or None.
        """
        if not signals:
            return None

        # Find the highest-confidence signal (typically from ensemble)
        best_signal = max(signals, key=lambda s: s.confidence)

        if best_signal.confidence < self.min_confidence:
            logger.debug(f"{pair}: signal confidence {best_signal.confidence:.3f} "
                          f"below threshold {self.min_confidence}")
            return None

        # Determine action based on current position
        current_pos = self._positions.get(pair)
        action = self._determine_action(best_signal, current_pos)

        if action == "hold":
            return None

        # Calculate position size
        suggested_size = best_signal.confidence * self.max_position_pct

        return TradingDecision(
            pair=pair,
            action=action,
            direction=best_signal.direction,
            confidence=best_signal.confidence,
            suggested_size=suggested_size,
            stop_loss=best_signal.stop_loss,
            take_profit=best_signal.take_profit,
            reason=self._build_reason(best_signal, action),
            signals=signals,
            timestamp=best_signal.timestamp,
        )

    def _determine_action(self, signal: Signal, current_position: Optional[dict]) -> str:
        """Determine the appropriate action given signal and position.

        Args:
            signal: Current signal.
            current_position: Existing position dict or None.

        Returns:
            Action string: 'buy', 'sell', 'close_long', 'close_short', 'hold'.
        """
        has_position = current_position is not None
        direction = signal.direction

        if not has_position:
            if direction == SignalDirection.LONG:
                return "buy"
            elif direction == SignalDirection.SHORT:
                return "sell"
            return "hold"

        pos_side = current_position.get("side", "")

        if direction == SignalDirection.FLAT:
            return f"close_{pos_side}" if pos_side else "hold"

        if direction == SignalDirection.LONG and pos_side == "short":
            return "close_short"  # Close short, may open long next cycle
        elif direction == SignalDirection.SHORT and pos_side == "long":
            return "close_long"  # Close long, may open short next cycle
        elif direction == SignalDirection.LONG and pos_side == "long":
            return "hold"  # Already in desired position
        elif direction == SignalDirection.SHORT and pos_side == "short":
            return "hold"

        return "hold"

    def _check_exit_conditions(
        self, pair: str, position: dict, current_price: float
    ) -> Optional[TradingDecision]:
        """Check if exit conditions are met for a position.

        Args:
            pair: Trading pair.
            position: Position dict.
            current_price: Current price.

        Returns:
            TradingDecision to close or None.
        """
        stop_loss = position.get("stop_loss")
        take_profit = position.get("take_profit")
        side = position.get("side", "long")

        if side == "long":
            if stop_loss and current_price <= stop_loss:
                return TradingDecision(
                    pair=pair, action="close_long",
                    direction=SignalDirection.FLAT,
                    confidence=1.0,
                    reason=f"Stop loss hit at {current_price:.2f}",
                )
            if take_profit and current_price >= take_profit:
                return TradingDecision(
                    pair=pair, action="close_long",
                    direction=SignalDirection.FLAT,
                    confidence=1.0,
                    reason=f"Take profit hit at {current_price:.2f}",
                )
        elif side == "short":
            if stop_loss and current_price >= stop_loss:
                return TradingDecision(
                    pair=pair, action="close_short",
                    direction=SignalDirection.FLAT,
                    confidence=1.0,
                    reason=f"Stop loss hit at {current_price:.2f}",
                )
            if take_profit and current_price <= take_profit:
                return TradingDecision(
                    pair=pair, action="close_short",
                    direction=SignalDirection.FLAT,
                    confidence=1.0,
                    reason=f"Take profit hit at {current_price:.2f}",
                )

        return None

    def update_position(self, pair: str, position: Optional[dict]) -> None:
        """Update tracked position state.

        Args:
            pair: Trading pair.
            position: Position dict or None to remove.
        """
        if position is None:
            self._positions.pop(pair, None)
        else:
            self._positions[pair] = position

    @staticmethod
    def _build_reason(signal: Signal, action: str) -> str:
        """Build human-readable reason string."""
        parts = [f"{action.upper()} {signal.pair}"]
        parts.append(f"conf={signal.confidence:.2f}")
        parts.append(f"model={signal.model_name}")
        if signal.metadata:
            for k, v in list(signal.metadata.items())[:3]:
                if isinstance(v, float):
                    parts.append(f"{k}={v:.3f}")
        return " | ".join(parts)
