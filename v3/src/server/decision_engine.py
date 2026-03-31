"""Server-owned decision engine for v3."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..strategy.types import SignalEnvelope, SignalType, TradeAction, TradeDecision
from ..utils.logger import get_logger

logger = get_logger("v3.server.decision_engine")


@dataclass
class PairSnapshot:
    """Latest known signals for one pair."""

    latest: dict[SignalType, SignalEnvelope] = field(default_factory=dict)


class DecisionEngine:
    """Fuses signal envelopes into structured trade decisions."""

    def __init__(self, config: dict) -> None:
        self.config = config
        strategy_cfg = config.get("strategy", {})
        execution_cfg = config.get("execution", {})

        self.long_score_threshold = float(strategy_cfg.get("long_score_threshold", 0.60))
        self.short_score_threshold = float(strategy_cfg.get("short_score_threshold", 0.35))
        self.short_requires_bear_regime = bool(strategy_cfg.get("short_requires_bear_regime", True))
        self.default_order_type = str(strategy_cfg.get("default_order_type", "limit"))
        self.base_size_fraction = float(execution_cfg.get("base_size_fraction", 0.10))
        self.default_leverage = int(execution_cfg.get("default_leverage", 1))

        self.state: dict[str, PairSnapshot] = {}

    def ingest(self, envelope: SignalEnvelope) -> list[TradeDecision]:
        """Update state from one signal envelope and emit zero or one decision."""
        snapshot = self.state.setdefault(envelope.pair, PairSnapshot())
        snapshot.latest[envelope.signal_type] = envelope

        decision = self._decide(envelope.pair, snapshot)
        return [decision] if decision is not None else []

    def _decide(self, pair: str, snapshot: PairSnapshot) -> TradeDecision | None:
        market_state_signal = snapshot.latest.get(SignalType.MARKET_STATE)
        regime_signal = snapshot.latest.get(SignalType.REGIME)
        edge_signal = snapshot.latest.get(SignalType.EDGE)
        risk_signal = snapshot.latest.get(SignalType.RISK)
        event_signal = snapshot.latest.get(SignalType.EVENT)

        if regime_signal is None or edge_signal is None:
            return None

        if risk_signal and risk_signal.payload.get("trading_halted"):
            return TradeDecision(
                pair=pair,
                action=TradeAction.HOLD,
                confidence=1.0,
                reason="risk halt active",
                metadata={"source": "risk"},
            )

        regime = str(regime_signal.payload.get("regime", "unknown")).lower()
        edge_score = float(edge_signal.payload.get("score", 0.5))
        event_multiplier = 1.0
        market_state = market_state_signal.payload if market_state_signal is not None else {}
        market_ready = bool(market_state.get("ready", market_state_signal is None))
        position_side = int(market_state.get("position_side", 0) or 0)
        long_breakout = bool(market_state.get("long_breakout", False))
        short_breakout = bool(market_state.get("short_breakout", False))
        exit_long = bool(market_state.get("exit_long", False))
        exit_short = bool(market_state.get("exit_short", False))

        if event_signal is not None:
            event_multiplier = float(event_signal.payload.get("size_multiplier", 1.0))

        if not market_ready:
            return TradeDecision(
                pair=pair,
                action=TradeAction.HOLD,
                confidence=max(edge_score, 1.0 - edge_score),
                reason="market state not warmed up yet",
                metadata={"regime": regime, "edge_score": edge_score},
            )

        if position_side > 0:
            if exit_long:
                return TradeDecision(
                    pair=pair,
                    action=TradeAction.CLOSE,
                    confidence=max(edge_score, 1.0 - edge_score),
                    reason="long exit gate triggered",
                    order_type=self.default_order_type,
                    size_fraction=self.base_size_fraction * event_multiplier,
                    leverage=self.default_leverage,
                    metadata={"regime": regime, "edge_score": edge_score, "close_side": "long"},
                )
            return TradeDecision(
                pair=pair,
                action=TradeAction.HOLD,
                confidence=max(edge_score, 1.0 - edge_score),
                reason="long position active; no exit signal",
                metadata={"regime": regime, "edge_score": edge_score},
            )

        if position_side < 0:
            if exit_short:
                return TradeDecision(
                    pair=pair,
                    action=TradeAction.CLOSE,
                    confidence=max(edge_score, 1.0 - edge_score),
                    reason="short exit gate triggered",
                    order_type=self.default_order_type,
                    size_fraction=self.base_size_fraction * event_multiplier,
                    leverage=self.default_leverage,
                    metadata={"regime": regime, "edge_score": edge_score, "close_side": "short"},
                )
            return TradeDecision(
                pair=pair,
                action=TradeAction.HOLD,
                confidence=max(edge_score, 1.0 - edge_score),
                reason="short position active; no exit signal",
                metadata={"regime": regime, "edge_score": edge_score},
            )

        if regime == "bull" and long_breakout and edge_score >= self.long_score_threshold:
            return TradeDecision(
                pair=pair,
                action=TradeAction.BUY,
                confidence=edge_score,
                reason="bull breakout with strong edge score",
                order_type=self.default_order_type,
                size_fraction=self.base_size_fraction * event_multiplier,
                leverage=self.default_leverage,
                metadata={"regime": regime, "edge_score": edge_score},
            )

        if regime == "bear" and short_breakout and edge_score <= self.short_score_threshold:
            if self.short_requires_bear_regime:
                return TradeDecision(
                    pair=pair,
                    action=TradeAction.SELL,
                    confidence=1.0 - edge_score,
                    reason="bear breakout with short gate satisfied",
                    order_type=self.default_order_type,
                    size_fraction=self.base_size_fraction * event_multiplier,
                    leverage=self.default_leverage,
                    metadata={"regime": regime, "edge_score": edge_score},
                )

        return TradeDecision(
            pair=pair,
            action=TradeAction.HOLD,
            confidence=max(edge_score, 1.0 - edge_score),
            reason="no aligned breakout execution signal",
            metadata={"regime": regime, "edge_score": edge_score},
        )
