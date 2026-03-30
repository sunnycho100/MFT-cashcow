"""Minimal bootstrap for the v3 decision server scaffold."""

from __future__ import annotations

from v3.src.execution.kraken_gateway import KrakenExecutionGateway
from v3.src.server.decision_engine import DecisionEngine
from v3.src.strategy.types import SignalEnvelope, SignalType
from v3.src.transport.signal_bus import InMemorySignalBus
from v3.src.utils.config import load_config
from v3.src.utils.logger import get_logger

logger = get_logger("v3.main")


def bootstrap() -> None:
    """Wire the starter v3 components together."""
    config = load_config()
    bus = InMemorySignalBus(max_size=config.get("transport", {}).get("max_queue_size", 10000))
    engine = DecisionEngine(config)
    gateway = KrakenExecutionGateway(config)

    logger.info("V3 bootstrap complete")

    demo_signal = SignalEnvelope(
        source="bootstrap",
        pair="BTC/USD",
        timeframe="1h",
        signal_type=SignalType.REGIME,
        payload={"regime": "bull"},
        confidence=0.75,
    )
    bus.publish(demo_signal)

    for envelope in bus.drain():
        decisions = engine.ingest(envelope)
        for decision in decisions:
            logger.info(
                "Decision: pair={} action={} confidence={:.2f} reason={}",
                decision.pair,
                decision.action.value,
                decision.confidence,
                decision.reason,
            )
            logger.info("Preview order payload: {}", gateway.preview_payload(decision, reference_price=100000.0))


if __name__ == "__main__":
    bootstrap()
