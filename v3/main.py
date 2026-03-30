"""Minimal bootstrap for the v3 decision server scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.src.execution.kraken_gateway import KrakenExecutionGateway
from v3.src.server.decision_engine import DecisionEngine
from v3.src.server.paper_runtime import IntegratedPaperRuntime
from v3.src.strategy.types import SignalEnvelope, SignalType
from v3.src.transport.signal_bus import InMemorySignalBus
from v3.src.utils.config import load_config
from v3.src.utils.logger import get_logger

logger = get_logger("v3.main")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["bootstrap", "paper-once"], default="bootstrap")
    parser.add_argument("--days", type=int, default=None)
    return parser


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
    args = build_parser().parse_args()
    if args.mode == "paper-once":
        runtime = IntegratedPaperRuntime(load_config())
        records = runtime.run_once(days=args.days)
        for record in records:
            logger.info(
                "Paper cycle: pair={} action={} regime={} edge={:.3f} reason={}",
                record.pair,
                record.action,
                record.regime,
                record.edge_score,
                record.reason,
            )
    else:
        bootstrap()
