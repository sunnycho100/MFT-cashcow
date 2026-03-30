"""Paper-trading runtime for the v3 server-owned decision path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from v3.src.execution.kraken_gateway import KrakenExecutionGateway
from v3.src.server.decision_engine import DecisionEngine
from v3.src.strategy.hybrid_stack import (
    build_funding_overlay_frames,
    build_funding_premium_overlay_frames,
    load_base_feature_frames_with_refresh,
    load_v2_runtime_config,
    write_summary_json,
)
from v3.src.strategy.types import SignalEnvelope, SignalType
from v3.src.transport.signal_bus import InMemorySignalBus
from v3.src.utils.config import load_config

from v2.src.features.pipeline import get_feature_names
from v2.src.models.lgbm_model import LGBMModel


@dataclass(slots=True)
class PaperDecisionRecord:
    pair: str
    variant: str
    regime: str
    edge_score: float
    action: str
    reason: str
    preview_payload: dict
    submit_result: dict


class IntegratedPaperRuntime:
    """Train the current champion variant and run one server-owned paper cycle."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or load_config()
        self.bus = InMemorySignalBus(max_size=self.config.get("transport", {}).get("max_queue_size", 10000))
        self.engine = DecisionEngine(self.config)
        self.gateway = KrakenExecutionGateway(self.config)

    def _resolve_variant(self) -> str:
        configured = str(self.config.get("paper", {}).get("model_variant", "auto"))
        if configured != "auto":
            return configured
        walkforward_path = Path(self.config.get("paper", {}).get("walkforward_summary_path", "v3/data/walkforward/hybrid_overlay_walkforward_summary.json"))
        if walkforward_path.exists():
            import json

            with open(walkforward_path) as handle:
                payload = json.load(handle)
            return str(payload.get("champion_variant", "funding"))
        return "funding"

    def _detect_regime(self, row: dict) -> str:
        adx_56 = float(row.get("adx_56", 0.0))
        dist_ema_168 = float(row.get("dist_ema_168", 0.0))
        dist_ema_504 = float(row.get("dist_ema_504", 0.0))
        if adx_56 >= 18.0 and dist_ema_168 > 0 and dist_ema_504 > 0:
            return "bull"
        if adx_56 >= 18.0 and dist_ema_168 < 0 and dist_ema_504 < 0:
            return "bear"
        return "range"

    def _build_variant_frames(self, days: int, variant: str) -> dict:
        v2_config = load_v2_runtime_config()
        base_frames = load_base_feature_frames_with_refresh(v2_config, days=days, refresh_latest=True)
        if variant == "base":
            return base_frames
        if variant == "funding":
            frames, _ = build_funding_overlay_frames(base_frames)
            return frames
        if variant == "funding_premium":
            frames, _ = build_funding_premium_overlay_frames(base_frames)
            return frames
        raise ValueError(f"Unsupported paper variant: {variant}")

    def run_once(self, days: int | None = None) -> list[PaperDecisionRecord]:
        runtime_days = int(days or self.config.get("paper", {}).get("history_days", 1095))
        variant = self._resolve_variant()
        v2_config = load_v2_runtime_config()
        pair_frames = self._build_variant_frames(runtime_days, variant)

        model = LGBMModel(v2_config)
        feature_names = get_feature_names(next(iter(pair_frames.values())))
        train_parts = [model.create_labels(frame) for frame in pair_frames.values()]
        combined = pl.concat(train_parts)
        split_idx = max(int(len(combined) * 0.9), 1)
        metrics = model.train(combined[:split_idx], combined[split_idx:], feature_names)

        records = []
        for pair, frame in pair_frames.items():
            latest = model.predict(frame.tail(1)).to_dicts()[0]
            regime = self._detect_regime(latest)
            score = float(latest["pred_long_prob"])
            latest_ts = latest["timestamp"]
            stale = False
            if isinstance(latest_ts, datetime):
                stale = latest_ts < datetime.now(timezone.utc) - timedelta(hours=2)

            envelopes = [
                SignalEnvelope(
                    source=f"paper:{variant}",
                    pair=pair.replace("USDT", "USD"),
                    timeframe="1h",
                    signal_type=SignalType.RISK,
                    payload={"trading_halted": stale},
                    confidence=1.0,
                ),
                SignalEnvelope(
                    source=f"paper:{variant}",
                    pair=pair.replace("USDT", "USD"),
                    timeframe="1h",
                    signal_type=SignalType.REGIME,
                    payload={"regime": regime, "variant": variant},
                    confidence=max(score, 1.0 - score),
                ),
                SignalEnvelope(
                    source=f"paper:{variant}",
                    pair=pair.replace("USDT", "USD"),
                    timeframe="1h",
                    signal_type=SignalType.EDGE,
                    payload={"score": score, "variant": variant, "timestamp": str(latest["timestamp"])},
                    confidence=max(score, 1.0 - score),
                ),
            ]

            for envelope in envelopes:
                self.bus.publish(envelope)

        for envelope in self.bus.drain():
            decisions = self.engine.ingest(envelope)
            if envelope.signal_type != SignalType.EDGE:
                continue
            for decision in decisions:
                reference_frame = pair_frames[decision.pair.replace("/USD", "/USDT")]
                reference_price = float(reference_frame["close"][-1])
                preview = self.gateway.preview_payload(decision, reference_price=reference_price)
                submit_result = self.gateway.submit(decision, reference_price=reference_price)
                records.append(
                    PaperDecisionRecord(
                        pair=decision.pair,
                        variant=variant,
                        regime=str(decision.metadata.get("regime", "unknown")),
                        edge_score=float(decision.metadata.get("edge_score", 0.5)),
                        action=decision.action.value,
                        reason=decision.reason,
                        preview_payload=preview,
                        submit_result=submit_result,
                    )
                )

        artifact_path = Path(self.config.get("paper", {}).get("artifact_path", "v3/data/paper/latest_paper_cycle.json"))
        write_summary_json(
            {
                "variant": variant,
                "train_metrics": metrics,
                "decisions": [asdict(record) for record in records],
            },
            artifact_path,
        )
        return records
