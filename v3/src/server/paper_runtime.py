"""Paper-trading runtime for the v3 server-owned decision path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from v3.src.execution.kraken_gateway import KrakenExecutionGateway
from v3.src.server.decision_engine import DecisionEngine
from v3.src.server.paper_metrics import append_cycle_log
from v3.src.strategy.hybrid_stack import (
    BEST_TREND_CONFIG,
    build_funding_overlay_frames,
    build_funding_premium_overlay_frames,
    load_latest_candle_timestamps,
    load_base_feature_frames_with_refresh,
    load_v2_runtime_config,
    refresh_latest_candles,
    write_summary_json,
)
from v3.src.strategy.calibration import predict_frame_with_calibration, train_model_with_tail_calibration
from v3.src.strategy.trend_runtime import (
    PositionState,
    apply_market_snapshot,
    apply_trade_action,
    build_market_state_payload,
    detect_regime_from_row,
    prepare_runtime_frame,
)
from v3.src.strategy.types import SignalEnvelope, SignalType
from v3.src.transport.signal_bus import InMemorySignalBus
from v3.src.utils.config import load_config


@dataclass(slots=True)
class PaperDecisionRecord:
    pair: str
    variant: str
    calibration_method: str
    signal_timestamp: str
    reference_price: float
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
        self.position_states: dict[str, PositionState] = self._restore_position_states()

    def _load_optimization_summary(self) -> dict | None:
        summary_path = Path(
            self.config.get("paper", {}).get(
                "optimization_summary_path",
                self.config.get("optimization", {}).get("summary_path", "v3/data/optimization/funding_premium_threshold_optimization.json"),
            )
        )
        if not summary_path.exists():
            return None
        with open(summary_path) as handle:
            return json.load(handle)

    def _resolve_variant(self) -> str:
        configured = str(self.config.get("paper", {}).get("model_variant", "auto"))
        if configured != "auto":
            return configured
        optimization = self._load_optimization_summary()
        if optimization is not None:
            return str(optimization.get("variant", "funding_premium"))
        walkforward_path = Path(self.config.get("paper", {}).get("walkforward_summary_path", "v3/data/walkforward/hybrid_overlay_walkforward_summary.json"))
        if walkforward_path.exists():
            with open(walkforward_path) as handle:
                payload = json.load(handle)
            return str(payload.get("champion_variant", "funding"))
        return "funding"

    def _build_variant_frames(self, days: int, variant: str, refresh_latest: bool = True) -> dict:
        v2_config = load_v2_runtime_config()
        base_frames = load_base_feature_frames_with_refresh(v2_config, days=days, refresh_latest=refresh_latest)
        if variant == "base":
            return base_frames
        if variant == "funding":
            frames, _ = build_funding_overlay_frames(base_frames)
            return frames
        if variant == "funding_premium":
            frames, _ = build_funding_premium_overlay_frames(base_frames)
            return frames
        raise ValueError(f"Unsupported paper variant: {variant}")

    def _apply_threshold_profile(self, optimization: dict | None) -> dict:
        calibration_method = str(self.config.get("optimization", {}).get("default_calibration_method", "raw"))
        if optimization is None:
            return {"calibration_method": calibration_method}

        best = optimization.get("best", {})
        self.engine.long_score_threshold = float(best.get("decision_long_threshold", self.engine.long_score_threshold))
        self.engine.short_score_threshold = float(best.get("decision_short_score_threshold", self.engine.short_score_threshold))
        return {
            "calibration_method": str(best.get("calibration_method", calibration_method)),
            "decision_long_threshold": self.engine.long_score_threshold,
            "decision_short_score_threshold": self.engine.short_score_threshold,
        }

    def _latest_timestamps(self, pair_frames: dict[str, object]) -> dict[str, str]:
        return {pair: str(frame["timestamp"][-1]) for pair, frame in pair_frames.items()}

    def _load_last_artifact(self) -> dict | None:
        artifact_path = Path(self.config.get("paper", {}).get("artifact_path", "v3/data/paper/latest_paper_cycle.json"))
        if not artifact_path.exists():
            return None
        with open(artifact_path) as handle:
            return json.load(handle)

    def _restore_position_states(self) -> dict[str, PositionState]:
        last_artifact = self._load_last_artifact()
        if not last_artifact:
            return {}
        payload = last_artifact.get("position_states", {})
        return {pair: PositionState.from_dict(state) for pair, state in payload.items()}

    def run_once(self, days: int | None = None) -> list[PaperDecisionRecord]:
        runtime_days = int(days or self.config.get("paper", {}).get("history_days", 1095))
        variant = self._resolve_variant()
        optimization = self._load_optimization_summary() if self.config.get("paper", {}).get("use_optimized_thresholds", True) else None
        threshold_profile = self._apply_threshold_profile(optimization)
        v2_config = load_v2_runtime_config()
        refresh_latest_candles(v2_config, timeframe="1h")
        latest_candle_timestamps = load_latest_candle_timestamps(v2_config, timeframe="1h")

        if self.config.get("paper", {}).get("skip_if_same_candle", True):
            last_artifact = self._load_last_artifact()
            if last_artifact and last_artifact.get("latest_timestamps") == latest_candle_timestamps:
                return []

        pair_frames = self._build_variant_frames(runtime_days, variant, refresh_latest=False)
        latest_timestamps = self._latest_timestamps(pair_frames)

        model, feature_names, calibration_bundle, metrics = train_model_with_tail_calibration(
            v2_config,
            pair_frames,
            calibration_method=threshold_profile["calibration_method"],
            calibration_fraction=float(self.config.get("optimization", {}).get("calibration_fraction", 0.2)),
        )

        records = []
        latest_context = {}
        for pair, frame in pair_frames.items():
            runtime_frame = prepare_runtime_frame(
                frame,
                entry_period=int(BEST_TREND_CONFIG["entry_period"]),
                exit_period=int(BEST_TREND_CONFIG["exit_period"]),
                atr_period=int(BEST_TREND_CONFIG["atr_period"]),
            )
            latest = predict_frame_with_calibration(model, runtime_frame.tail(1), calibration_bundle, feature_names=feature_names).to_dicts()[0]
            regime = detect_regime_from_row(latest)
            score = float(latest["pred_long_prob"])
            latest_ts = latest["timestamp"]
            stale = False
            if isinstance(latest_ts, datetime):
                stale = latest_ts < datetime.now(timezone.utc) - timedelta(hours=2)
            usd_pair = pair.replace("USDT", "USD")
            current_position = self.position_states.get(usd_pair, PositionState())
            market_state = build_market_state_payload(latest, current_position, atr_stop_mult=float(BEST_TREND_CONFIG["atr_stop_mult"]))
            latest_context[usd_pair] = {
                "timestamp": str(latest["timestamp"]),
                "price": float(latest["close"]),
                "market_state": market_state,
            }

            envelopes = [
                SignalEnvelope(
                    source=f"paper:{variant}",
                    pair=usd_pair,
                    timeframe="1h",
                    signal_type=SignalType.RISK,
                    payload={"trading_halted": stale},
                    confidence=1.0,
                ),
                SignalEnvelope(
                    source=f"paper:{variant}",
                    pair=usd_pair,
                    timeframe="1h",
                    signal_type=SignalType.MARKET_STATE,
                    payload=market_state,
                    confidence=1.0,
                ),
                SignalEnvelope(
                    source=f"paper:{variant}",
                    pair=usd_pair,
                    timeframe="1h",
                    signal_type=SignalType.REGIME,
                    payload={"regime": regime, "variant": variant},
                    confidence=max(score, 1.0 - score),
                ),
                SignalEnvelope(
                    source=f"paper:{variant}",
                    pair=usd_pair,
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
                current_context = latest_context[decision.pair]
                market_state = current_context["market_state"]
                preview = self.gateway.preview_payload(decision, reference_price=reference_price)
                submit_result = self.gateway.submit(decision, reference_price=reference_price)
                before_position = self.position_states.get(decision.pair, PositionState())
                after_position = apply_trade_action(
                    before_position,
                    decision.action,
                    price=float(current_context["price"]),
                    timestamp=current_context["timestamp"],
                    market_state=market_state,
                    atr_stop_mult=float(BEST_TREND_CONFIG["atr_stop_mult"]),
                )
                self.position_states[decision.pair] = after_position
                records.append(
                    PaperDecisionRecord(
                        pair=decision.pair,
                        variant=variant,
                        calibration_method=calibration_bundle.method,
                        signal_timestamp=current_context["timestamp"],
                        reference_price=reference_price,
                        regime=str(decision.metadata.get("regime", "unknown")),
                        edge_score=float(decision.metadata.get("edge_score", 0.5)),
                        action=decision.action.value,
                        reason=decision.reason,
                        preview_payload={**preview, "position_before": before_position.to_dict(), "position_after": after_position.to_dict()},
                        submit_result=submit_result,
                    )
                )

        for pair, context in latest_context.items():
            if pair not in self.position_states:
                self.position_states[pair] = PositionState()
            if not any(record.pair == pair for record in records):
                self.position_states[pair] = apply_market_snapshot(self.position_states[pair], context["market_state"])

        artifact_path = Path(self.config.get("paper", {}).get("artifact_path", "v3/data/paper/latest_paper_cycle.json"))
        payload = {
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "variant": variant,
            "calibration_method": calibration_bundle.method,
            "thresholds": threshold_profile,
            "latest_timestamps": latest_timestamps,
            "train_metrics": metrics,
            "decisions": [asdict(record) for record in records],
            "position_states": {pair: state.to_dict() for pair, state in self.position_states.items()},
        }
        write_summary_json(payload, artifact_path)
        append_cycle_log(
            self.config.get("paper", {}).get("cycle_log_path", "v3/data/paper/paper_cycles.jsonl"),
            payload,
        )
        return records
