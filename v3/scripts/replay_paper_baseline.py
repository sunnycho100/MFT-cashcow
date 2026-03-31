"""Replay the champion paper-decision path over recent history and score signal quality."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v3.src.server.decision_engine import DecisionEngine
from v3.src.server.paper_metrics import (
    append_cycle_log,
    evaluate_decisions,
    evaluate_round_trip_trades,
    flatten_decisions,
    load_cycle_log,
)
from v3.src.strategy.calibration import predict_frame_with_calibration, train_model_with_tail_calibration
from v3.src.strategy.hybrid_stack import (
    BEST_TREND_CONFIG,
    build_funding_overlay_frames,
    build_funding_premium_overlay_frames,
    load_base_feature_frames,
    load_v2_runtime_config,
    write_summary_json,
)
from v3.src.strategy.trend_runtime import (
    PositionState,
    apply_market_snapshot,
    apply_trade_action,
    build_market_state_payload,
    detect_regime_from_row,
    prepare_runtime_frame,
)
from v3.src.strategy.types import SignalEnvelope, SignalType
from v3.src.utils.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=1095)
    parser.add_argument("--replay-hours", type=int, default=24 * 30)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--log-path", default="v3/data/paper/replay_cycles.jsonl")
    parser.add_argument("--out", default="v3/data/paper/replay_quality_report.json")
    return parser

def build_variant_frames(config: dict, variant: str, days: int):
    base_frames = load_base_feature_frames(config, days=days)
    if variant == "base":
        return base_frames
    if variant == "funding":
        frames, _ = build_funding_overlay_frames(base_frames)
        return frames
    if variant == "funding_premium":
        frames, _ = build_funding_premium_overlay_frames(base_frames)
        return frames
    raise ValueError(f"Unsupported variant: {variant}")


def main() -> None:
    args = build_parser().parse_args()
    v3_config = load_config()
    v2_config = load_v2_runtime_config()

    optimization_path = Path(v3_config.get("paper", {}).get("optimization_summary_path", "v3/data/optimization/funding_premium_threshold_optimization.json"))
    with open(optimization_path) as handle:
        optimization = json.load(handle)

    variant = str(optimization.get("variant", "funding_premium"))
    best = optimization.get("best", {})
    calibration_method = str(best.get("calibration_method", v3_config.get("optimization", {}).get("default_calibration_method", "raw")))

    full_frames = build_variant_frames(v2_config, variant, args.days)
    replay_len = min(args.replay_hours, min(len(frame) for frame in full_frames.values()) - 500)
    train_frames = {pair: frame.slice(0, len(frame) - replay_len) for pair, frame in full_frames.items()}
    replay_frames = {pair: frame.tail(replay_len) for pair, frame in full_frames.items()}

    model, feature_names, calibration_bundle, train_metrics = train_model_with_tail_calibration(
        v2_config,
        train_frames,
        calibration_method=calibration_method,
        calibration_fraction=float(v3_config.get("optimization", {}).get("calibration_fraction", 0.2)),
    )
    predicted_frames = {
        pair: predict_frame_with_calibration(
            model,
            prepare_runtime_frame(
                frame,
                entry_period=int(BEST_TREND_CONFIG["entry_period"]),
                exit_period=int(BEST_TREND_CONFIG["exit_period"]),
                atr_period=int(BEST_TREND_CONFIG["atr_period"]),
            ),
            calibration_bundle,
            feature_names=feature_names,
        )
        for pair, frame in replay_frames.items()
    }

    engine = DecisionEngine(v3_config)
    engine.long_score_threshold = float(best.get("decision_long_threshold", engine.long_score_threshold))
    engine.short_score_threshold = float(best.get("decision_short_score_threshold", engine.short_score_threshold))
    position_states = {pair.replace("USDT", "USD"): PositionState() for pair in predicted_frames}

    log_path = Path(args.log_path)
    if log_path.exists():
        log_path.unlink()

    replay_hours = min(len(frame) for frame in predicted_frames.values())
    for idx in range(replay_hours):
        payload = {
            "executed_at": datetime.now(tz=UTC).isoformat(),
            "variant": variant,
            "calibration_method": calibration_bundle.method,
            "thresholds": {
                "decision_long_threshold": engine.long_score_threshold,
                "decision_short_score_threshold": engine.short_score_threshold,
            },
            "decisions": [],
        }

        for pair_name, frame in predicted_frames.items():
            row = frame.row(idx, named=True)
            pair = pair_name.replace("USDT", "USD")
            regime = detect_regime_from_row(row)
            score = float(row["pred_long_prob"])
            signal_timestamp = str(row["timestamp"])
            price = float(row["close"])
            market_state = build_market_state_payload(
                row,
                position_states.get(pair, PositionState()),
                atr_stop_mult=float(BEST_TREND_CONFIG["atr_stop_mult"]),
            )

            envelopes = [
                SignalEnvelope(
                    source=f"replay:{variant}",
                    pair=pair,
                    timeframe="1h",
                    signal_type=SignalType.RISK,
                    payload={"trading_halted": False},
                    confidence=1.0,
                ),
                SignalEnvelope(
                    source=f"replay:{variant}",
                    pair=pair,
                    timeframe="1h",
                    signal_type=SignalType.MARKET_STATE,
                    payload=market_state,
                    confidence=1.0,
                ),
                SignalEnvelope(
                    source=f"replay:{variant}",
                    pair=pair,
                    timeframe="1h",
                    signal_type=SignalType.REGIME,
                    payload={"regime": regime, "variant": variant},
                    confidence=max(score, 1.0 - score),
                ),
                SignalEnvelope(
                    source=f"replay:{variant}",
                    pair=pair,
                    timeframe="1h",
                    signal_type=SignalType.EDGE,
                    payload={"score": score, "variant": variant, "timestamp": signal_timestamp},
                    confidence=max(score, 1.0 - score),
                ),
            ]

            for envelope in envelopes:
                decisions = engine.ingest(envelope)
                if envelope.signal_type != SignalType.EDGE:
                    continue
                for decision in decisions:
                    before_position = position_states.get(pair, PositionState())
                    after_position = apply_trade_action(
                        before_position,
                        decision.action,
                        price=price,
                        timestamp=signal_timestamp,
                        market_state=market_state,
                        atr_stop_mult=float(BEST_TREND_CONFIG["atr_stop_mult"]),
                    )
                    position_states[pair] = after_position
                    payload["decisions"].append(
                        {
                            "pair": pair,
                            "variant": variant,
                            "calibration_method": calibration_bundle.method,
                            "signal_timestamp": signal_timestamp,
                            "reference_price": price,
                            "regime": regime,
                            "edge_score": score,
                            "action": decision.action.value,
                            "reason": decision.reason,
                            "preview_payload": {
                                "pair": pair,
                                "price": price,
                                "position_before": before_position.to_dict(),
                                "position_after": after_position.to_dict(),
                            },
                            "submit_result": {"status": "replay"},
                        }
                    )

            if not payload["decisions"] or payload["decisions"][-1]["pair"] != pair:
                position_states[pair] = apply_market_snapshot(position_states.get(pair, PositionState()), market_state)

        append_cycle_log(log_path, payload)

    cycles = load_cycle_log(log_path)
    decisions = flatten_decisions(cycles)
    price_frames = {pair: frame.select(["timestamp", "close"]) for pair, frame in full_frames.items()}
    evaluation = evaluate_decisions(decisions, price_frames, horizon_hours=args.horizon_hours)
    round_trip = evaluate_round_trip_trades(decisions, price_frames)

    summary = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "variant": variant,
        "calibration_method": calibration_bundle.method,
        "train_metrics": train_metrics,
        "thresholds": {
            "decision_long_threshold": engine.long_score_threshold,
            "decision_short_score_threshold": engine.short_score_threshold,
        },
        "replay_hours": replay_hours,
        "horizon_hours": args.horizon_hours,
        "evaluation": evaluation,
        "round_trip_evaluation": round_trip,
        "source_log_path": str(log_path),
    }
    out_path = Path(args.out)
    write_summary_json(summary, out_path)

    print(f"Replay decisions: {len(decisions)}")
    print(f"Evaluation: {evaluation['overall']}")
    print(f"Round-trip evaluation: {round_trip['closed_trades']}")
    print(f"Saved summary to {out_path.resolve()}")


if __name__ == "__main__":
    main()
