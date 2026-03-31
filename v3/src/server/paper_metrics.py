"""Paper-cycle logging and evaluation helpers."""

from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl


def append_cycle_log(path: str | Path, payload: dict[str, Any]) -> Path:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as handle:
        handle.write(json.dumps(payload) + "\n")
    return log_path


def load_cycle_log(path: str | Path) -> list[dict[str, Any]]:
    log_path = Path(path)
    if not log_path.exists():
        return []
    rows = []
    with open(log_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def flatten_decisions(cycles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for cycle in cycles:
        for decision in cycle.get("decisions", []):
            flattened.append({**decision, "cycle_executed_at": cycle.get("executed_at"), "cycle_variant": cycle.get("variant")})
    return flattened


@dataclass(slots=True)
class OpenTrade:
    pair: str
    side: str
    entry_price: float
    entry_timestamp: datetime


def _build_price_lookup(frame: pl.DataFrame) -> tuple[list[datetime], list[float]]:
    timestamps = [value if isinstance(value, datetime) else datetime.fromisoformat(str(value)) for value in frame["timestamp"].to_list()]
    closes = [float(value) for value in frame["close"].to_list()]
    return timestamps, closes


def _price_at_or_after(target_ts: datetime, lookup: tuple[list[datetime], list[float]]) -> float | None:
    timestamps, closes = lookup
    idx = bisect_left(timestamps, target_ts)
    if idx >= len(closes):
        return None
    return closes[idx]


def evaluate_decisions(
    decisions: list[dict[str, Any]],
    price_frames: dict[str, pl.DataFrame],
    horizon_hours: int = 24,
) -> dict[str, Any]:
    lookups = {pair: _build_price_lookup(frame.sort("timestamp")) for pair, frame in price_frames.items()}
    action_counts = {"buy": 0, "sell": 0, "hold": 0, "close": 0}
    matured = []
    by_pair: dict[str, list[dict[str, Any]]] = {}

    for decision in decisions:
        action = str(decision.get("action", "hold")).lower()
        action_counts[action] = action_counts.get(action, 0) + 1

        if action not in {"buy", "sell"}:
            continue

        raw_pair = str(decision.get("pair", "")).replace("/USD", "/USDT")
        lookup = lookups.get(raw_pair)
        if lookup is None:
            continue

        signal_ts = datetime.fromisoformat(str(decision["signal_timestamp"]))
        if signal_ts.tzinfo is None:
            signal_ts = signal_ts.replace(tzinfo=UTC)
        future_ts = signal_ts + timedelta(hours=horizon_hours)

        current_price = _price_at_or_after(signal_ts, lookup)
        future_price = _price_at_or_after(future_ts, lookup)
        if current_price is None or future_price is None or current_price <= 0:
            continue

        forward_return = future_price / current_price - 1.0
        edge_return = forward_return if action == "buy" else -forward_return
        row = {
            "pair": raw_pair,
            "action": action,
            "signal_timestamp": signal_ts.isoformat(),
            "future_timestamp": future_ts.isoformat(),
            "edge_return_pct": round(edge_return * 100, 4),
            "forward_return_pct": round(forward_return * 100, 4),
            "hit": edge_return > 0,
            "regime": decision.get("regime"),
            "edge_score": decision.get("edge_score"),
        }
        matured.append(row)
        by_pair.setdefault(raw_pair, []).append(row)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"count": 0, "hit_rate": None, "avg_edge_return_pct": None}
        hits = sum(1 for row in rows if row["hit"])
        avg_edge = sum(row["edge_return_pct"] for row in rows) / len(rows)
        buys = [row for row in rows if row["action"] == "buy"]
        sells = [row for row in rows if row["action"] == "sell"]
        return {
            "count": len(rows),
            "hit_rate": round(hits / len(rows), 4),
            "avg_edge_return_pct": round(avg_edge, 4),
            "buy_count": len(buys),
            "sell_count": len(sells),
            "avg_buy_edge_return_pct": round(sum(row["edge_return_pct"] for row in buys) / len(buys), 4) if buys else None,
            "avg_sell_edge_return_pct": round(sum(row["edge_return_pct"] for row in sells) / len(sells), 4) if sells else None,
        }

    pair_summary = {pair: summarize(rows) for pair, rows in by_pair.items()}

    return {
        "horizon_hours": horizon_hours,
        "action_counts": action_counts,
        "matured_trade_decisions": len(matured),
        "overall": summarize(matured),
        "by_pair": pair_summary,
        "sample": matured[:25],
    }


def evaluate_round_trip_trades(
    decisions: list[dict[str, Any]],
    price_frames: dict[str, pl.DataFrame],
) -> dict[str, Any]:
    def parse_ts(value: Any) -> datetime:
        ts = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)

    sorted_decisions = sorted(
        decisions,
        key=lambda row: (
            parse_ts(row.get("signal_timestamp") or row.get("cycle_executed_at")),
            str(row.get("pair", "")),
        ),
    )

    open_positions: dict[str, OpenTrade] = {}
    closed_rows = []

    for decision in sorted_decisions:
        action = str(decision.get("action", "hold")).lower()
        if action not in {"buy", "sell", "close"}:
            continue

        pair = str(decision.get("pair", "")).replace("/USD", "/USDT")
        timestamp = parse_ts(decision.get("signal_timestamp") or decision.get("cycle_executed_at"))
        price = decision.get("reference_price")
        if price is None:
            lookup = _build_price_lookup(price_frames[pair].sort("timestamp")) if pair in price_frames else None
            price = _price_at_or_after(timestamp, lookup) if lookup is not None else None
        if price is None or float(price) <= 0:
            continue
        price = float(price)

        current = open_positions.get(pair)
        if action == "close":
            if current is None:
                continue
            exit_side = current.side
            pnl = price / current.entry_price - 1.0 if exit_side == "long" else current.entry_price / price - 1.0
            closed_rows.append(
                {
                    "pair": pair,
                    "side": exit_side,
                    "entry_price": round(current.entry_price, 4),
                    "exit_price": round(price, 4),
                    "entry_timestamp": current.entry_timestamp.isoformat(),
                    "exit_timestamp": timestamp.isoformat(),
                    "return_pct": round(pnl * 100, 4),
                    "hold_hours": round((timestamp - current.entry_timestamp).total_seconds() / 3600, 2),
                    "closed_by": "close",
                }
            )
            open_positions.pop(pair, None)
            continue

        target_side = "long" if action == "buy" else "short"
        if current is None:
            open_positions[pair] = OpenTrade(pair=pair, side=target_side, entry_price=price, entry_timestamp=timestamp)
            continue

        if current.side == target_side:
            continue

        pnl = price / current.entry_price - 1.0 if current.side == "long" else current.entry_price / price - 1.0
        closed_rows.append(
            {
                "pair": pair,
                "side": current.side,
                "entry_price": round(current.entry_price, 4),
                "exit_price": round(price, 4),
                "entry_timestamp": current.entry_timestamp.isoformat(),
                "exit_timestamp": timestamp.isoformat(),
                "return_pct": round(pnl * 100, 4),
                "hold_hours": round((timestamp - current.entry_timestamp).total_seconds() / 3600, 2),
                "closed_by": "reversal",
            }
        )
        open_positions[pair] = OpenTrade(pair=pair, side=target_side, entry_price=price, entry_timestamp=timestamp)

    marked_open = []
    for pair, current in open_positions.items():
        frame = price_frames.get(pair)
        if frame is None or frame.is_empty():
            continue
        last_timestamp = frame["timestamp"][-1]
        last_price = float(frame["close"][-1])
        pnl = last_price / current.entry_price - 1.0 if current.side == "long" else current.entry_price / last_price - 1.0
        marked_open.append(
            {
                "pair": pair,
                "side": current.side,
                "entry_price": round(current.entry_price, 4),
                "mark_price": round(last_price, 4),
                "entry_timestamp": current.entry_timestamp.isoformat(),
                "mark_timestamp": str(last_timestamp),
                "marked_return_pct": round(pnl * 100, 4),
            }
        )

    def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
        if not rows:
            return {"count": 0, "win_rate": None, "avg_return_pct": None}
        wins = sum(1 for row in rows if float(row[key]) > 0)
        avg_return = sum(float(row[key]) for row in rows) / len(rows)
        avg_hold = sum(float(row.get("hold_hours", 0.0)) for row in rows) / len(rows)
        longs = [row for row in rows if row.get("side") == "long"]
        shorts = [row for row in rows if row.get("side") == "short"]
        return {
            "count": len(rows),
            "win_rate": round(wins / len(rows), 4),
            "avg_return_pct": round(avg_return, 4),
            "avg_hold_hours": round(avg_hold, 2),
            "long_count": len(longs),
            "short_count": len(shorts),
            "avg_long_return_pct": round(sum(float(row[key]) for row in longs) / len(longs), 4) if longs else None,
            "avg_short_return_pct": round(sum(float(row[key]) for row in shorts) / len(shorts), 4) if shorts else None,
        }

    return {
        "closed_trades": summarize(closed_rows, "return_pct"),
        "marked_open_positions": summarize(marked_open, "marked_return_pct"),
        "closed_trade_sample": closed_rows[:25],
        "open_position_sample": marked_open[:25],
    }
