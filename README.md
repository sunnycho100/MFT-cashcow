# MFT-cashcow

Crypto quant trading research repo with multiple generations of systems.

The project has moved away from "one ML model predicts the market" and toward a hybrid stack:
- market-structure and trend logic as the execution backbone,
- ML as a filter / confidence layer,
- external data overlays where they actually improve results,
- server-owned decision making with Kraken as the execution venue.

## Versions

| Folder | Description | Status |
|--------|-------------|--------|
| `v1/` | Deep learning + classical ensemble experiments | Archived |
| `v2/` | Feature-first LightGBM research stack and 3-year local market-data backbone | Stable research baseline |
| `v3/` | Hybrid server-owned trading system built on `v2` data/features with external overlays and Kraken-safe paper execution | Active development |

## Current v3 Snapshot

As of March 30, 2026, the leading `v3` variant is `funding_premium`:
- base market features from `v2`
- Deribit funding overlay
- Coinbase cross-exchange premium overlay
- calibrated ML probabilities
- trend-aligned Donchian breakout / exit gating
- Kraken validate-safe paper trading path

Current benchmark snapshot:
- 12-window walk-forward: `+1.02%` average monthly return, `0.5717` average AUC
- 180-day replay: `20` closed trades, `60.0%` win rate, `+3.1454%` average round-trip return
- 365-day replay: `27` closed trades, `44.44%` win rate, `+1.2754%` average round-trip return

This is still a research and paper-trading system, not a live-ready production bot.

## Repo Map

- `v3/`: current implementation target
- `v2/`: legacy-but-important feature/data/model stack still used by `v3`
- `research/`: internet research notes, API notes, and strategy comparisons
- `docs/planning/`: dated working notes and progress summaries

## Where To Start

- Read [`v3/README.md`](v3/README.md) for the current system.
- Read [`research/README.md`](research/README.md) for the research trail.
- Read [`docs/planning/3-30-2026-v3-live-alignment-paper-monitor-brief.md`](docs/planning/3-30-2026-v3-live-alignment-paper-monitor-brief.md) for the latest implementation summary.
