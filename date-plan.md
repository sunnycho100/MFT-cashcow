# March 30, 2026 Plan

## Objective

Kick off `v3` as a server-owned trading system that improves on `v2` by separating:
- data collection,
- signal transport,
- decision-making,
- execution through Kraken.

## Today’s Plan

- [x] Re-read the current repo state, especially `v2`.
- [x] Research strategy families worth considering for `v3`.
- [x] Research Kraken and supporting external APIs.
- [x] Create a new `research/` folder with written records.
- [x] Define a `v3` architecture centered on a server-owned decision engine.
- [x] Start coding the `v3` scaffold.
- [x] Add the first real collector: Polymarket historical downloader.
- [ ] Port one baseline strategy from `v2` into `v3`.
- [ ] Build a replay / paper mode around the new `v3` message schema.

## Decisions Made Today

- `v3` will be hybrid, not ML-only.
- Kraken will remain the execution venue.
- Signals can come from multiple venues and data sources, but execution authority stays on the server.
- The first transmission layer will be a simple local/in-memory bus in code, with Redis Streams as the likely first real deployment target.

## Immediate Coding Order

1. Create `v3` package skeleton.
2. Define `SignalEnvelope` and `TradeDecision` types.
3. Create a transport layer.
4. Create a decision engine.
5. Create a Kraken execution gateway.
6. Then add real collectors and a baseline strategy port.

## Outputs Created Today

- `research/README.md`
- `research/01-strategy-map.md`
- `research/02-api-catalog.md`
- `research/03-v3-architecture.md`
- `research/04-polymarket-history.md`
- `v3/README.md`
- `v3/config.yaml`
- `v3/main.py`
- starter `v3/src/` packages for transport, strategy, server, execution, data, models, and utils
- `v3/src/data/polymarket.py`
- `v3/scripts/download_polymarket_history.py`

## Notes

- Research benchmarks recorded today are paper-reported only.
- We still need our own 3-year backtests and paper-forward validation before trusting any module with capital.
- The first external collector now fetches official Polymarket metadata and outcome-price history into reproducible files for later feature joins.
