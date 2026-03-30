# MFT-Cashcow V3

`v3` is the next step after `v2`.

Design goals:
- keep Kraken as the execution venue,
- move decision-making onto a server-owned engine,
- treat ML as one input among several,
- support multiple signal sources with one normalized message schema,
- make replay, paper trading, and live trading share the same core interfaces.

## V3 Direction

`v2` proved that:
- simpler models are easier to inspect,
- trend and regime structure matter,
- execution and risk rules matter as much as prediction.

`v3` adds:
- a transmission layer for normalized signals,
- a decision engine that owns the final trade decision,
- an execution gateway for Kraken,
- room for external data sources like order flow, derivatives state, and event overlays.

## Initial Modules

- `src/transport/`: signal bus and message transport
- `src/strategy/`: signal and decision types
- `src/server/`: server-owned decision engine
- `src/execution/`: Kraken execution gateway
- `src/utils/`: logger and config loader

## Current Status

This is an architecture scaffold, not a complete trading system yet.

The first milestone is to:
- feed normalized signals into the server,
- produce structured trade decisions,
- and route those decisions through Kraken in paper / validate mode.

## Polymarket Downloader

`v3` now includes a first external-data collector for Polymarket history:

- client module:
  - `v3/src/data/polymarket.py`
- CLI entrypoint:
  - `v3/scripts/download_polymarket_history.py`

Example:

```bash
python3 v3/scripts/download_polymarket_history.py \
  --event-slug microstrategy-sell-any-bitcoin-in-2025 \
  --interval 1d \
  --fidelity 1440 \
  --output v3/data/polymarket/event_microstrategy-sell-any-bitcoin-in-2025.csv
```

What it does:
- fetches event or market metadata from the Polymarket Gamma API,
- resolves outcome token ids from `clobTokenIds`,
- downloads price history from the CLOB `prices-history` endpoint,
- saves a flat CSV plus a metadata JSON companion file.

This gives us a clean first step toward testing event overlays against the existing crypto candles.
