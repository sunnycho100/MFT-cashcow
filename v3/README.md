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
