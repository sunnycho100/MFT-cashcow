# Polymarket Production TODO

## Current State

The Polymarket downloader works as a research collector, but it is not production-ready yet.

## TODO

- [ ] Add backoff and retry logic for Gamma and CLOB requests.
- [ ] Add market-quality filtering so we ignore sparse, low-signal, or unusable markets.
- [ ] Add a continuous collector mode instead of only one-shot downloads.
- [ ] Build a candle joiner to align Polymarket series with our 1H crypto candles.
- [ ] Add feature engineering on top of the joined data:
  - probabilities
  - momentum
  - shocks
  - event disagreement
- [ ] Prove with backtests that Polymarket features improve trading returns before they affect live decisions.

## Important Note

This is a working research downloader, not yet a complete Polymarket trading module by itself.
