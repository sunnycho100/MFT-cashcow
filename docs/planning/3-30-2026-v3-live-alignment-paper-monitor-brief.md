# 3-30-2026 v3 live alignment paper monitor brief

## What changed

- Added calibration and threshold optimization tooling for the `funding_premium` champion.
- Added incremental Deribit/Coinbase cache extension plus early same-candle skip for paper cycles.
- Aligned the live paper path with the profitable trend system by adding Donchian breakout/exit gating, ATR-aware position state, and persisted paper position state across cycles.
- Added paper log evaluation for both fixed-horizon edge and round-trip trade performance.

## Current tuned profile

- Variant: `funding_premium`
- Calibration: `sigmoid`
- Decision long threshold: `0.30`
- Decision short score threshold: `0.35`
- Optimizer profile: `ml_long_threshold=0.30`, `ml_short_confidence_threshold=0.65`

## Replay results

- 30d replay: too small to trust; `1` closed trade, avg round-trip return `-3.4866%`
- 180d replay: `20` closed trades, win rate `60.0%`, avg round-trip return `+3.1454%`, avg hold `55.4h`
- 365d replay: `27` closed trades, win rate `44.44%`, avg round-trip return `+1.2754%`, avg hold `48.78h`
- Takeaway: the old 24h scoring looked misleading for a trend system; round-trip trade scoring is the more useful monitor for this model.

## Paper runtime status

- `paper-once` now loads the optimized profile, respects breakout/exit gates, and saves `position_states` in the artifact.
- Latest paper cycle on local 1H candles produced `HOLD` on BTC/USD, ETH/USD, and SOL/USD because regime and breakout conditions were not aligned.
- Continuous paper loop now skips duplicate-candle cycles much earlier and no longer rebuilds the full model path when there is no new candle.

## Next

- Let the paper loop run long enough to accumulate a real decision log.
- Score that log with round-trip trade metrics, not just 24h forward return.
- If paper behavior stays stable, then add one more strong market-structure layer like open interest or order-book imbalance.
