# Polymarket Integration Backtest

## What Was Built

- A Polymarket-to-candle joiner:
  - `v3/src/data/polymarket_overlay.py`
- An integrated research runner:
  - `v3/scripts/run_polymarket_integrated_backtest.py`

The runner:
- downloads or reuses cached Polymarket market histories,
- aligns them to our 1H crypto candles with a backward-looking join,
- derives overlay features,
- trains a candle-only baseline and a candle+Polymarket model,
- runs both through the same backtest logic.

## Markets Used

- `microstrategy-sells-any-bitcoin-by-march-31-2026`
- `microstrategy-sells-any-bitcoin-by-june-30-2026`
- `microstrategy-sells-any-bitcoin-by-december-31-2026`
- `kraken-ipo-by-march-31-2026`
- `kraken-ipo-by-december-31-2026-513`

## Important Limitation

The usable Polymarket history for the retained open markets starts on `2026-02-28`, while our local candle data currently ends on `2026-03-08`.

That means the first integrated backtest only has about:
- 8 days of overlap,
- about `0.05` months in the final test split.

So the result is useful for pipeline validation, but not for strategy trust.

## First Result

Baseline candle-only model on the overlap:
- AUC: `0.5577`
- combined return: `-1.67%`

Integrated candle + Polymarket model on the same overlap:
- AUC: `0.6394`
- combined return: `-3.89%`

Delta:
- integrated minus baseline: `-2.22%`

## Interpretation

- The integration pipeline works.
- On the currently available overlap, Polymarket did **not** improve returns.
- The AUC improvement is not enough to trust because the sample is far too short.

## Output Artifact

- Saved machine-readable summary:
  - `v3/data/polymarket/integrated_backtest_summary.json`
