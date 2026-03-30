# March 30, 2026 Brief

## Current Best Signal Stack

- `v2` hybrid trend + ML is the current core.
- Adding Deribit funding improved the held-out result from `+22.83%` to `+34.58%`.
- Polymarket remains a research overlay, but it does not yet have enough overlapping history for a serious 3-year test.

## What Matters Next

- Validate the Deribit result with walk-forward windows, not just one split.
- Add one more external layer that actually has enough history to backtest.
- Promote the stronger version into the `v3` paper-trading path so the server makes the final decision before Kraken execution.

## Data Availability Decision

- Historical open-interest and historical order-book imbalance are not cleanly available at multi-year depth from the easiest public sources we checked.
- Cross-exchange premium is the best next practical layer because multi-year public hourly candles are available and can be aligned to our existing 1H candles.

## Execution Tracks

1. Walk-forward validator for `base`, `base + funding`, and `base + funding + premium`.
2. Coinbase premium collector + premium feature joiner.
3. `v3` paper runtime that emits signals, lets the server decide, and sends Kraken preview/validate payloads.

## Outcome

- Walk-forward winner: `funding + premium`.
- Walk-forward averages:
  - `base`: `+0.40%/month`, AUC `0.5457`
  - `funding`: `-0.02%/month`, AUC `0.5549`
  - `funding + premium`: `+1.02%/month`, AUC `0.5717`
- `v3` paper runtime now:
  - refreshes fresh Kraken 1H candles into DuckDB before inference,
  - loads the walk-forward champion automatically,
  - trains the integrated model and sends server-owned decisions to Kraken in validate-safe mode.
- Latest paper cycle result:
  - all three pairs stayed `HOLD` because the current regime is `range` and edge scores were below execution thresholds.
