# Deribit Funding Hybrid Backtest

## What We Did

- Added a reusable Deribit funding downloader:
  - `v3/src/data/deribit.py`
- Added a funding-to-candle join layer:
  - `v3/src/data/deribit_overlay.py`
- Added a comparison runner on top of the stronger positive `v2` hybrid system:
  - `v3/scripts/run_deribit_funding_hybrid_backtest.py`

This test uses:
- the existing `v2` candle features,
- the existing LightGBM filter,
- the existing positive trend-following backtest,
- plus Deribit hourly funding overlays.

## Why This Path Was Chosen

Compared with the other next-step options:
- Polymarket had too little overlapping history for a serious backtest.
- Extending local candles is useful but does not add new alpha by itself.
- Deribit funding has deep official history and much better overlap with our 3-year candle set.

## Coverage

Official Deribit funding history covered all three traded assets:
- BTC perpetual
- ETH perpetual
- SOL perpetual

Cached rows per instrument:
- `25,756`

## Result

Baseline hybrid trend + ML:
- AUC: `0.5702`
- combined portfolio return: `+22.83%`
- monthly average: `+3.24%`

Integrated hybrid + Deribit funding:
- AUC: `0.5822`
- combined portfolio return: `+34.58%`
- monthly average: `+4.91%`

Delta:
- `+11.75%` combined return improvement

## Interpretation

- This is the first external-data integration that improved both:
  - model discrimination
  - portfolio return
- Funding is working best here as a positioning/regime feature, not as a standalone strategy.

## Output Artifact

- Saved machine-readable result:
  - `v3/data/deribit/funding_hybrid_backtest_summary.json`

## Sources

- Deribit funding history:
  - `https://docs.deribit.com/api-reference/market-data/public-get_funding_rate_history`
- Deribit book summary / open interest:
  - `https://docs.deribit.com/api-reference/market-data/public-get_book_summary_by_instrument`
