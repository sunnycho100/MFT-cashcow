# Next Step Options Comparison

## Option 1: Longer-History Polymarket

- Strength:
  - official public history exists
  - good for event and macro overlays
- Weakness:
  - our usable overlap is currently too short
  - relevant crypto-linked markets are sparse and short-lived
  - first integrated run had only about 8 days of overlap

Verdict:
- good overlay later
- not the best next source for a serious backtest right now

## Option 2: Deeper-History Derivatives Data

- Strength:
  - official hourly funding history is available from Deribit
  - covers BTC, ETH, and SOL perpetuals
  - much better historical overlap with our 3-year candles
  - derivatives are where a large share of crypto price discovery happens
- Weakness:
  - funding alone is not enough; it is best used as a positioning/regime feature
  - open-interest history is not as straightforward from the official public API as funding history

Verdict:
- best next option for a real integrated backtest

## Option 3: Extend Local Candle Data Forward

- Strength:
  - easy and necessary
  - repo already supports incremental public OHLC fetching
- Weakness:
  - adds no new alpha by itself
  - helps evaluation, but does not solve signal quality

Verdict:
- should still be done
- but not the best standalone next experiment

## Decision

Best next step:
- Option 2 first

Operational follow-up:
- Option 3 after that, to keep overlap current and to evaluate future overlays more fairly

## Sources

- Polymarket price history:
  - `https://docs.polymarket.com/api-reference/markets/get-prices-history`
- Polymarket market metadata:
  - `https://docs.polymarket.com/api-reference/markets/get-market-by-slug`
- Deribit funding history:
  - `https://docs.deribit.com/api-reference/market-data/public-get_funding_rate_history`
- Deribit book summary / open interest:
  - `https://docs.deribit.com/api-reference/market-data/public-get_book_summary_by_instrument`
