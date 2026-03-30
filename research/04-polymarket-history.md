# Polymarket History For V3

## Summary

We can backtest Polymarket-based event overlays with official public endpoints.

The useful split is:
- metadata from the Gamma API,
- historical outcome prices from the CLOB API.

For `v3`, Polymarket should stay:
- an event overlay,
- a macro context source,
- not a primary market engine for BTC direction.

## Official Endpoints Used

- Event by slug:
  - `GET https://gamma-api.polymarket.com/events/slug/{slug}`
- Market by slug:
  - `GET https://gamma-api.polymarket.com/markets/slug/{slug}`
- Historical prices:
  - `GET https://clob.polymarket.com/prices-history?market={token_id}&interval=...&fidelity=...`

Key implementation detail:
- The history endpoint works on the outcome token id from `clobTokenIds`, not the event slug itself.

## What We Added

- Downloader module:
  - `v3/src/data/polymarket.py`
- CLI:
  - `v3/scripts/download_polymarket_history.py`

The downloader:
- fetches event or market metadata,
- parses `outcomes` and `clobTokenIds`,
- downloads outcome-price history,
- writes a flat CSV for backtest joins,
- writes a metadata JSON file for reproducibility.

## Backtest Join Idea

The intended join path is:
- choose event markets relevant to BTC or macro risk,
- download `1h` history,
- resample or align to our crypto candle timestamps,
- derive features such as:
  - `yes_price`,
  - `no_price`,
  - spread between outcomes,
  - 24h change,
  - probability shock,
  - event momentum,
  - event disagreement across related markets.

## Caveats

- Some markets return sparse history.
- Some event pages contain several grouped markets, so each market must be handled separately.
- Public access may reject very bare HTTP clients, so a non-empty `User-Agent` header is important.

## Sources

- <https://docs.polymarket.com/api-reference/events/get-event-by-slug>
- <https://docs.polymarket.com/api-reference/markets/get-market-by-slug>
- <https://docs.polymarket.com/api-reference/markets/get-prices-history>
- <https://docs.polymarket.com/api-reference/rate-limits>
