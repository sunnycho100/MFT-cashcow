# API Catalog For V3

## Executive Summary

`v3` should separate:
- signal collection,
- server-side decision-making,
- execution via Kraken private APIs.

The server should own:
- state,
- risk limits,
- calibrated thresholds,
- Kraken credentials.

Collectors should publish structured signal envelopes to the server, not place trades directly.

---

## 1. Kraken APIs

### What Kraken should do in `v3`

- Live execution.
- Account state.
- Private order and fill updates.
- Primary market data for traded pairs.

### Official APIs to use

- Spot WebSocket v2 order book:
  - `book`
  - use for spread, depth, imbalance, sweep detection
  - <https://docs.kraken.com/api/docs/websocket-v2/book/>

- Spot WebSocket v2 trades:
  - `trade`
  - use for trade flow and aggressor imbalance proxies
  - <https://docs.kraken.com/api/docs/websocket-v2/trade/>

- Spot WebSocket v2 private trading:
  - `add_order`
  - supports `margin`, `reduce_only`, `validate`, and conditional order flows
  - <https://docs.kraken.com/api/docs/websocket-v2/add_order>

- Spot WebSocket auth:
  - `GetWebSocketsToken`
  - token must be used shortly after issuance
  - <https://docs.kraken.com/api/docs/rest-api/get-websockets-token/>
  - <https://docs.kraken.com/api/docs/guides/spot-ws-auth/>

- Dead-man switch:
  - REST `CancelAllOrdersAfter`
  - WS v2 `cancel_after`
  - <https://docs.kraken.com/api/docs/rest-api/cancel-all-orders-after/>
  - <https://docs.kraken.com/api/docs/websocket-v2/cancel_after/>

- Rate limits:
  - <https://docs.kraken.com/api/docs/guides/spot-rest-ratelimits/>
  - <https://docs.kraken.com/api/docs/guides/spot-ratelimits>

### Account eligibility notes

- Kraken Futures eligibility:
  - <https://support.kraken.com/hc/en-us/articles/360023786632-Kraken-Futures-eligibility>
- U.S. margin trading restrictions / ECP requirements:
  - <https://support.kraken.com/articles/360061972272-margin-trading-and-eligible-contract-participant-ecp-self-certification-for-u-s-clients>

### V3 use

- Kraken is the execution venue and one of the market-data venues.
- Even if we use outside data for signals, all final decisions should be routed through a Kraken-owned execution gateway.

---

## 2. Coinbase APIs

### Why use Coinbase

- Good US-accessible cross-exchange benchmark venue.
- Useful for Coinbase premium / lead-lag / quote quality comparisons.

### Official APIs

- Exchange WebSocket feed:
  - public real-time orders and trades
  - `wss://ws-feed.exchange.coinbase.com`
  - <https://docs.cdp.coinbase.com/exchange/websocket-feed>

- Advanced Trade WebSocket:
  - market data and user-order channels
  - `wss://advanced-trade-ws.coinbase.com`
  - <https://docs.cdp.coinbase.com/coinbase-business/advanced-trade-apis/websocket/websocket-overview>
  - channels include `level2` and `market_trades`
  - <https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/websocket/websocket-channels>

- Rate limits / best practices:
  - <https://docs.cdp.coinbase.com/exchange/websocket-feed/rate-limits>
  - <https://docs.cdp.coinbase.com/exchange/websocket-feed/best-practices>

### V3 use

- Cross-exchange premium:
  - Kraken mid vs Coinbase mid,
  - Kraken spread vs Coinbase spread,
  - quote divergence alerts.

---

## 3. Deribit APIs

### Why use Deribit

- Strong derivatives data source.
- Useful even if execution stays on Kraken.
- Good source for funding, open interest, and perpetual market state.

### Official APIs

- API docs home:
  - <https://docs.deribit.com/index.html>

- Book summary by instrument:
  - includes open interest, mark price, bid/ask, and market stats
  - <https://docs.deribit.com/api-reference/market-data/public-get_book_summary_by_instrument>

- Funding rate history:
  - hourly historical funding for perpetuals
  - <https://docs.deribit.com/api-reference/market-data/public-get_funding_rate_history>

- Ticker subscriptions:
  - includes current funding, funding 8h, open interest
  - <https://docs.deribit.com/subscriptions/market-data/tickerinstrument_nameinterval>

- Rate limits:
  - <https://docs.deribit.com/articles/rate-limits>

### V3 use

- Derivatives-state collector:
  - funding,
  - funding changes,
  - open interest,
  - mark/index divergence.

---

## 4. CoinGecko APIs

### Why use CoinGecko

- Market-cap rotation data.
- BTC dominance and global market context.
- Useful for slower regime features.

### Official APIs

- Global market data:
  - `/global`
  - <https://docs.coingecko.com/reference/crypto-global>

- Global market-cap chart:
  - `/global/market_cap_chart`
  - <https://docs.coingecko.com/reference/global-market-cap-chart>

- Coins markets:
  - `/coins/markets`
  - <https://docs.coingecko.com/reference/coins-markets>

### V3 use

- BTC dominance change.
- Market-cap rotation regime.
- Risk-on / risk-off crypto breadth features.

---

## 5. Polymarket APIs

### Why use Polymarket

- Event and macro overlay, not main price engine.

### Official APIs

- Markets and events concepts:
  - <https://docs.polymarket.com/concepts/markets-events>

- Order book:
  - <https://docs.polymarket.com/trading/orderbook>

- Rate limits:
  - <https://docs.polymarket.com/api-reference/rate-limits>

- Geoblock / restrictions:
  - <https://docs.polymarket.com/api-reference/geoblock>

### V3 use

- Event probability overlays only.
- No direct automated trading dependence.

---

## 6. Reddit APIs

### Why use Reddit carefully

- Could help as a low-weight attention feature.
- High noise.
- Policy limitations matter.

### Official references

- Data API wiki:
  - <https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki>

- Data API terms:
  - <https://redditinc.com/policies/data-api-terms>

### Important constraint

- Reddit’s terms prohibit using User Content to train ML / AI models without required permissions.
- That means we should not design `v3` around training directly on Reddit data unless we revisit licensing and usage very carefully.

### V3 use

- Optional attention-only features later.
- Not phase 1.

---

## 7. On-Chain / Analytics APIs

### Why use them

- Slower but structurally useful regime and flow signals.

### Candidate APIs

- Glassnode:
  - exchange balances / flows and on-chain metrics
  - <https://docs.glassnode.com/>

- Dune API:
  - custom on-chain query outputs
  - <https://docs.dune.com/api-reference/overview/introduction>

### V3 use

- Exchange inflow / outflow regime features.
- Stablecoin issuance and treasury flow proxies.
- Not required for first `v3` milestone.

---

## Recommended API Stack For Phase 1

- Execution:
  - Kraken private REST / WS
- Primary market data:
  - Kraken WS `book` + `trade`
- Cross-exchange confirmation:
  - Coinbase WS `level2` + `market_trades`
- Derivatives-state:
  - Deribit funding / OI / ticker
- Slow regime context:
  - CoinGecko global / BTC dominance

## Recommended API Stack For Later

- Event overlay:
  - Polymarket
- On-chain:
  - Glassnode or Dune
- Social:
  - Reddit, only if we can keep the feature policy-safe and low weight
