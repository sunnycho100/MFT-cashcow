# V3 Architecture Research Notes

## Goal

Build `v3` as a server-owned trading system where:
- collectors gather data,
- collectors publish normalized signal messages,
- the server decides,
- Kraken executes,
- risk management and calibration live on the server.

---

## Recommended Architecture

## 1. Collectors

Collectors should be stateless or near-stateless processes that only gather and normalize data.

Examples:
- `kraken_market_collector`
- `coinbase_market_collector`
- `deribit_state_collector`
- `coingecko_regime_collector`
- `polymarket_event_collector`

Their job:
- connect to APIs,
- normalize payloads,
- stamp each message with:
  - source,
  - pair,
  - timeframe,
  - observed timestamp,
  - receive timestamp,
  - feature payload,
  - optional confidence.

They should **not** place trades.

---

## 2. Data Transmission Layer

### Recommended first version

Start with a simple, auditable server-owned queue:
- in-process queue for local dev,
- Redis Streams for the first real deployment,
- NATS JetStream later if we need stronger multi-service semantics.

Why Redis Streams first:
- simple,
- persistent enough for small systems,
- easy replay,
- easy local + cloud story.

### Message format

Each collector publishes a `SignalEnvelope`:

```json
{
  "signal_id": "uuid",
  "source": "kraken_book",
  "pair": "BTC/USD",
  "timeframe": "1m",
  "signal_type": "market_state",
  "observed_at": "2026-03-30T12:00:00Z",
  "received_at": "2026-03-30T12:00:00.150Z",
  "confidence": 0.84,
  "payload": {
    "spread_bps": 3.2,
    "imbalance_5": 0.61,
    "microprice_delta": 0.0008
  }
}
```

---

## 3. Decision Server

The decision server should own:
- latest signal state by pair,
- feature fusion,
- regime logic,
- calibrated thresholds,
- position/risk state,
- execution routing rules.

### Responsibilities

- ingest signals from the bus,
- merge them into a pair snapshot,
- decide:
  - buy,
  - sell,
  - hold,
  - reduce / close,
- log why the decision happened,
- forward executable instructions to Kraken.

### Why server-side decisions are better

- one source of truth,
- easier replay and audit,
- easier risk halts,
- easier champion/challenger deployment,
- no collector can accidentally trade on stale or partial context.

---

## 4. Execution Gateway

The execution gateway should be the only component allowed to hold Kraken trading credentials.

Responsibilities:
- translate normalized trade decisions into Kraken order payloads,
- route orders,
- keep dead-man switch active,
- reconcile fills,
- publish fill / rejection events back onto the bus.

---

## 5. Risk Layer

Server-level risk checks should run before every executable order:
- per-trade risk cap,
- max total exposure,
- per-pair exposure,
- spread / slippage guard,
- trading-halt state,
- shorting eligibility guard,
- leverage / margin eligibility guard.

This is where we enforce:
- `short only when multiple confirmations agree`,
- `no trade if liquidity is weak`,
- `no trade if current data is stale`.

---

## 6. Model Layer

`v3` should use multiple narrow models instead of one big monolith.

Recommended model roles:
- `regime_model`
- `trade_quality_model`
- `short_gate_model`
- later:
  - `pair_spread_model`
  - `event_overlay_model`

The model layer publishes scores, not direct exchange orders.

---

## 7. Validation Layer

Before anything is live:
- walk-forward validation,
- realistic fee and slippage assumptions,
- paper-forward replay,
- shadow mode against live market data,
- execution-quality comparison.

---

## First Coding Milestones

1. Create `v3` package and config.
2. Define signal envelope schema.
3. Implement local signal bus and decision engine.
4. Wrap Kraken execution behind a single gateway.
5. Port one baseline strategy from `v2`:
   - trend/regime with no fancy external data.
6. Add Kraken order book and trade collectors.
7. Add Coinbase and Deribit collectors.
8. Add backtest / replay harness for the same signal envelope schema.

## What Should Not Happen In V3

- No collector should directly call Kraken private trading APIs.
- No raw Reddit feature should become a required signal dependency.
- No single model should be the only reason a trade is opened.
