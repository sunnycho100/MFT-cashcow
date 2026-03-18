# Plan: Self-Healing Autonomous Crypto Trading System

Transform the current static LightGBM system into a **self-improving, continuously-adapting trading bot** that runs autonomously on AWS, trades Kraken spot margin (long + short), and automatically detects when its model is degrading and retrains itself. The architecture uses **trend-following as the core edge** (proven, rules-based) with **ML as a regime filter and trade quality gate** (not the signal generator). The system monitors its own performance and triggers retraining when drift, loss streaks, or calibration decay are detected — no human intervention needed.

**Constraints:** $500-2K capital, Kraken spot margin (BTC/ETH/SOL, up to 5-10x), 30% max drawdown, AWS (~$20-30/mo).

---

## Phase 1: Core Infrastructure — Data Engine + Kraken Execution *(blocks everything)*

### Step 1.1 — Continuous Data Pipeline
- Add `fetch_incremental()` to `v2/src/data/fetcher.py` — query DuckDB for latest timestamp per pair, fetch only new candles since then
- Add Kraken as first-class exchange (currently defaults to Coinbase → fallback chain)
- Cron: run every hour on AWS to keep DuckDB current

### Step 1.2 — Kraken Spot Margin Execution Service
- NEW `v2/src/execution/kraken_client.py` — Kraken REST API wrapper:
  - HMAC-SHA512 auth signing (nonce + API-Sign)
  - `place_order()`, `cancel_order()`, `get_open_orders()`, `get_balance()`, `get_trade_history()`
  - `cancel_all_after(timeout)` — dead-man switch (if bot dies, Kraken cancels pending orders in 90s)
  - `validate_order()` — preflight with `validate=true`
- Pairs: BTC/USD, ETH/USD, SOL/USD (Kraken uses `/USD` for margin, not `/USDT`)
- Margin: 2-3x leverage (conservative for micro account — max available is 5-10x)
- Shorts: `side=sell` + `leverage=2` opens a short margin position

### Step 1.3 — Position & Risk Manager
- NEW `v2/src/execution/risk_manager.py`:
  - **Global halt**: equity drops 30% below high-water mark → close all, disable trading, alert
  - **Daily loss limit**: 8% → halt for 24 hours
  - **Per-trade risk**: size = `equity × risk_pct / (entry - stop)`, capped
  - **Margin ceiling**: never use >60% of available margin
  - **Dead-man heartbeat**: `cancel_all_after(90)` every 60 seconds
  - **Loss-streak cooldown**: 3 consecutive losses → halve position size for next 3 trades

### Step 1.4 — Order Router
- NEW `v2/src/execution/order_router.py`:
  - **Limit orders for entries** (maker fee 0.16% vs taker 0.26%) — critical for $500-2K
  - If limit not filled in 3 bars → convert to market
  - **Market orders for exits** — don't risk missing stop-losses
  - Fill reconciliation with expected state

---

## Phase 2: Self-Healing ML Loop *(the core innovation, depends on 1.1)*

This is the answer to the key question: *how does the model sustain itself and continuously improve?*

### Step 2.1 — Drift Detection Module
- NEW `v2/src/models/drift_detector.py` using `river` library:
  - **Feature drift (ADWIN)**: Monitor top 10 features' distributions. Feed each bar's feature values into per-feature ADWIN detectors. If ≥3 features simultaneously drift → flag
  - **Performance drift (Page-Hinkley)**: Feed each trade's P&L. Detects when mean P&L is shifting downward (losing streak isn't just randomness)
  - **Calibration drift**: Rolling Brier score of predictions vs outcomes (window=50 bars). If score increases >0.05 from baseline → flag
  - **Regime shift**: Rolling ATR/ADX distributions vs training-time distributions using KS-test (`scipy.stats.ks_2samp`). p-value < 0.01 → flag

### Step 2.2 — Retraining Triggers *(any single one fires retraining)*
- NEW `v2/src/models/retrain_manager.py`:
  1. Feature drift detected (ADWIN flags ≥3 features)
  2. Performance drift detected (Page-Hinkley fires)
  3. Calibration drift (Brier score deteriorated)
  4. 5+ consecutive losing trades
  5. Rolling 30-trade Sharpe drops below -0.5
  6. **Scheduled weekly retrain** (Sunday 00:00 UTC) — baseline even without drift
  7. No trades for 14+ days (model too conservative, needs recalibration)
  - **Cooldown**: min 48h between retrains (prevent thrashing)
  - **Emergency halt**: 3 retrains in a row don't improve → halt trading, alert human

### Step 2.3 — Champion/Challenger Retraining
- Extend `v2/src/models/lgbm_model.py` with `retrain()`:
  1. Save current champion as `checkpoints/champion_YYYYMMDD.pkl`
  2. Load latest 365 days from DuckDB
  3. Compute features → create labels → train new "challenger"
  4. Walk-forward backtest challenger on last 60 days OOS (purge gap = 12 bars)
  5. **Promotion criteria (ALL must pass):**
     - Profit factor > champion's on same OOS window
     - Sharpe ≥ champion's
     - Max drawdown ≤ 1.2× champion's
     - Brier score ≤ champion's
  6. Promote → replace champion, reset drift detectors, log
  7. Reject → keep champion, extend cooldown to 72h, log why

### Step 2.4 — Incremental Refinement (mild drift)
- LightGBM supports `init_model` — start from existing weights, add 100 trees on new data
- Use for **mild drift** (1-2 features shifted, calibration slightly off)
- Full retrain for **severe drift** (multiple features, losing streaks, regime shift)
- Guard: cap total trees at 2000 to prevent model bloat; if exceeded → full retrain

---

## Phase 3: Strategy Layer — Trend Core + ML Filter *(depends on Phase 2)*

### Step 3.1 — Regime-Aware Strategy Engine
- NEW `v2/src/strategy/strategy_engine.py`:
  - **Regime classification** (rules-based + ML confirmation):
    - `TRENDING_UP`: ADX > 20, EMA_21 > EMA_55, price > EMA_21
    - `TRENDING_DOWN`: ADX > 20, EMA_21 < EMA_55, price < EMA_21
    - `RANGING`: ADX < 20 → **don't trade** (this alone blocks ~60% of false signals)
  - **Signals** (trend-following core):
    - LONG: Donchian 20-day breakout above + TRENDING_UP + ML confidence > threshold
    - SHORT: Donchian 20-day breakout below + TRENDING_DOWN + ML pred_long_prob < 0.25 + ADX > 25 + 4H EMA confirms down + margin budget available
    - EXIT: 4× ATR trailing stop, or opposite channel breach, or time exit (48 bars max, 72 for margin cost reasons)

### Step 3.2 — Confidence-Tiered Position Sizing (adapted for $500-2K)

| Conviction | Equity % | With 2x margin | On $1K account |
|-----------|---------|----------------|----------------|
| High (>0.70) | 15% | 30% exposure | ~$150 per trade |
| Medium (0.55-0.70) | 8% | 16% exposure | ~$80 |
| Low (0.45-0.55) | 5% or skip | 10% exposure | ~$50 |

### Step 3.3 — Multi-Pair Portfolio
- Trade BTC/USD, ETH/USD, SOL/USD simultaneously
- Max total exposure: 60% of equity across all pairs
- Correlation filter: if BTC and ETH signal simultaneously, take the one with higher conviction (they're ~0.85 correlated)
- SOL treated independently (lower correlation, stronger trending nature — was the star performer in backtest)

---

## Phase 4: Autonomous Cloud Deployment (AWS) *(depends on Phases 1-3)*

### Step 4.1 — AWS Architecture
- **EC2 t3.micro** (~$8.50/mo) — sufficient for hourly cron + lightweight LightGBM
- Ubuntu 22.04 + Python 3.11 + DuckDB (single-file DB, no separate database needed)
- S3 for weekly model checkpoint backups
- Single Python process with `apscheduler` (simpler than Lambda, maintains state)

### Step 4.2 — Cron Schedule

| Schedule | Task |
|----------|------|
| Every 60s | Dead-man switch heartbeat to Kraken |
| Every 1h (HH:02) | Fetch new candle, compute features |
| Every 1h (HH:03) | Run signals, check positions, execute |
| Every 1h (HH:05) | Update drift detectors |
| On trade close | Feed P&L to performance drift detector |
| Sunday 00:00 UTC | Scheduled retrain (champion/challenger) |
| On drift trigger | Emergency retrain |
| Daily 06:00 UTC | Performance report → Telegram |

### Step 4.3 — Telegram Alerting (free, instant)
- NEW `v2/src/utils/telegram.py`
- Alerts: trade open/close with P&L, drift detected, retrain started/completed, system halt, heartbeat failure

### Step 4.4 — Deployment
- `systemd` service for auto-restart on crash
- Env vars for Kraken API keys + Telegram bot token (never in code)
- NEW `v2/deploy/` — service file, env template

---

## Phase 5: Validation Before Live *(parallel with Phase 4)*

### Step 5.1 — Self-Healing Backtest
- NEW `v2/src/validation/self_healing_backtest.py`
- Simulate full loop on historical data: train → trade → detect drift → retrain → continue
- Must show: self-healing system reduces losses during Aug 2025 – Mar 2026 bear period vs static model

### Step 5.2 — Paper Trading (2-4 weeks)
- Full system on AWS with `validate=true` on all Kraken orders (no real money)
- Collect: would-be trades, drift events, retrain events, uptime metrics

### Step 5.3 — Micro-Live (2 weeks)
- $20-50 per trade positions
- Compare actual fills vs paper expectations
- If fill quality matches → scale to full sizing

---

## Verification

1. **Unit tests**: Drift detector fires on synthetic shifted data; retrain manager respects cooldown; risk manager halts at DD limit
2. **Integration test**: Full loop on 6mo historical: fetch → features → signal → backtest → drift → retrain → valid result
3. **Self-healing backtest**: Must outperform static model on Aug 2025 – Mar 2026 bear period
4. **Paper trade**: 2-4 weeks on AWS, zero real capital — system runs 24/7 without crashes
5. **Micro-live**: 2 weeks at $20-50 positions, actual vs expected fill comparison

---

## Key Decisions

- **Trend-following is the core edge, ML is the filter** — the single most impactful architectural decision. Research proved this
- **Kraken spot margin** for shorts (not futures) — BTC/USD, ETH/USD, SOL/USD at 2-3x
- **Dual retraining**: weekly scheduled + event-triggered — model stays fresh even without explicit drift
- **LightGBM `init_model`** for mild drift refinement, full retrain for severe drift
- **Limit orders for entries, market for exits** — saves ~0.10% per round trip on micro account
- **Trade frequency ~2-3/month** is ideal for $500-2K (fewer fees, fewer chances for consecutive losses to wipe account)
- **Max hold 72 hours for margin positions** — Kraken charges rollover fees every 4 hours (~$0.02-0.10 per period); for multi-day trades keep costs in check

---

## File Map

**Existing (modify):**
- `v2/src/data/fetcher.py` — add `fetch_incremental()`, Kraken support
- `v2/src/models/lgbm_model.py` — add `retrain()`, `refine()`, model versioning
- `v2/config.yaml` — add Kraken, margin, AWS, alerting, drift config sections

**New files:**
- `v2/src/execution/kraken_client.py` — Kraken REST API client
- `v2/src/execution/risk_manager.py` — position/risk management
- `v2/src/execution/order_router.py` — order routing and fill tracking
- `v2/src/models/drift_detector.py` — ADWIN + Page-Hinkley + calibration drift
- `v2/src/models/retrain_manager.py` — retraining trigger logic + champion/challenger
- `v2/src/strategy/strategy_engine.py` — regime-aware trend + ML filter strategy
- `v2/src/utils/telegram.py` — alerting
- `v2/src/validation/self_healing_backtest.py` — backtest the self-healing loop
- `v2/main.py` — new entry point with scheduler
- `v2/deploy/` — systemd service, env template
