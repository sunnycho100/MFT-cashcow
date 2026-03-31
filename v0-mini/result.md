# v0-mini — results record

This file is a **static log** of what was implemented and what the harness reported on a full historical sample available in-repo. Re-running `scripts/run_v0_mini.py` will produce new numbers if data or code changes.

**Recorded:** 2026-03-30

## What was built

- **`config.yaml`** — pairs (BTC/USDT, ETH/USDT, SOL/USDT), EMA periods (30 / 50 / 120), strategy knobs for price-vs-EMA and triple EMA, fees and slippage.
- **`src/data_loader.py`** — load OHLCV from v2 DuckDB when available, else v3 Coinbase hourly CSVs.
- **`src/ema_backtest.py`** — Polars-based EMAs (`ewm_mean`, `adjust=False`), long-only equity simulation, monthly return series, three strategy variants.
- **`scripts/run_v0_mini.py`** — runs all three modes per pair, prints JSON summaries and a compact comparison table, optional export to `data/`.

**Design note:** DuckDB (v2) stores **OHLCV only**. EMAs are **derived from close**; optional Parquet export materializes them under `v0-mini/data/` for inspection.

## Backtest assumptions (snapshot)

| Setting | Value |
|--------|--------|
| Initial capital | $100,000 per asset (independent runs) |
| Fee | 0.1% per side |
| Slippage | 5 bps |
| Bar size | 1h |

## Run output (three strategies, same data window)

Numbers below are **total return %** and **closed round-trips** from the last full multi-strategy run (`--no-export` used only to skip writing files; math is unchanged).

### (1) Price vs EMA(50), trend filter EMA(120)

| Pair | Total return % | Trades |
|------|----------------|--------|
| BTC/USDT | −65.48 | 606 |
| ETH/USDT | −66.05 | 643 |
| SOL/USDT | −27.08 | 620 |

Interpretation: crossing **price** through EMA(50) with an EMA(120) long filter produced **many** trades and **large negative** outcomes on BTC/ETH in this sample; SOL was less bad but still negative.

### (2) EMA(30) × EMA(50), trend filter `close >` EMA(120)

| Pair | Total return % | Trades |
|------|----------------|--------|
| BTC/USDT | −32.80 | 194 |
| ETH/USDT | −32.27 | 186 |
| SOL/USDT | +147.83 | 170 |

Interpretation: classic fast/slow EMA cross with a 120-bar trend filter was **weak on BTC/ETH**, **strong on SOL** over the window (trend-following friendly period for SOL).

### (3) EMA(50) × EMA(120), filter `close >` EMA(30)

| Pair | Total return % | Trades |
|------|----------------|--------|
| BTC/USDT | +2.69 | 106 |
| ETH/USDT | −36.94 | 121 |
| SOL/USDT | +176.18 | 93 |

Interpretation: slower crossover (50 vs 120) with a short EMA(30) entry filter had **fewer trades** than the price/EMA(50) rule; **BTC** turned slightly positive, **ETH** negative, **SOL** strongest in this table.

## Quick comparison (total return %)

| Strategy | BTC | ETH | SOL |
|----------|-----|-----|-----|
| Price × EMA50 | −65.5 | −66.1 | −27.1 |
| EMA30 × EMA50 + EMA120 trend | −32.8 | −32.3 | +147.8 |
| EMA50 × EMA120 + EMA30 filter | +2.7 | −36.9 | +176.2 |

## Caveats

- Results depend on **exact date range** stored in DuckDB or CSV.
- **No** portfolio merge: each asset is a separate $100k run.
- Past backtests do not predict future performance.
