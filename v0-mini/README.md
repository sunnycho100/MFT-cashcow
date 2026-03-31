# v0-mini

Small research harness for **long-only spot** backtests on BTC, ETH, and SOL using **EMA golden cross / dead cross** logic on **1h** candles. It shares data with the rest of the repo (v2 DuckDB or v3 Coinbase CSVs) and computes EMAs from close prices—nothing proprietary is stored as “EMA” in the database.

## What it does

Three strategy modes are run together (see `config.yaml` and `scripts/run_v0_mini.py`):

1. **Price vs EMA** — Golden: `close` crosses **above** EMA(`signal`, default 50). Dead: `close` crosses **below** that EMA. Entry only if `close >` EMA(`trend`, default 120) on the entry bar.
2. **EMA(fast) × EMA(slow)** — Golden: EMA(30) crosses above EMA(50). Dead: EMA(30) crosses below EMA(50). Entry only if `close >` EMA(120).
3. **EMA(mid) × EMA(slow)** — Golden: EMA(50) crosses above EMA(120). Dead: EMA(50) crosses below EMA(120). Entry only if `close >` EMA(30).

Costs in the backtest: configurable **fee per side** and **slippage (bps)** on entries and exits (`config.yaml` → `backtest`).

## Data

- **Primary:** `v2/data/v2.duckdb` OHLCV if present (`pair`, `timeframe`, `timestamp`, OHLCV columns).
- **Fallback:** `v3/data/coinbase/ohlcv_*_USD_1h.csv` (same universe as v3 Coinbase snapshots).

EMA columns are **computed in memory** (and can be exported to Parquet under `data/`).

## Run

From the repository root:

```bash
python3 v0-mini/scripts/run_v0_mini.py
```

Options:

- `--csv-only` — skip DuckDB; use only v3 CSVs.
- `--no-export` — do not write `v0-mini/data/*.parquet` or `v0_mini_comparison.json`.

## Outputs

With default settings, the script can write:

- `v0-mini/data/ohlcv_with_ema_<PAIR>.parquet` — candles plus EMA columns used in mode (2).
- `v0-mini/data/v0_mini_comparison.json` — per-strategy summaries and optional monthly series.

A frozen snapshot of benchmark numbers and narrative lives in **`result.md`**.

## Disclaimer

Backtest results are for research. They are not trading advice and do not account for full live frictions, funding, or operational risk.
