# MFT-Cashcow V2 — Feature-First Gradient Boosting

> **TL;DR:**
> LightGBM 3-class direction classifier (UP/DOWN/FLAT) trained on 59 engineered features from hourly crypto OHLCV data.
> Fetches 3 years of data from Coinbase, trains on 1 year, backtests on held-out 20%.
> First run: -0.9% return vs -25.6% buy & hold — preserved 99% of capital during a bear period.

---

## Why V2 Exists

**V1 failed.** It stacked deep learning models (TFT, LSTM-CNN, PPO reinforcement learning) and produced a system that never traded. The models were overfit, training was unstable, and crypto's low signal-to-noise ratio crushed the deep learning approach without massive data and compute.

**V2's philosophy:** Simple models + great features > complex models + basic features.

| Aspect | V1 (Failed) | V2 (This) |
|--------|-------------|-----------|
| Core model | TFT + LSTM-CNN + PPO RL | LightGBM gradient boosting |
| Features | ~20 basic indicators | 59 engineered features |
| Training time | 6-12 hours | ~1 second |
| Validation | basic train/test split | 80/20 chronological + early stopping |
| Backtesting | minimal | vectorized with fees + slippage |
| Debuggability | black box | feature importance built-in |

---

## Directory Structure

```
v2/
├── README.md                ← you are here
├── config.yaml              ← all configuration (pairs, model params, risk)
├── requirements.txt         ← Python dependencies
├── app.py                   ← Textual TUI dashboard (main entry point)
├── test_pipeline.py         ← end-to-end pipeline test script
├── scripts/
│   └── fetch_data.py        ← CLI to fetch OHLCV data from exchange
├── data/
│   └── v2.duckdb            ← local DuckDB database (auto-created)
├── checkpoints/
│   ├── lgbm_model.pkl       ← trained model (auto-created)
│   └── lgbm_meta.json       ← model metadata + metrics (auto-created)
└── src/
    ├── data/
    │   ├── fetcher.py        ← CCXT paginated data fetcher
    │   └── store.py          ← DuckDB storage layer
    ├── features/
    │   └── pipeline.py       ← 59-feature engineering pipeline
    ├── models/
    │   └── lgbm_model.py     ← LightGBM classifier + checkpoint persistence
    ├── validation/
    │   └── backtest.py       ← vectorized backtester with transaction costs
    ├── strategy/
    │   └── paper_trade.py    ← paper trading logic
    └── utils/
        ├── config.py         ← YAML config loader
        └── logger.py         ← structured logging
```

---

## Full Pipeline (Step by Step)

### Step 1: Data Fetching

**File:** `src/data/fetcher.py` + `scripts/fetch_data.py`

```
Coinbase (CCXT)  →  paginated hourly OHLCV  →  DuckDB
```

- **Exchange:** Coinbase (public API, no key needed)
  - Fallback chain: `coinbase → kraken → binanceus → binance`
  - Binance is blocked in the US (451 error), Kraken ignores the `since` parameter
  - yfinance as last-resort fallback
- **Pairs:** BTC/USDT, ETH/USDT, SOL/USDT
  - Auto-resolves naming differences (e.g., `BTC/USDT → BTC/USD` on exchanges that use USD)
- **Timeframe:** 1h candles
- **History:** 1,095 days (~3 years)
- **Pagination:** Fetches in batches of 1,000 candles, advances cursor by last timestamp + 1 candle
- **Storage:** DuckDB with `(pair, timeframe, timestamp)` primary key — upserts on save, fast analytical reads
- **Current data:** 78,789 total rows (26,275 each for BTC & ETH, 26,239 for SOL)
- **Date range:** 2023-03-10 → 2026-03-09

**Usage:**
```bash
# Fetch all 3 pairs, 3 years
python3 scripts/fetch_data.py --days 1095 --pairs BTC/USDT ETH/USDT SOL/USDT

# Check what's stored
python3 scripts/fetch_data.py --summary
```

### Step 2: Feature Engineering

**File:** `src/features/pipeline.py`

Converts raw OHLCV into **59 features** across 8 categories. All computed using TA-Lib (C-speed) and NumPy.

| Category | Count | Features | Purpose |
|----------|-------|----------|---------|
| **Trend** | 12 | EMA(8,21,55,200), SMA(50,200), MACD/signal/hist, ADX, +DI/-DI, SAR, Aroon up/down | Direction & trend strength |
| **Momentum** | 10 | RSI(7,14), Stoch K/D, StochRSI K/D, Williams %R, CCI, ROC, MOM, UltOsc | Overbought/oversold, mean reversion |
| **Volatility** | 8 | Bollinger (upper/lower/width/%B), ATR, NATR, Keltner (upper/lower) | Regime detection, stop placement |
| **Volume** | 5 | OBV, MFI, A/D line, ADOSC, relative volume | Smart money tracking |
| **Returns** | 7 | Log returns at 1, 2, 4, 8, 12, 24, 48 bar lags | Multi-scale momentum |
| **Microstructure** | 4 | Bar range, close-in-range, body ratio, vol-weighted momentum | Price action quality |
| **Regime** | 5 | Realized vol (20,50), trend strength, variance ratio, ADX regime | Let LightGBM learn regime boundaries |
| **Candlestick** | 4 | Doji, Engulfing, Hammer, Morning Star | Short-term reversal patterns |
| | **59 total** | | |

**Key design decisions:**
- Regime features are fed as numeric inputs (not a separate detector) — LightGBM learns the thresholds itself
- Variance ratio < 1 indicates mean-reverting, > 1 indicates trending
- Relative volume = `volume / SMA(volume, 20)` — detects unusual activity
- All NaN rows from lookback warmup are dropped (no imputation)

### Step 3: Label Creation

**File:** `src/models/lgbm_model.py` → `create_labels()`

Creates a 3-class target variable:

```
Forward return = (close[t + horizon] - close[t]) / close[t]

If forward_return >  +0.5%  →  UP   (class 2)
If forward_return <  -0.5%  →  DOWN (class 0)
Otherwise                   →  FLAT (class 1)
```

- **Horizon:** 12 bars (12 hours with 1h candles)
- **Threshold:** ±0.5% (`direction_threshold` in config)
- Rows where forward return can't be computed (last 12 bars) are dropped

**Typical class distribution (1 year BTC):**

| Class | Count | Percentage |
|-------|-------|------------|
| DOWN | 2,770 | 31.7% |
| FLAT | 3,183 | 36.4% |
| UP | 2,790 | 31.9% |

Reasonably balanced — no need for class weighting.

### Step 4: Model Training

**File:** `src/models/lgbm_model.py` → `train()`

- **Model:** `LGBMClassifier` (multiclass, 3 classes)
- **Split:** Chronological 80/20 (no shuffle — respects time ordering)
- **Early stopping:** 50 rounds of no improvement on validation set
- **Training data:** Last 365 days (configurable via `train_days`)
- **Training time:** ~1 second on Apple M4

**Hyperparameters (from config.yaml):**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `n_estimators` | 500 | max boosting rounds (early stopping usually stops at ~8) |
| `max_depth` | 6 | shallow trees for noisy data |
| `num_leaves` | 31 | complexity control |
| `learning_rate` | 0.05 | step size |
| `min_child_samples` | 100 | large to prevent overfitting |
| `subsample` | 0.8 | row sampling per tree |
| `colsample_bytree` | 0.7 | feature sampling per tree |
| `reg_alpha` | 0.1 | L1 regularization |
| `reg_lambda` | 1.0 | L2 regularization |

**Checkpoint persistence:**
- Model saved to `checkpoints/lgbm_model.pkl` (pickle)
- Metadata saved to `checkpoints/lgbm_meta.json` (training time, accuracy, feature names, per-class metrics)
- On next launch, `load()` restores the model — no re-training needed

### Step 5: Prediction

**File:** `src/models/lgbm_model.py` → `predict()`

Adds 5 columns to the DataFrame:
- `pred_class` — integer (0, 1, 2)
- `pred_prob_down` — probability of DOWN
- `pred_prob_flat` — probability of FLAT
- `pred_prob_up` — probability of UP
- `pred_label` — string ("DOWN", "FLAT", "UP")

### Step 6: Backtesting

**File:** `src/validation/backtest.py`

Vectorized bar-by-bar simulation with realistic transaction costs.

**Trading rules:**
- **LONG** when `pred_class == UP` and `prob_up >= 0.45`
- **SHORT** when `pred_class == DOWN` and `prob_down >= 0.45`
- **FLAT** otherwise (no position)
- Position size: 10% of equity per trade (`max_position_pct`)

**Cost model:**
- Taker fee: 10 bps per side (0.1%) — 20 bps round-trip
- Slippage: 5 bps per execution

**Metrics calculated:**
- Total return (%) vs buy & hold (%)
- Sharpe ratio (annualized for hourly data, √8760)
- Max drawdown (%)
- Win rate, avg win %, avg loss %
- Profit factor (gross wins / gross losses)
- Final equity

### Step 7: Paper Trading

**File:** `src/strategy/paper_trade.py`

Fetches the latest 15 days of data per pair, builds features on the most recent bar, and generates a signal:
- **BUY** if predicted UP with >45% confidence
- **SELL** if predicted DOWN with >45% confidence
- **HOLD** otherwise

---

## First Training Results

```
Training: 6,994 train / 1,749 test rows, 59 features
Early stopped at iteration 8 (of 500)

Accuracy: 39.28% (vs 33% random baseline)

Per-class performance:
  DOWN:  P=0.402  R=0.050  F1=0.089   — very conservative, rarely predicts
  FLAT:  P=0.457  R=0.572  F1=0.508   — captures the majority class well
  UP:    P=0.345  R=0.627  F1=0.445   — biased toward UP predictions

Backtest (last 20%, ~1,750 bars):
  Total Return:   -0.90%
  Buy & Hold:     -25.58%
  Sharpe Ratio:   -1.71
  Max Drawdown:   -1.61%
  Total Trades:   44
  Win Rate:       50.0%

Top 10 features by importance:
  realized_vol_50       47
  ad_line               42
  obv                   34
  sma_200               29
  realized_vol_20       28
  natr_14               24
  trend_strength        21
  adx_14                19
  log_return_48         19
  sma_50                18
```

**Interpretation:** The model preserved 99% of capital during a period where buy & hold lost 25%. It's heavily biased toward FLAT/UP and almost never predicts DOWN (5% recall). The early stopping at iteration 8 suggests the model isn't finding strong patterns — expected for a first iteration without hyperparameter tuning.

---

## How to Use

### Prerequisites

```bash
# System dependency (macOS)
brew install ta-lib libomp

# Python dependencies
cd v2
python3 -m pip install -r requirements.txt
```

### Fetch Data (first time only)

```bash
cd v2
python3 scripts/fetch_data.py --days 1095 --pairs BTC/USDT ETH/USDT SOL/USDT
```

### Launch the TUI

```bash
cd v2
python3 app.py
```

The TUI has three pinned summary panels at the top (DATA / MODEL / BACKTEST) and a scrolling log below.

**Controls:**

| Key | Action |
|-----|--------|
| `t` | Train Model (skips if already trained) |
| `b` | Run Backtest |
| `p` | Paper Trade (fetches live data, generates signals) |
| `d` | Show Data Info |
| `q` | Quit |

### Run Pipeline Without TUI

```bash
cd v2
python3 test_pipeline.py
```

---

## Configuration

All parameters are in `config.yaml`:

```yaml
trading:
  pairs: [BTC/USDT, ETH/USDT, SOL/USDT]
  primary_timeframe: 1h

data:
  db_path: ./data/v2.duckdb
  fetch_days: 1095        # 3 years of history
  train_days: 365         # train on last 1 year
  exchange: coinbase

models:
  lgbm:
    prediction_horizon: 12        # predict 12 bars ahead
    direction_threshold: 0.005    # ±0.5% = UP/DOWN
    n_estimators: 500
    max_depth: 6
    learning_rate: 0.05

risk:
  max_position_pct: 0.10         # 10% per trade
  stop_loss_pct: 0.03            # 3% stop loss
  take_profit_pct: 0.06          # 6% take profit
  kelly_fraction: 0.25           # quarter-Kelly sizing
```

---

## Tech Stack

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| Model | LightGBM | 4.6.0 | Gradient boosting classifier |
| Features | TA-Lib | 0.6.8 | 158 technical indicators (C-speed) |
| DataFrames | Polars | 1.38+ | Fast columnar operations (replaces pandas) |
| Storage | DuckDB | 1.4+ | Local analytical database |
| Exchange | CCXT | 4.5+ | Coinbase data fetching |
| Metrics | scikit-learn | 1.8+ | accuracy, classification report |
| TUI | Textual | 8.0+ | Terminal dashboard |
| Config | PyYAML | 6.0+ | YAML config loading |
| Runtime | Python | 3.13 | macOS Apple M4 |

**System dependencies:** `brew install ta-lib libomp`

---

## Data Flow Diagram

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Coinbase    │────→│  fetcher.py  │────→│  DuckDB      │
│  (CCXT)     │     │  paginated   │     │  store.py    │
│  public API │     │  1000/batch  │     │  78K+ rows   │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                │
                                    load_ohlcv(last 365 days)
                                                │
                                                ▼
                                       ┌────────────────┐
                                       │  pipeline.py   │
                                       │  59 features   │
                                       │  TA-Lib + NumPy│
                                       └───────┬────────┘
                                               │
                                    create_labels(±0.5%, 12h)
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │  lgbm_model.py  │
                                      │  LGBMClassifier │
                                      │  3-class: U/D/F │
                                      └───────┬─────────┘
                                              │
                              ┌───────────────┼───────────────┐
                              │               │               │
                              ▼               ▼               ▼
                     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
                     │  backtest.py │ │ paper_trade  │ │  checkpoint  │
                     │  vectorized  │ │  live signal │ │  .pkl + .json│
                     │  fees+slip   │ │  BUY/SELL/   │ │  persistence │
                     │  Sharpe, DD  │ │  HOLD        │ │              │
                     └──────────────┘ └──────────────┘ └──────────────┘
                              │               │
                              └───────┬───────┘
                                      ▼
                              ┌──────────────┐
                              │   app.py     │
                              │  Textual TUI │
                              │  pinned      │
                              │  summary     │
                              └──────────────┘
```

---

## Next Steps (Not Yet Implemented)

- **Purged walk-forward CV** — proper time-series cross-validation with purge gaps (currently just 80/20 split)
- **Optuna hyperparameter tuning** — Bayesian search over LightGBM params
- **SHAP analysis** — model interpretability, per-prediction explanations
- **Cross-asset features** — BTC dominance, pair correlations, lead-lag signals
- **Multi-pair training** — train on all 3 pairs simultaneously
- **Continuous paper trading** — hourly loop instead of snapshot
- **Daily retraining** — automated retrain with latest data

---

## Key Lessons from V1

1. **Don't stack models without validating each one individually.** V1 had TFT + LSTM-CNN + PPO + XGBoost — none were proven alone.
2. **Features > model complexity.** A LightGBM with 59 good features beats a Transformer with 10 bad features.
3. **Proper CV is non-negotiable.** Without respecting time ordering, all results are suspect.
4. **Track costs.** A model with 60% accuracy is unprofitable if fees eat the edge.
5. **Start small and iterate.** V2 trains in 1 second, not 12 hours — experiment fast.
