# V2 — Feature-First Gradient Boosting

> **TL;DR (read this, skip the rest):**
> 1. Ditch deep learning. Use LightGBM/XGBoost with **heavy feature engineering** (150+ indicators, regime features, cross-asset signals) — the model is only as good as the features.
> 2. Proper financial ML pipeline: purged walk-forward CV, Optuna hyperparameter tuning, custom Polars-based vectorized backtester with realistic slippage/fees.
> 3. Ship fast: retrain daily, predict direction (up/down/flat), size positions with Kelly fraction, run paper trading — iterate weekly on features, not models.

---

## Why V2 Exists

**V1 failed.** It threw every deep learning model at the problem (TFT, LSTM-CNN, PPO reinforcement learning) and the result was an over-engineered system that didn't produce viable trading signals. The models were too complex, training was unstable, and the signal-to-noise ratio in crypto is too low for deep learning to shine without massive data and compute.

**V2's philosophy:** Simple models + great features > complex models + basic features.

---

## What Changed from V1

| Aspect | V1 (Failed) | V2 (This) |
|--------|-------------|-----------|
| Core model | TFT + LSTM-CNN + PPO RL | LightGBM gradient boosting |
| Feature count | ~20 basic indicators | 150+ engineered features |
| Training time | 6-12 hours | 5-15 minutes |
| Regime awareness | None | Rolling stats fed as features |
| Validation | Basic train/test split | Purged walk-forward CV |
| Hyperparameter tuning | Manual | Optuna Bayesian optimization |
| Backtesting | Minimal | Custom vectorized Polars backtester |
| Debuggability | Black box | Feature importance + SHAP |

---

## Architecture

```
v2/
├── README.md                   ← you are here
├── config.yaml                 ← trading pairs, risk params, model hyperparams
├── requirements.txt            ← pinned dependencies
├── src/
│   ├── data/
│   │   ├── fetcher.py          ← CCXT exchange data + yfinance fallback
│   │   ├── store.py            ← DuckDB local storage
│   │   └── universe.py         ← asset universe management
│   ├── features/
│   │   ├── technical.py        ← TA-Lib indicators (RSI, MACD, Bollinger, etc.)
│   │   ├── microstructure.py   ← volume profiles, VWAP, order flow proxies
│   │   ├── cross_asset.py      ← BTC dominance, correlation, lead-lag
│   │   ├── regime.py           ← volatility regime, trend strength, variance ratio
│   │   └── pipeline.py         ← feature assembly + cleaning + selection
│   ├── models/
│   │   ├── lgbm_model.py       ← LightGBM classifier (primary model)
│   │   ├── xgb_model.py        ← XGBoost classifier (secondary/comparison)
│   │   └── meta_labeling.py    ← meta-labeling for bet sizing (optional)
│   ├── validation/
│   │   ├── purged_cv.py        ← purged walk-forward cross-validation
│   │   ├── backtest.py         ← vectorized Polars backtester
│   │   └── metrics.py          ← Sharpe, Sortino, max drawdown, win rate
│   ├── optimization/
│   │   └── tuner.py            ← Optuna hyperparameter optimization
│   ├── strategy/
│   │   ├── signals.py          ← model output → trade signals
│   │   └── position_sizer.py   ← Kelly criterion + risk controls
│   ├── execution/
│   │   └── exchange.py         ← CCXT paper/live execution
│   └── utils/
│       ├── logger.py           ← structured logging
│       └── config.py           ← config loader
├── scripts/
│   ├── train.py                ← training pipeline (fetch → features → fit → evaluate)
│   ├── optimize.py             ← Optuna hyperparameter search
│   ├── backtest.py             ← run backtest with trained model
│   └── paper_trade.py          ← live paper trading loop
└── notebooks/
    └── feature_analysis.ipynb  ← EDA, feature importance, SHAP plots
```

---

## Core Components — Technical Detail

### 1. Feature Engineering (the actual alpha)

This is where V2 lives or dies. The model only learns what features teach it.

#### Technical Indicators (~60 features) — `src/features/technical.py`
Built with **TA-Lib** (C-speed, 158 indicators available). Key groups:

| Category | Indicators | Why |
|----------|-----------|-----|
| Trend | EMA(8,21,55,200), MACD, ADX, Aroon, Parabolic SAR | Direction + trend strength |
| Momentum | RSI(14), Stochastic RSI, Williams %R, CCI, ROC | Overbought/oversold, mean reversion signals |
| Volatility | Bollinger Bands, ATR(14), Keltner Channels, historical vs implied vol ratio | Regime detection, stop placement |
| Volume | OBV, VWAP, MFI, Chaikin Money Flow, A/D line | Smart money tracking |
| Pattern | Candlestick patterns (Doji, Engulfing, Hammer, etc.) | Short-term reversal signals |

**Implementation pattern:**
```python
import talib
import polars as pl
import numpy as np

def add_technical_features(df: pl.DataFrame) -> pl.DataFrame:
    close = df["close"].to_numpy().astype(np.float64)
    high = df["high"].to_numpy().astype(np.float64)
    low = df["low"].to_numpy().astype(np.float64)
    volume = df["volume"].to_numpy().astype(np.float64)

    return df.with_columns([
        pl.Series("rsi_14", talib.RSI(close, timeperiod=14)),
        pl.Series("macd", talib.MACD(close)[0]),
        pl.Series("macd_signal", talib.MACD(close)[1]),
        pl.Series("bb_upper", talib.BBANDS(close)[0]),
        pl.Series("bb_lower", talib.BBANDS(close)[2]),
        pl.Series("adx_14", talib.ADX(high, low, close, timeperiod=14)),
        pl.Series("atr_14", talib.ATR(high, low, close, timeperiod=14)),
        pl.Series("obv", talib.OBV(close, volume)),
        pl.Series("mfi_14", talib.MFI(high, low, close, volume, timeperiod=14)),
        # ... etc
    ])
```

#### Microstructure Features (~30 features) — `src/features/microstructure.py`
Custom-built. These are the alpha features most retail quants miss.

| Feature | Formula / Logic | Why It Matters |
|---------|----------------|----------------|
| VWAP deviation | `(close - vwap) / atr` | Institutional fair value |
| Volume profile | Volume at price buckets over lookback | Support/resistance from actual volume |
| Relative volume | `volume / sma(volume, 20)` | Unusual activity detection |
| Bar-to-bar returns | `log(close / close.shift(n))` for n=1,4,12,24 | Multi-scale momentum |
| High-low range ratio | `(high - low) / close` | Intraday volatility |
| Close position in range | `(close - low) / (high - low)` | Buying/selling pressure |
| Volume-weighted momentum | `returns * relative_volume` | Activity-weighted direction |
| Kyle's lambda proxy | `abs(returns) / volume` | Price impact estimation |

#### Cross-Asset Features (~20 features) — `src/features/cross_asset.py`

| Feature | Logic | Why |
|---------|-------|-----|
| BTC dominance change | `btc_mcap / total_mcap` delta | Risk-on/risk-off regime |
| BTC-ETH correlation (rolling) | 30-bar rolling Pearson | Pair divergence opportunities |
| BTC lead returns | BTC returns lagged 1-4 bars | BTC leads altcoin moves |
| Cross-pair spread z-score | Standardized price ratio | Mean reversion across pairs |

#### Regime Features (~20 features) — `src/features/regime.py`
Instead of a separate regime detector, **feed regime signals as features** and let LightGBM learn the boundaries.

| Feature | Calculation | What It Captures |
|---------|-------------|-----------------|
| Realized vol percentile | `std(returns, 20)` percentile over 252 bars | Low/medium/high vol regime |
| Trend strength | `abs(EMA_21 - EMA_55) / ATR_14` | Trending vs ranging |
| Variance ratio | `var(returns, 20) / (20 * var(returns, 1))` | Mean-reverting vs trending (< 1 = MR) |
| Hurst exponent (rolling) | Rolling 100-bar rescaled range | Persistence vs anti-persistence |
| ADX regime | ADX > 25 = trending, < 20 = ranging | Trend filter |
| Vol-of-vol | `std(rolling_vol, 20)` | Regime instability |

### 2. LightGBM Model — `src/models/lgbm_model.py`

**Why LightGBM over XGBoost:**
- 2-5x faster training (leaf-wise growth vs level-wise)
- Built-in categorical feature support (no one-hot encoding)
- Better implicit regularization for noisy data
- Lower memory usage

**Target variable:** 3-class direction (UP / DOWN / FLAT) based on forward returns exceeding a threshold (e.g., ±0.5% over 12 bars).

**Recommended hyperparameter ranges** (Optuna will search within these):

```python
PARAM_SPACE = {
    "n_estimators": (200, 2000),             # boosting rounds
    "max_depth": (4, 8),                      # shallow trees for noisy data
    "num_leaves": (15, 63),                   # 2^max_depth - 1 max
    "learning_rate": (0.01, 0.1),             # log scale
    "min_child_samples": (50, 200),           # large to prevent overfitting
    "subsample": (0.6, 0.9),                  # row sampling
    "colsample_bytree": (0.5, 0.8),          # feature sampling
    "reg_alpha": (1e-3, 10.0),               # L1 regularization, log scale
    "reg_lambda": (1e-3, 10.0),              # L2 regularization, log scale
    "min_split_gain": (0.01, 1.0),           # minimum gain to split
}
```

**Key library:** `lightgbm >= 4.5` — use `objective="multiclass"`, `metric="multi_logloss"`, `verbosity=-1`.

### 3. Purged Walk-Forward CV — `src/validation/purged_cv.py`

Standard `TimeSeriesSplit` from scikit-learn is **not sufficient** because:
- No purge gap between train and test (data leakage from overlapping labels)
- No embargo period after test (prevents peeking into the future)
- Fixed expanding window (we want sliding window for non-stationarity)

**Custom implementation (~60 lines):**

```python
def purged_walk_forward_cv(
    n_samples: int,
    n_splits: int = 5,
    train_size: int = 5000,    # fixed sliding window
    test_size: int = 500,
    purge_gap: int = 12,       # >= prediction_horizon
    embargo_pct: float = 0.01  # % of train to embargo after test
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate train/test indices with purging and embargo."""
    splits = []
    for i in range(n_splits):
        test_end = n_samples - i * test_size
        test_start = test_end - test_size
        train_end = test_start - purge_gap
        train_start = max(0, train_end - train_size)
        embargo = int(train_size * embargo_pct)

        train_idx = np.arange(train_start, train_end - embargo)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))
    return splits[::-1]  # chronological order
```

### 4. Optuna Hyperparameter Optimization — `src/optimization/tuner.py`

```python
import optuna
import lightgbm as lgb

def objective(trial: optuna.Trial, X, y, cv_splits) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": -1,
        "n_jobs": -1,
    }

    scores = []
    for train_idx, test_idx in cv_splits:
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[test_idx], y[test_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        scores.append(model.best_score_["valid_0"]["multi_logloss"])
    return np.mean(scores)

# Run: optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
# study.optimize(objective, n_trials=100, timeout=3600)
```

### 5. Vectorized Polars Backtester — `src/validation/backtest.py`

No existing backtesting framework supports Polars natively. A custom vectorized backtester is ~100 lines and faster than all alternatives.

**Core logic:**
```python
def backtest(
    df: pl.DataFrame,           # must have: timestamp, close, signal, confidence
    initial_capital: float = 100_000,
    fee_rate: float = 0.001,    # 10bps taker fee (Coinbase)
    slippage_bps: float = 5,    # 5bps slippage estimate
    max_position_pct: float = 0.10,
) -> pl.DataFrame:
    """Vectorized backtest with transaction costs."""
    # 1. Convert signals to positions (with sizing)
    # 2. Calculate returns with fees + slippage on each trade
    # 3. Track equity curve, drawdowns, per-trade PnL
    # Returns DataFrame with: equity, drawdown, daily_returns, trade_log
```

**Key metrics calculated in `src/validation/metrics.py`:**
- Sharpe Ratio (annualized)
- Sortino Ratio
- Max Drawdown (% and duration)
- Win Rate + Profit Factor
- Average Win / Average Loss
- Calmar Ratio
- Total Return vs Buy-and-Hold

### 6. Execution & Risk — `src/strategy/` + `src/execution/`

Carried forward from V1 conceptually, simplified:
- **Position sizing:** Fractional Kelly criterion (quarter-Kelly for safety)
- **Risk controls:** Max position 10%, max exposure 50%, 3% stop-loss, 6% take-profit
- **Execution:** CCXT for paper/live mode, same exchange support as V1

---

## Libraries & Dependencies

```
# Core ML
lightgbm>=4.5,<5.0          # primary model
xgboost>=2.1,<4.0            # secondary/comparison model
scikit-learn>=1.5,<2.0       # preprocessing, metrics, base utilities
optuna>=3.6,<5.0             # Bayesian hyperparameter optimization
shap>=0.45                   # model interpretability

# Feature Engineering
TA-Lib>=0.4.28               # 158 technical indicators (C-speed)
                              # NOTE: requires system install first:
                              #   brew install ta-lib

# Data
polars>=1.0,<2.0             # fast DataFrames (replaces pandas)
duckdb>=1.0,<2.0             # local analytical storage
ccxt>=4.0                    # exchange connectivity
yfinance>=0.2.40             # fallback data source
requests>=2.31               # HTTP for APIs

# Visualization & Analysis
plotly>=5.20                  # interactive charts
dash>=2.17                   # dashboard (optional)
matplotlib>=3.9              # static plots for notebooks

# Infrastructure
pyyaml>=6.0                  # config loading
python-dotenv>=1.0           # env vars
loguru>=0.7                  # structured logging (upgrade from v1)

# Dev/Notebooks
jupyterlab>=4.2              # notebook environment
ipywidgets>=8.1              # interactive notebook widgets
```

**System dependency (macOS):**
```bash
brew install ta-lib
```

---

## Data Pipeline

```
Exchange (CCXT)          Fallback (yfinance)
       ↓                        ↓
    fetcher.py  ←───────── auto-fallback
       ↓
  DuckDB (store.py)  ←── append-only hourly OHLCV
       ↓
  feature pipeline.py  ──→  150+ features per bar
       ↓
  LightGBM model  ──→  probability(UP, DOWN, FLAT)
       ↓
  signals.py  ──→  direction + confidence threshold
       ↓
  position_sizer.py  ──→  Kelly-sized position
       ↓
  exchange.py  ──→  paper/live order
```

---

## Training & Evaluation Workflow

```bash
# 1. Fetch latest data
python scripts/train.py --fetch-only

# 2. Train model with default config
python scripts/train.py

# 3. Run hyperparameter optimization (1 hour)
python scripts/optimize.py --n-trials 100 --timeout 3600

# 4. Backtest best model
python scripts/backtest.py --model checkpoints/best_lgbm.pkl

# 5. Start paper trading
python scripts/paper_trade.py
```

---

## Target Metrics (what "working" means for V2)

| Metric | Target | V1 Actual |
|--------|--------|-----------|
| Direction accuracy | > 55% (after fees, this is profitable) | Unknown/poor |
| Sharpe ratio | > 1.0 (annualized) | N/A |
| Max drawdown | < 15% | N/A |
| Win rate | > 50% | N/A |
| Profit factor | > 1.3 | N/A |
| Training time | < 15 min | 6-12 hours |
| Retrain frequency | Daily | Manual |

---

## Implementation Plan

| Phase | What | Est. Time |
|-------|------|-----------|
| 1 | Data pipeline (fetcher, store, universe) | 2-3 hours |
| 2 | Feature engineering (technical, microstructure, regime, cross-asset) | 4-6 hours |
| 3 | LightGBM model + purged CV + label creation | 2-3 hours |
| 4 | Backtester + metrics | 2-3 hours |
| 5 | Optuna hyperparameter optimization | 1-2 hours |
| 6 | Strategy (signals, position sizing, risk) | 2-3 hours |
| 7 | Execution (CCXT paper trading) | 1-2 hours |
| 8 | Training script + paper trading loop | 1-2 hours |
| **Total** | | **~15-22 hours** |

---

## Key Lessons from V1 (Don't Repeat)

1. **Don't stack models for the sake of it.** V1 had TFT + LSTM-CNN + PPO + XGBoost + Mean Reversion + Momentum. None were individually validated.
2. **Validate before ensembling.** Each component must prove itself in backtesting before combining.
3. **Features > models.** A LightGBM with 150 good features will beat a Transformer with 10 bad features.
4. **Proper cross-validation is non-negotiable.** Without purged walk-forward CV, all results are suspect.
5. **Track costs.** A model with 60% accuracy is unprofitable if fees eat the edge. Always backtest with realistic transaction costs.
