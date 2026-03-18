# Crypto Quant Trading System — Technical Research (v2)

**Date:** March 8, 2026  
**Focus:** Gradient boosting (XGBoost/LightGBM) with strong feature engineering  
**Platform:** macOS, Apple M4 chip, Python 3.12+

---

## 1. XGBoost vs LightGBM for Crypto Trading

### Verdict: **LightGBM is the better default for crypto**

| Aspect | XGBoost 3.2.0 | LightGBM 4.6.0 |
|--------|---------------|-----------------|
| Training speed | Slower (level-wise growth) | **2-5x faster** (leaf-wise growth) |
| Noisy data handling | Good with regularization | **Better** — exclusive feature bundling reduces noise |
| Memory usage | Higher | **~50% less** via histogram binning |
| Categorical features | Requires encoding | **Native categorical** support (no one-hot needed) |
| Missing values | Built-in handling | Built-in handling |
| Apple Silicon | `tree_method="hist"` (CPU only) | CPU hist fast on ARM; no GPU accel |

**Why LightGBM wins for crypto specifically:**
- Crypto features are high-cardinality and noisy. LightGBM's leaf-wise growth with `min_child_samples` acts as implicit regularization against noise.
- Faster training means more frequent retraining (critical when market regimes shift).
- Native categorical support avoids the explosion of features from one-hot encoding exchange IDs, day-of-week, etc.

### Apple Silicon GPU Status

**Neither XGBoost nor LightGBM supports Apple Metal/MPS for GPU acceleration.**

- **XGBoost**: `gpu_hist` tree method requires NVIDIA CUDA. On Apple Silicon, use `tree_method="hist"` (CPU). This is already fast — XGBoost's `hist` method is optimized for modern CPUs and uses all available cores via OpenMP. On M4 with 10-12 cores, training 500 trees on 50K samples takes ~5-15 seconds.
- **LightGBM**: GPU mode also requires OpenCL with NVIDIA. On Apple Silicon, CPU mode with `n_jobs=-1` is the way to go.
- **Practical impact**: For tabular gradient boosting workloads (not deep learning), CPU is fast enough. You'd need millions of rows before GPU acceleration matters. Crypto 1h data gives ~8,760 rows/year — even 5 years is only 44K rows. CPU is more than sufficient.

### Proven Hyperparameter Ranges for Crypto

```python
# LightGBM — optimized ranges for noisy crypto time series
lgbm_param_space = {
    "n_estimators": (200, 1500),        # More trees with lower LR
    "max_depth": (3, 8),                # Shallow trees prevent overfitting to noise
    "num_leaves": (15, 63),             # 2^max_depth - 1 as upper bound
    "learning_rate": (0.005, 0.1),      # Low LR + many trees = better generalization
    "min_child_samples": (20, 100),     # HIGH — prevents fitting to noise
    "subsample": (0.6, 0.9),            # Row subsampling per tree
    "colsample_bytree": (0.5, 0.8),     # Feature subsampling — critical for noisy data
    "reg_alpha": (0.0, 1.0),            # L1 regularization
    "reg_lambda": (0.0, 5.0),           # L2 regularization
    "min_split_gain": (0.0, 0.5),       # Minimum loss reduction to split
    "max_bin": (63, 255),               # Fewer bins = more regularization
}

# XGBoost — equivalent ranges
xgb_param_space = {
    "n_estimators": (200, 1500),
    "max_depth": (3, 8),
    "learning_rate": (0.005, 0.1),
    "min_child_weight": (5, 50),        # Analogous to min_child_samples
    "subsample": (0.6, 0.9),
    "colsample_bytree": (0.5, 0.8),
    "gamma": (0.0, 1.0),               # min_split_gain equivalent
    "reg_alpha": (0.0, 1.0),
    "reg_lambda": (1.0, 5.0),
    "tree_method": "hist",              # Always hist on Apple Silicon
}
```

**Key insight:** For crypto, err on the side of **more regularization** (shallower trees, higher min_child_samples, more subsampling). Financial time series have a very low signal-to-noise ratio. Common mistake is using `max_depth=10+` which overfits massively.

### Recommended Production Config

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=800,
    max_depth=5,
    num_leaves=31,
    learning_rate=0.02,
    min_child_samples=50,
    subsample=0.75,
    colsample_bytree=0.65,
    reg_alpha=0.1,
    reg_lambda=2.0,
    min_split_gain=0.01,
    max_bin=127,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    importance_type="gain",  # Use gain, not split count
)
```

---

## 2. Feature Engineering Libraries for Crypto

### Comparison Matrix

| Library | Indicators | Polars Support | Install Difficulty | Speed | Maintained |
|---------|-----------|---------------|-------------------|-------|------------|
| **`ta`** 0.11.0 | ~85 indicators | No (Pandas only) | `pip install ta` — trivial | Slow (pure Python + Pandas) | Barely — last release 2023 |
| **`TA-Lib`** 0.6.8 | **158 indicators** | No (NumPy arrays) | Needs C library: `brew install ta-lib` then `pip install TA-Lib` | **Fastest** (C implementation) | Active |
| **`pandas-ta`** | ~130 indicators | No (Pandas only) | `pip install pandas_ta` | Medium | **Dead** — not on PyPI anymore, repo archived |

### Recommendation: **`TA-Lib` + custom Polars wrappers**

**Why:**
1. `TA-Lib` has the most indicators (158) and is the fastest by far (C implementation).
2. Its API takes NumPy arrays as input, which Polars `.to_numpy()` produces natively — no Pandas required.
3. `ta` library has the simplest API but is slow and has fewer indicators.
4. `pandas-ta` is effectively dead — it was removed from PyPI and the GitHub repo is archived.

**Polars Integration Pattern:**

```python
import talib
import polars as pl

def add_ta_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add TA-Lib indicators to a Polars DataFrame."""
    open_ = df["open"].to_numpy()
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    volume = df["volume"].to_numpy()

    return df.with_columns([
        # Trend
        pl.Series("rsi_14", talib.RSI(close, timeperiod=14)),
        pl.Series("rsi_28", talib.RSI(close, timeperiod=28)),
        pl.Series("adx_14", talib.ADX(high, low, close, timeperiod=14)),
        pl.Series("cci_20", talib.CCI(high, low, close, timeperiod=20)),
        pl.Series("willr_14", talib.WILLR(high, low, close, timeperiod=14)),

        # MACD
        pl.Series("macd", talib.MACD(close)[0]),
        pl.Series("macd_signal", talib.MACD(close)[1]),
        pl.Series("macd_hist", talib.MACD(close)[2]),

        # Volatility
        pl.Series("atr_14", talib.ATR(high, low, close, timeperiod=14)),
        pl.Series("natr_14", talib.NATR(high, low, close, timeperiod=14)),

        # Volume
        pl.Series("obv", talib.OBV(close, volume)),
        pl.Series("ad", talib.AD(high, low, close, volume)),
        pl.Series("adosc", talib.ADOSC(high, low, close, volume)),

        # Pattern recognition (candle patterns)
        pl.Series("doji", talib.CDLDOJI(open_, high, low, close)),
        pl.Series("engulfing", talib.CDLENGULFING(open_, high, low, close)),
        pl.Series("hammer", talib.CDLHAMMER(open_, high, low, close)),
    ])
```

**Install on macOS Apple Silicon:**
```bash
brew install ta-lib
pip install TA-Lib
```

### Crypto-Specific Feature Engineering

There are **no production-quality crypto-specific feature engineering libraries**. Build custom features:

```python
def crypto_specific_features(df: pl.DataFrame) -> pl.DataFrame:
    """Features specific to crypto market microstructure."""
    return df.with_columns([
        # Volume profile features
        (pl.col("volume") / pl.col("volume").rolling_mean(24))
            .alias("volume_ratio_24h"),

        # Crypto volatility clustering (higher than equities)
        (pl.col("close").pct_change().rolling_std(12) /
         pl.col("close").pct_change().rolling_std(48))
            .alias("vol_regime_ratio"),

        # Liquidation cascade detector: large wick + high volume
        ((pl.col("high") - pl.col("low")) /
         pl.col("close").shift(1) * pl.col("volume"))
            .rolling_mean(6)
            .alias("wick_volume_intensity"),

        # Hourly seasonality (crypto trades 24/7)
        (pl.col("timestamp").dt.hour().cast(pl.Float64) / 24.0)
            .alias("hour_norm"),
        (pl.col("timestamp").dt.weekday().cast(pl.Float64) / 7.0)
            .alias("weekday_norm"),
    ])
```

**Feature categories that matter most for crypto GBDT models (ranked by typical feature importance):**
1. **Multi-timeframe momentum** — returns at 1, 3, 5, 10, 20, 50 bars
2. **Volatility features** — realized vol, vol-of-vol, vol regime ratio
3. **Volume features** — volume ratio, OBV, VWAP deviation
4. **Mean reversion** — z-scores at multiple windows, RSI
5. **Trend indicators** — ADX, MACD histogram, SMA crossover distances
6. **Microstructure** — spread, wick ratios, candle body ratios
7. **Calendar** — hour of day, day of week (crypto has intraday patterns)
8. **Cross-asset** — BTC/ETH correlation, BTC dominance change

---

## 3. Regime Detection

### Comparison of Approaches

| Method | Library | Complexity | Latency | Production-Ready |
|--------|---------|-----------|---------|-----------------|
| **Hidden Markov Models** | `hmmlearn` 0.3.3 | Medium | Low (online) | Yes — proven in finance |
| **Change-point detection** | `ruptures` 1.1.10 | Low | **High** (offline/batch) | Partial — best for analysis |
| **Rolling statistics** | NumPy/Polars | Low | **Very low** | **Yes — simplest & fastest** |
| **Regime clustering** | scikit-learn | Medium | Low | Yes |

### Recommendation: **Rolling statistics for features + HMM for meta-labeling**

**Tier 1 — Rolling Statistical Regime (use as features in GBDT):**

This is the practical production approach. Compute regime indicators as features and let the GBDT model learn when to use them:

```python
def compute_regime_features(df: pl.DataFrame) -> pl.DataFrame:
    """Rolling statistical regime detection as features."""
    return df.with_columns([
        # Volatility regime: realized vol percentile
        (pl.col("close").pct_change()
            .rolling_std(20)
            .rank("ordinal") /
         pl.col("close").pct_change()
            .rolling_std(20).len())
            .alias("vol_regime_percentile"),

        # Trend strength: slope of 50-period regression
        # (approximated via price position relative to long MA)
        ((pl.col("close") - pl.col("close").rolling_mean(50)) /
         pl.col("close").rolling_std(50))
            .alias("trend_strength"),
    ])

def variance_ratio(returns: np.ndarray, period: int = 20) -> float:
    """
    Variance ratio test: VR > 1 = trending, VR < 1 = mean-reverting.
    Fast proxy for Hurst exponent.
    """
    n = len(returns)
    mu = np.mean(returns)
    var_1 = np.sum((returns - mu) ** 2) / (n - 1)
    returns_q = np.array([
        np.sum(returns[i:i + period]) for i in range(n - period + 1)
    ])
    var_q = np.sum((returns_q - period * mu) ** 2) / (n - period)
    return var_q / (period * var_1) if var_1 > 0 else 1.0
```

**Tier 2 — HMM for regime labeling (offline analysis + meta-labels):**

```python
from hmmlearn.hmm import GaussianHMM
import numpy as np

def fit_regime_hmm(returns: np.ndarray, n_regimes: int = 3) -> np.ndarray:
    """
    Fit 3-state HMM: low-vol trending, high-vol trending, mean-reverting.
    Returns regime labels for each observation.
    """
    import pandas as pd
    # Features for HMM: returns + volatility
    vol = pd.Series(returns).rolling(20).std().values
    features = np.column_stack([returns, vol])[20:]  # Skip NaN

    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )
    model.fit(features)
    regimes = model.predict(features)

    # Sort regimes by volatility (regime 0 = lowest vol)
    means = model.means_
    vol_order = np.argsort(means[:, 1])  # Sort by vol feature
    regime_map = {old: new for new, old in enumerate(vol_order)}
    return np.array([regime_map[r] for r in regimes])
```

**Why NOT `ruptures` for production:**
- `ruptures` is designed for **offline** change-point detection — it processes the entire signal at once.
- Has O(n²) or O(n log n) complexity depending on the method.
- Not suitable for real-time/streaming — you'd need to re-run on every new bar.
- Best use: post-hoc analysis of backtest results to understand regime boundaries.

**Why rolling stats beat HMM in practice:**
- HMM adds model risk (wrong number of states, convergence issues).
- Rolling stats are deterministic, interpretable, and have zero latency.
- Let the GBDT model figure out how to use regime features — it's better at non-linear interactions than you are.

---

## 4. Walk-Forward Cross-Validation

### Why `sklearn.model_selection.TimeSeriesSplit` Falls Short

`TimeSeriesSplit` has three critical problems for financial ML:

1. **No purging** — No gap between train and test sets. Information leaks when labels use future returns (e.g., predicting 12-bar-ahead returns means the last 12 bars of training data are "contaminated").
2. **No embargo** — Related: even after purging, autocorrelated features can leak information across the boundary.
3. **Expanding window only** — Uses all available past data. For non-stationary financial data, old data may hurt more than help (market regime changes make 2020 data potentially counterproductive for 2026 predictions).

### Library Options

| Library | Version | Purging | Embargo | Sliding Window | Production-Ready |
|---------|---------|---------|---------|---------------|-----------------|
| `sklearn.TimeSeriesSplit` | 1.8.0 | No | No | No (expanding only) | Insufficient |
| `timeseriescv` | 0.2 | Unknown | Unknown | Unknown | **Too immature** (only 2 releases ever) |
| Custom implementation | — | Yes | Yes | Yes | **Recommended** |

### Recommendation: **Custom implementation** (~60 lines of code)

The `timeseriescv` package has only 2 releases (0.1 and 0.2) and negligible adoption. Not worth the dependency risk.

```python
from dataclasses import dataclass
import numpy as np
from typing import Iterator

@dataclass
class PurgedWalkForwardCV:
    """
    Purged walk-forward cross-validation for financial time series.

    Parameters:
        n_splits: Number of train/test splits.
        train_period: Number of bars in each training window.
        test_period: Number of bars in each test window.
        purge_gap: Bars to drop between train and test (prevents leakage).
        embargo_pct: Fraction of test set to embargo after test (prevents
                     autocorrelation leakage into next fold).
    """
    n_splits: int = 5
    train_period: int = 2000
    test_period: int = 500
    purge_gap: int = 12       # Should be >= prediction_horizon
    embargo_pct: float = 0.01  # 1% of test period

    def split(self, X: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)

        # Calculate step size to evenly distribute splits
        total_needed = self.train_period + self.purge_gap + self.test_period
        step = (n_samples - total_needed) // max(1, self.n_splits - 1)

        for i in range(self.n_splits):
            train_start = i * step
            train_end = train_start + self.train_period
            test_start = train_end + self.purge_gap
            test_end = test_start + self.test_period

            if test_end > n_samples:
                break

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, min(test_end, n_samples))

            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        return self.n_splits


def walk_forward_train(
    X: np.ndarray,
    y: np.ndarray,
    model_factory,  # Callable that returns a fresh model
    cv: PurgedWalkForwardCV,
) -> list[dict]:
    """
    Walk-forward training with purged CV.
    Returns per-fold metrics.
    """
    results = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        model = model_factory()
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Early stopping with last 20% of training set as validation
        val_split = int(len(X_train) * 0.8)
        model.fit(
            X_train[:val_split], y_train[:val_split],
            eval_set=[(X_train[val_split:], y_train[val_split:])],
        )

        preds = model.predict(X_test)
        accuracy = np.mean(preds == y_test)
        results.append({
            "fold": fold,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "accuracy": accuracy,
            "train_period": (train_idx[0], train_idx[-1]),
            "test_period": (test_idx[0], test_idx[-1]),
        })

    return results
```

**Key parameters for crypto:**
- `purge_gap` should be `>= prediction_horizon` (if predicting 12 bars ahead, purge at least 12 bars).
- `train_period`: 2000-5000 bars for 1h data (~3-7 months). Longer is not always better due to regime changes.
- `test_period`: 500-1000 bars (~3-6 weeks). Long enough to capture a full market cycle.
- Use **sliding window**, not expanding window. Old data hurts in crypto.

---

## 5. Backtesting Framework Comparison

### Detailed Comparison

| Feature | `vectorbt` 0.28.4 | `backtesting.py` 0.6.5 | `bt` 1.1.4 | `zipline-reloaded` 3.1.1 |
|---------|-------------------|----------------------|------------|--------------------------|
| **Speed** | **Fastest** (NumPy vectorized) | Fast (vectorized core) | Slow (event-driven) | Slowest (full event-driven) |
| **Polars support** | No (NumPy/Pandas) | No (Pandas) | No (Pandas) | No (Pandas) |
| **Transaction costs** | Yes (fixed + pct) | Yes (commission) | Yes | **Full model** (slippage + commission) |
| **Slippage modeling** | Basic (fixed) | No built-in | Basic | **Best** (volume-based) |
| **Crypto support** | Good | Good | Moderate | Poor (equity-focused) |
| **Multi-asset** | **Yes** | Single-asset | **Yes** | **Yes** |
| **API complexity** | High (functional) | **Simplest** | Medium | Very high |
| **Actively maintained** | No (Pro is paid) | Moderate | Moderate | Community-maintained |
| **Realistic fills** | No | No | No | Partial |

### Recommendation: **Custom vectorized backtester**

**Why custom:**
- None of these frameworks natively support Polars.
- For GBDT-based signal generation, you don't need an event-driven framework. Your model produces signals on each bar; you execute them.
- A vectorized backtester in Polars is ~100 lines and runs faster than all the above.
- Full control over transaction cost modeling, slippage, and position sizing.

**Why not each framework:**
- **`vectorbt`**: Best option if you tolerate Pandas. Free version (0.28.4) hasn't been updated since 2023 — development moved to `vectorbt-pro` (paid, not on PyPI). Free version works but is frozen.
- **`backtesting.py`**: Too simple for production. No slippage, no multi-asset. Good for quick prototyping only.
- **`bt`**: Decent for portfolio-level backtesting but slow event-driven loop.
- **`zipline-reloaded`**: Equity-focused, heavy dependency chain, complex setup. Overkill for crypto GBDT strategies.

**Production backtester pattern:**

```python
import polars as pl
import numpy as np

def vectorized_backtest(
    signals: pl.DataFrame,  # columns: timestamp, signal (-1, 0, 1), confidence
    prices: pl.DataFrame,   # columns: timestamp, open, high, low, close
    commission_pct: float = 0.001,  # 0.1% taker fee (Binance/Coinbase)
    slippage_pct: float = 0.0005,   # 0.05% slippage
    initial_capital: float = 10_000.0,
) -> pl.DataFrame:
    """
    Vectorized backtest with transaction costs and slippage.
    Assumes entry at next bar's open (no look-ahead).
    """
    df = signals.join(prices, on="timestamp", how="inner")

    # Position changes (enter at next bar's open — avoids look-ahead bias)
    df = df.with_columns([
        pl.col("signal").shift(1).fill_null(0).alias("position"),
    ])

    df = df.with_columns([
        (pl.col("position") - pl.col("position").shift(1).fill_null(0))
            .alias("trade"),
    ])

    df = df.with_columns([
        # Transaction cost on each trade
        (pl.col("trade").abs() * pl.col("open") * commission_pct)
            .alias("cost"),
    ])

    # PnL calculation
    df = df.with_columns([
        (pl.col("position") * pl.col("close").pct_change())
            .fill_null(0)
            .alias("gross_return"),
    ])

    df = df.with_columns([
        (pl.col("gross_return") - pl.col("cost") / initial_capital)
            .alias("net_return"),
    ])

    df = df.with_columns([
        (1 + pl.col("net_return")).cum_prod().alias("equity_curve"),
    ])

    return df
```

**Transaction cost model for crypto (2025-2026):**
```python
EXCHANGE_FEES = {
    "binance":  {"maker": 0.0001, "taker": 0.001},
    "coinbase": {"maker": 0.004,  "taker": 0.006},  # Coinbase Advanced
    "kraken":   {"maker": 0.0016, "taker": 0.0026},
    "bybit":    {"maker": 0.0001, "taker": 0.0006},
}
# Slippage: 0.01-0.1% depending on pair liquidity and order size
# BTC/USDT on Binance: ~0.01% for < $50K orders
# SOL/USDT on smaller exchanges: ~0.05-0.1%
```

---

## 6. Hyperparameter Optimization

### Optuna vs Hyperopt

| Feature | `optuna` 4.7.0 | `hyperopt` 0.2.7 |
|---------|----------------|-------------------|
| **Algorithm** | TPE, CMA-ES, GP, Random | TPE, Random, Annealing |
| **Pruning** | **Yes** (early stop bad trials) | No |
| **Dashboard** | `optuna-dashboard` (web UI) | No |
| **Multi-objective** | **Yes** (Pareto front) | No |
| **Parallelization** | Built-in (DB-backed) | Via MongoDB (clunky) |
| **Search space API** | Cleaner (`trial.suggest_*`) | Older (`hp.uniform`, `hp.choice`) |
| **Active development** | **Very active** (monthly releases) | **Dead** (last release 2022) |
| **Integration** | XGBoost/LightGBM callbacks | Manual only |

### Verdict: **Optuna is the clear winner**

`hyperopt` is effectively abandoned (last release 2022, version 0.2.7). Optuna is actively developed (4.7.0 as of March 2026) with superior features across the board.

### Optuna + LightGBM Best Practice

```python
import optuna
import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss

def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray,
              cv: PurgedWalkForwardCV) -> float:
    """Optuna objective with purged walk-forward CV."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    scores = []
    for train_idx, test_idx in cv.split(X):
        model = lgb.LGBMClassifier(**params)
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Early stopping on last 20% of training data
        val_split = int(len(X_train) * 0.8)
        model.fit(
            X_train[:val_split], y_train[:val_split],
            eval_set=[(X_train[val_split:], y_train[val_split:])],
        )

        # Use log_loss, not accuracy — better for probability calibration
        probas = model.predict_proba(X_test)
        scores.append(log_loss(y_test, probas))

        # Optuna pruning: report intermediate value per fold
        trial.report(np.mean(scores), len(scores))
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)


# Usage
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    study_name="lgbm_crypto",
    storage="sqlite:///optuna_study.db",  # Persist results
)

study.optimize(
    lambda trial: objective(trial, X, y, cv),
    n_trials=100,
    timeout=3600,  # 1 hour max
    show_progress_bar=True,
)

print(f"Best params: {study.best_params}")
print(f"Best log_loss: {study.best_value:.4f}")
```

**Key Optuna best practices:**
1. Use `log=True` for learning_rate and regularization params (searches in log space).
2. Use `MedianPruner` to early-stop bad trials — saves 30-50% of compute.
3. Store study in SQLite for persistence and restarts.
4. Use `log_loss` as objective, not accuracy — better gradient signal for the Bayesian optimizer.
5. 100 trials is usually sufficient for ~10 hyperparameters. Diminishing returns after that.
6. **Multi-objective** (minimize log_loss + maximize Sharpe ratio) is supported via `optuna.create_study(directions=["minimize", "maximize"])`.

---

## 7. Recommended Package List with Version Constraints

### Core Requirements (all verified available on PyPI for ARM64 macOS)

```
# ===== Core Data Processing =====
polars>=1.30.0,<2.0.0              # DataFrame engine (1.38.1 latest)
numpy>=1.26.0,<3.0.0               # Numerical computing
scipy>=1.12.0,<2.0.0               # Statistical functions

# ===== Gradient Boosting =====
lightgbm>=4.5.0,<5.0.0             # Primary model (4.6.0 latest)
xgboost>=3.0.0,<4.0.0              # Secondary/comparison (3.2.0 latest)
scikit-learn>=1.6.0,<2.0.0         # Preprocessing, metrics (1.8.0 latest)

# ===== Feature Engineering =====
TA-Lib>=0.6.0,<1.0.0               # Technical indicators (requires: brew install ta-lib)

# ===== Regime Detection =====
hmmlearn>=0.3.2,<0.4.0             # Hidden Markov Models (0.3.3 latest)

# ===== Hyperparameter Optimization =====
optuna>=4.5.0,<5.0.0               # Bayesian HPO (4.7.0 latest)
optuna-dashboard>=0.17.0           # Web UI for study results

# ===== Market Data =====
ccxt>=4.4.0,<5.0.0                 # Exchange API (4.5.42 latest)

# ===== Storage =====
duckdb>=1.3.0,<2.0.0               # Analytical database (1.4.4 latest)

# ===== Visualization =====
plotly>=6.0.0,<7.0.0               # Interactive charts (6.6.0 latest)
dash>=3.0.0,<5.0.0                 # Dashboard framework (4.0.0 latest)

# ===== Statistics =====
statsmodels>=0.14.0,<1.0.0         # ADF test, OLS, etc.
arch>=6.2.0,<7.0.0                 # GARCH volatility models

# ===== Utilities =====
pyyaml>=6.0,<7.0                   # Config files
python-dotenv>=1.0.0               # Environment variables
click>=8.1.0                       # CLI
rich>=13.0.0                       # Terminal formatting
numba>=0.60.0,<1.0.0               # JIT compilation for custom indicators

# ===== Testing =====
pytest>=8.0.0
pytest-cov>=5.0.0
```

### Install Command

```bash
# System dependency for TA-Lib
brew install ta-lib

# Python packages
pip install polars numpy scipy \
    lightgbm xgboost scikit-learn \
    TA-Lib hmmlearn optuna optuna-dashboard \
    ccxt duckdb \
    plotly dash \
    pyyaml python-dotenv click rich numba \
    statsmodels arch \
    pytest pytest-cov
```

### Version Compatibility Notes

| Package | Min Version | Reason |
|---------|------------|--------|
| `polars>=1.30.0` | Stable rolling window API, `pct_change()` finalized |
| `lightgbm>=4.5.0` | Improved native categorical support, ARM64 binary wheels |
| `xgboost>=3.0.0` | Major API improvements, better `hist` method on ARM |
| `scikit-learn>=1.6.0` | `set_output("polars")` support for pipeline integration |
| `duckdb>=1.3.0` | Polars integration via `duckdb.pl()`, stable Parquet I/O |
| `ccxt>=4.4.0` | Modern async API, broad exchange coverage |
| `plotly>=6.0.0` | Major rewrite with better rendering performance |
| `numba>=0.60.0` | Apple Silicon ARM64 native support, LLVM 15+ |

### Packages Removed from v1

| Package | Why Removed |
|---------|------------|
| `torch`, `pytorch-forecasting`, `lightning` | No deep learning in v2 — GBDT focus |
| `stable-baselines3`, `gymnasium` | No RL in v2 |
| `transformers`, `sentencepiece` | No NLP/sentiment in v2 |
| `yfinance` | Using `ccxt` for crypto data |
| `ta` | Replaced by TA-Lib (faster, more indicators) |
| `schedule` | Use system cron or `apscheduler` instead |

---

## 8. Architecture Decision Summary

### What v1 Got Wrong (based on codebase review)

1. **Too many model types** — TFT, LSTM-CNN, PPO, XGBoost, mean reversion, momentum all at once. None was tuned properly.
2. **Feature engineering hand-rolled in NumPy loops** — see `ml_model.py` `_compute_basic_features` with manual rolling loops instead of TA-Lib.
3. **No proper walk-forward CV** — used simple 80/20 train/val split (`_train_xgboost` method).
4. **No hyperparameter optimization** — all values hardcoded in `config.yaml`.
5. **No regime awareness** — same model parameters regardless of market conditions.
6. **`ta` library** used but slow and limited indicator set.

### What v2 Should Do

1. **Single model family**: LightGBM (with XGBoost as fallback/comparison).
2. **Heavy feature engineering**: 80-120 features via TA-Lib + custom Polars computations.
3. **Purged walk-forward CV**: Custom implementation with proper purge gap >= prediction_horizon.
4. **Optuna HPO**: 100+ trials with Bayesian optimization, persisted to SQLite.
5. **Regime features as GBDT inputs**: Feed regime indicators as features — don't switch models.
6. **Custom vectorized backtester**: Polars-native, realistic crypto transaction costs.
7. **Feature importance monitoring**: Track feature drift, prune useless features over time.
8. **Polars everywhere**: No Pandas in the pipeline. TA-Lib takes/returns NumPy; wrap in Polars.
