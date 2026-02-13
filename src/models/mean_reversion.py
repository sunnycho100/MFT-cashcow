"""Mean Reversion model using Ornstein-Uhlenbeck process.

Implements z-score based entry/exit for crypto pairs trading,
with rolling cointegration testing and half-life estimation.
"""

import numpy as np
import polars as pl
from scipy import stats
from typing import Optional

from .base import BaseModel, Signal, SignalDirection
from ..utils.logger import get_logger

logger = get_logger("crypto_trader.models.mean_reversion")


class MeanReversionModel(BaseModel):
    """Ornstein-Uhlenbeck mean reversion model for crypto pairs.

    This model looks for mean-reverting spreads between correlated
    crypto pairs (e.g., BTC/ETH) and generates signals when the
    spread deviates significantly from its mean.

    Parameters (via config):
        lookback_window: Rolling window for z-score calculation.
        entry_zscore: Z-score threshold to enter a position.
        exit_zscore: Z-score threshold to exit a position.
        min_half_life: Minimum acceptable half-life for the spread.
        max_half_life: Maximum acceptable half-life for the spread.
        cointegration_pvalue: P-value threshold for cointegration test.
    """

    def __init__(self, config: dict):
        super().__init__("mean_reversion", config)
        mr_cfg = config.get("models", {}).get("mean_reversion", {})
        self.lookback_window = mr_cfg.get("lookback_window", 100)
        self.entry_zscore = mr_cfg.get("entry_zscore", 2.0)
        self.exit_zscore = mr_cfg.get("exit_zscore", 0.5)
        self.min_half_life = mr_cfg.get("min_half_life", 5)
        self.max_half_life = mr_cfg.get("max_half_life", 120)
        self.coint_pvalue = mr_cfg.get("cointegration_pvalue", 0.05)
        self.pairs = mr_cfg.get("pairs", [["BTC/USDT", "ETH/USDT"]])

        # Fitted parameters
        self._hedge_ratios: dict[str, float] = {}
        self._spread_mean: dict[str, float] = {}
        self._spread_std: dict[str, float] = {}
        self._half_lives: dict[str, float] = {}

    def fit(self, data: pl.DataFrame, pair_data: Optional[dict[str, pl.DataFrame]] = None) -> None:
        """Fit the mean reversion model on historical data.

        For single-asset mean reversion, computes rolling statistics on price.
        For pairs, estimates hedge ratios via OLS and tests cointegration.

        Args:
            data: Primary asset OHLCV data.
            pair_data: Dict mapping pair symbols to their OHLCV DataFrames.
                       Required for pairs trading mode.
        """
        self.validate_data(data)
        close = data["close"].to_numpy().astype(np.float64)

        if pair_data is not None:
            self._fit_pairs(data, pair_data)
        else:
            # Single-asset mean reversion on log prices
            self._fit_single(close, data["close"].name or "asset")

        self._is_fitted = True
        logger.info(f"Mean reversion model fitted. Half-lives: {self._half_lives}")

    def _fit_single(self, close: np.ndarray, symbol: str = "asset") -> None:
        """Fit single-asset mean reversion using log prices."""
        log_prices = np.log(close)
        half_life = self._estimate_half_life(log_prices)
        self._half_lives[symbol] = half_life

        window = min(self.lookback_window, len(log_prices))
        self._spread_mean[symbol] = float(np.mean(log_prices[-window:]))
        self._spread_std[symbol] = float(np.std(log_prices[-window:], ddof=1))

        logger.info(f"Single-asset fit: half_life={half_life:.1f}, "
                     f"mean={self._spread_mean[symbol]:.4f}")

    def _fit_pairs(self, data: pl.DataFrame, pair_data: dict[str, pl.DataFrame]) -> None:
        """Fit pairs trading model — estimate hedge ratios and test cointegration."""
        primary_close = data["close"].to_numpy().astype(np.float64)

        for pair_symbols in self.pairs:
            if len(pair_symbols) < 2:
                continue
            sym_a, sym_b = pair_symbols[0], pair_symbols[1]
            pair_key = f"{sym_a}_{sym_b}"

            if sym_b not in pair_data:
                logger.warning(f"Missing data for {sym_b}, skipping pair")
                continue

            other_close = pair_data[sym_b]["close"].to_numpy().astype(np.float64)

            # Align lengths
            min_len = min(len(primary_close), len(other_close))
            y = np.log(primary_close[-min_len:])
            x = np.log(other_close[-min_len:])

            # OLS hedge ratio
            hedge_ratio = self._ols_hedge_ratio(y, x)
            self._hedge_ratios[pair_key] = hedge_ratio

            # Spread
            spread = y - hedge_ratio * x

            # Test cointegration (ADF on spread)
            is_coint, pvalue = self._test_cointegration(spread)
            logger.info(f"Pair {pair_key}: hedge_ratio={hedge_ratio:.4f}, "
                         f"cointegrated={is_coint}, p={pvalue:.4f}")

            # Half-life
            half_life = self._estimate_half_life(spread)
            self._half_lives[pair_key] = half_life

            # Rolling stats
            window = min(self.lookback_window, len(spread))
            self._spread_mean[pair_key] = float(np.mean(spread[-window:]))
            self._spread_std[pair_key] = float(np.std(spread[-window:], ddof=1))

    def generate_signals(self, data: pl.DataFrame, pair_data: Optional[dict[str, pl.DataFrame]] = None) -> list[Signal]:
        """Generate mean reversion trading signals.

        Args:
            data: Primary asset OHLCV data.
            pair_data: Optional dict of secondary pair DataFrames.

        Returns:
            List of Signal objects with entry/exit signals.
        """
        if not self._is_fitted:
            logger.warning("Model not fitted, returning empty signals")
            return []

        self.validate_data(data)
        signals = []
        close = data["close"].to_numpy().astype(np.float64)
        timestamps = data["timestamp"].to_list()
        current_ts = timestamps[-1] if timestamps else None

        if pair_data is not None:
            signals.extend(self._signals_pairs(data, pair_data, current_ts))
        else:
            signals.extend(self._signals_single(close, current_ts, data))

        return signals

    def _signals_single(self, close: np.ndarray, timestamp, data: pl.DataFrame) -> list[Signal]:
        """Generate signals for single-asset mean reversion."""
        signals = []
        symbol = "asset"
        if symbol not in self._spread_mean:
            return signals

        log_price = np.log(close[-1])
        mean = self._spread_mean[symbol]
        std = self._spread_std[symbol]
        half_life = self._half_lives.get(symbol, float("inf"))

        if std < 1e-10:
            return signals

        zscore = (log_price - mean) / std

        # Check half-life validity
        if not (self.min_half_life <= half_life <= self.max_half_life):
            return signals

        # Determine pair name from data context
        pair_name = data.columns[0] if len(data.columns) > 0 else "UNKNOWN"
        # Use a reasonable default
        pair_name = "BTC/USDT"

        direction = SignalDirection.FLAT
        confidence = 0.0

        if zscore >= self.entry_zscore:
            # Price too high relative to mean → SHORT
            direction = SignalDirection.SHORT
            confidence = min(abs(zscore) / (self.entry_zscore * 2), 1.0)
        elif zscore <= -self.entry_zscore:
            # Price too low relative to mean → LONG
            direction = SignalDirection.LONG
            confidence = min(abs(zscore) / (self.entry_zscore * 2), 1.0)
        elif abs(zscore) <= self.exit_zscore:
            # Near mean → exit (FLAT)
            direction = SignalDirection.FLAT
            confidence = 1.0 - abs(zscore) / self.exit_zscore

        if direction != SignalDirection.FLAT or abs(zscore) <= self.exit_zscore:
            signals.append(Signal(
                direction=direction,
                confidence=confidence,
                pair=pair_name,
                timestamp=timestamp,
                model_name=self.name,
                metadata={
                    "zscore": float(zscore),
                    "half_life": half_life,
                    "spread_mean": mean,
                    "spread_std": std,
                },
            ))

        return signals

    def _signals_pairs(self, data: pl.DataFrame, pair_data: dict[str, pl.DataFrame], timestamp) -> list[Signal]:
        """Generate signals for pairs trading."""
        signals = []
        primary_close = data["close"].to_numpy().astype(np.float64)

        for pair_symbols in self.pairs:
            if len(pair_symbols) < 2:
                continue
            sym_a, sym_b = pair_symbols[0], pair_symbols[1]
            pair_key = f"{sym_a}_{sym_b}"

            if pair_key not in self._spread_mean or sym_b not in pair_data:
                continue

            other_close = pair_data[sym_b]["close"].to_numpy().astype(np.float64)
            hedge = self._hedge_ratios.get(pair_key, 1.0)

            spread = np.log(primary_close[-1]) - hedge * np.log(other_close[-1])
            mean = self._spread_mean[pair_key]
            std = self._spread_std[pair_key]
            half_life = self._half_lives.get(pair_key, float("inf"))

            if std < 1e-10:
                continue

            zscore = (spread - mean) / std

            if not (self.min_half_life <= half_life <= self.max_half_life):
                continue

            direction = SignalDirection.FLAT
            confidence = 0.0

            if zscore >= self.entry_zscore:
                direction = SignalDirection.SHORT  # Spread too wide, short A / long B
                confidence = min(abs(zscore) / (self.entry_zscore * 2), 1.0)
            elif zscore <= -self.entry_zscore:
                direction = SignalDirection.LONG  # Spread too narrow, long A / short B
                confidence = min(abs(zscore) / (self.entry_zscore * 2), 1.0)

            if direction != SignalDirection.FLAT:
                signals.append(Signal(
                    direction=direction,
                    confidence=confidence,
                    pair=f"{sym_a}/{sym_b}",
                    timestamp=timestamp,
                    model_name=self.name,
                    metadata={
                        "zscore": float(zscore),
                        "half_life": half_life,
                        "hedge_ratio": hedge,
                        "spread": float(spread),
                        "spread_mean": mean,
                        "spread_std": std,
                    },
                ))

        return signals

    @staticmethod
    def _ols_hedge_ratio(y: np.ndarray, x: np.ndarray) -> float:
        """Estimate hedge ratio via OLS regression.

        Args:
            y: Dependent variable (log prices of asset A).
            x: Independent variable (log prices of asset B).

        Returns:
            Hedge ratio (slope coefficient).
        """
        x_with_const = np.column_stack([x, np.ones_like(x)])
        beta, _, _, _ = np.linalg.lstsq(x_with_const, y, rcond=None)
        return float(beta[0])

    @staticmethod
    def _estimate_half_life(spread: np.ndarray) -> float:
        """Estimate mean-reversion half-life via AR(1) regression.

        The half-life is -log(2) / log(phi), where phi is the AR(1) coefficient
        of the spread process.

        Args:
            spread: Time series of spread values.

        Returns:
            Estimated half-life in periods.
        """
        if len(spread) < 10:
            return float("inf")

        y = np.diff(spread)
        x = spread[:-1]
        x_with_const = np.column_stack([x, np.ones_like(x)])
        beta, _, _, _ = np.linalg.lstsq(x_with_const, y, rcond=None)

        phi = beta[0]
        if phi >= 0:
            return float("inf")  # Not mean-reverting

        half_life = -np.log(2) / phi
        return float(half_life)

    @staticmethod
    def _test_cointegration(spread: np.ndarray) -> tuple[bool, float]:
        """Test cointegration using Augmented Dickey-Fuller test on spread.

        Args:
            spread: Spread time series.

        Returns:
            Tuple of (is_cointegrated, p_value).
        """
        from statsmodels.tsa.stattools import adfuller

        if len(spread) < 20:
            return False, 1.0

        try:
            result = adfuller(spread, maxlag=int(np.sqrt(len(spread))))
            pvalue = float(result[1])
            return pvalue < 0.05, pvalue
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
            return False, 1.0
