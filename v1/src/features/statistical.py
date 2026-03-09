"""Statistical features for trading models.

Includes z-score, Hurst exponent, half-life estimation, rolling
covariance, and other statistical measures useful for mean reversion
and regime detection.
"""

import numpy as np
import polars as pl
from scipy import stats as scipy_stats
from typing import Optional

from ..utils.logger import get_logger

logger = get_logger("crypto_trader.features.statistical")


class StatisticalFeatures:
    """Compute statistical features on OHLCV data.

    These features are particularly useful for mean reversion detection,
    regime identification, and ML model inputs.
    """

    @staticmethod
    def compute_all(df: pl.DataFrame) -> pl.DataFrame:
        """Compute all statistical features.

        Args:
            df: OHLCV DataFrame.

        Returns:
            DataFrame with statistical features added.
        """
        result = df.clone()
        result = StatisticalFeatures.add_zscore(result)
        result = StatisticalFeatures.add_log_returns(result)
        result = StatisticalFeatures.add_realized_volatility(result)
        result = StatisticalFeatures.add_skewness_kurtosis(result)
        result = StatisticalFeatures.add_hurst_exponent(result)
        result = StatisticalFeatures.add_half_life(result)
        result = StatisticalFeatures.add_autocorrelation(result)
        result = StatisticalFeatures.add_variance_ratio(result)
        return result

    @staticmethod
    def add_zscore(df: pl.DataFrame, windows: list[int] = None) -> pl.DataFrame:
        """Add z-score of close price relative to rolling mean/std.

        Args:
            df: Input DataFrame.
            windows: Rolling windows. Defaults to [20, 50, 100].

        Returns:
            DataFrame with z-score columns.
        """
        if windows is None:
            windows = [20, 50, 100]

        result = df.clone()
        for w in windows:
            mean = pl.col("close").rolling_mean(window_size=w)
            std = pl.col("close").rolling_std(window_size=w)
            result = result.with_columns(
                pl.when(std > 0)
                .then((pl.col("close") - mean) / std)
                .otherwise(0.0)
                .alias(f"zscore_{w}")
            )
        return result

    @staticmethod
    def add_log_returns(df: pl.DataFrame) -> pl.DataFrame:
        """Add log returns at various frequencies.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with log return columns.
        """
        result = df.clone()
        result = result.with_columns(
            (pl.col("close").log() - pl.col("close").shift(1).log())
            .alias("log_return")
        )

        # Multi-period log returns
        for p in [5, 10, 20]:
            result = result.with_columns(
                (pl.col("close").log() - pl.col("close").shift(p).log())
                .alias(f"log_return_{p}")
            )
        return result

    @staticmethod
    def add_realized_volatility(df: pl.DataFrame, windows: list[int] = None) -> pl.DataFrame:
        """Add realized volatility (rolling std of log returns).

        Args:
            df: Input DataFrame.
            windows: Rolling windows. Defaults to [10, 20, 50].

        Returns:
            DataFrame with realized vol columns.
        """
        if windows is None:
            windows = [10, 20, 50]

        result = df.clone()

        # Ensure log_return exists
        if "log_return" not in result.columns:
            result = result.with_columns(
                (pl.col("close").log() - pl.col("close").shift(1).log())
                .alias("log_return")
            )

        for w in windows:
            result = result.with_columns(
                pl.col("log_return").rolling_std(window_size=w)
                .alias(f"realized_vol_{w}")
            )

        # Vol of vol (volatility clustering indicator)
        if f"realized_vol_{windows[0]}" in result.columns:
            result = result.with_columns(
                pl.col(f"realized_vol_{windows[0]}")
                .rolling_std(window_size=windows[0])
                .alias("vol_of_vol")
            )

        return result

    @staticmethod
    def add_skewness_kurtosis(df: pl.DataFrame, window: int = 50) -> pl.DataFrame:
        """Add rolling skewness and kurtosis of returns.

        Computed via numpy for accuracy, then added back as Polars columns.

        Args:
            df: Input DataFrame.
            window: Rolling window.

        Returns:
            DataFrame with skewness/kurtosis columns.
        """
        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)

        log_ret = np.zeros(n)
        log_ret[1:] = np.diff(np.log(close))

        skew_arr = np.full(n, np.nan)
        kurt_arr = np.full(n, np.nan)

        for i in range(window, n):
            chunk = log_ret[i - window:i]
            if np.std(chunk) > 1e-12:
                skew_arr[i] = float(scipy_stats.skew(chunk))
                kurt_arr[i] = float(scipy_stats.kurtosis(chunk))

        result = df.with_columns([
            pl.Series(f"rolling_skew_{window}", skew_arr),
            pl.Series(f"rolling_kurt_{window}", kurt_arr),
        ])
        return result

    @staticmethod
    def add_hurst_exponent(df: pl.DataFrame, window: int = 100, max_lag: int = 20) -> pl.DataFrame:
        """Add rolling Hurst exponent estimate.

        H < 0.5: Mean-reverting
        H = 0.5: Random walk
        H > 0.5: Trending

        Uses the rescaled range (R/S) method.

        Args:
            df: Input DataFrame.
            window: Rolling window for computation.
            max_lag: Maximum lag for R/S analysis.

        Returns:
            DataFrame with Hurst exponent column.
        """
        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)
        hurst = np.full(n, np.nan)

        for i in range(window, n):
            chunk = close[i - window:i]
            h = StatisticalFeatures._hurst_rs(chunk, max_lag)
            hurst[i] = h

        return df.with_columns(pl.Series("hurst_exponent", hurst))

    @staticmethod
    def _hurst_rs(series: np.ndarray, max_lag: int = 20) -> float:
        """Compute Hurst exponent using R/S method.

        Args:
            series: Price series.
            max_lag: Maximum lag.

        Returns:
            Estimated Hurst exponent.
        """
        if len(series) < max_lag * 2:
            return 0.5

        returns = np.diff(np.log(series))
        if len(returns) < 10 or np.std(returns) < 1e-12:
            return 0.5

        lags = range(2, min(max_lag + 1, len(returns) // 2))
        rs_values = []
        lag_values = []

        for lag in lags:
            n_chunks = len(returns) // lag
            if n_chunks < 1:
                continue

            rs_list = []
            for j in range(n_chunks):
                chunk = returns[j * lag:(j + 1) * lag]
                mean = np.mean(chunk)
                deviate = np.cumsum(chunk - mean)
                r = np.max(deviate) - np.min(deviate)
                s = np.std(chunk, ddof=1)
                if s > 1e-12:
                    rs_list.append(r / s)

            if rs_list:
                rs_values.append(np.mean(rs_list))
                lag_values.append(lag)

        if len(lag_values) < 3:
            return 0.5

        try:
            log_lags = np.log(lag_values)
            log_rs = np.log(rs_values)
            slope, _, _, _, _ = scipy_stats.linregress(log_lags, log_rs)
            return float(np.clip(slope, 0.0, 1.0))
        except Exception:
            return 0.5

    @staticmethod
    def add_half_life(df: pl.DataFrame, window: int = 100) -> pl.DataFrame:
        """Add rolling half-life of mean reversion.

        Estimated via AR(1) regression on log prices.

        Args:
            df: Input DataFrame.
            window: Rolling window.

        Returns:
            DataFrame with half_life column.
        """
        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)
        half_life = np.full(n, np.nan)

        log_prices = np.log(close)

        for i in range(window, n):
            chunk = log_prices[i - window:i]
            y = np.diff(chunk)
            x = chunk[:-1]

            if len(x) < 10 or np.std(x) < 1e-12:
                continue

            x_aug = np.column_stack([x, np.ones_like(x)])
            try:
                beta, _, _, _ = np.linalg.lstsq(x_aug, y, rcond=None)
                phi = beta[0]
                if phi < 0:
                    half_life[i] = float(-np.log(2) / phi)
                else:
                    half_life[i] = np.nan
            except Exception:
                continue

        return df.with_columns(pl.Series("half_life", half_life))

    @staticmethod
    def add_autocorrelation(df: pl.DataFrame, lags: list[int] = None) -> pl.DataFrame:
        """Add rolling autocorrelation of returns.

        Args:
            df: Input DataFrame.
            lags: Autocorrelation lags. Defaults to [1, 5, 10].

        Returns:
            DataFrame with autocorrelation columns.
        """
        if lags is None:
            lags = [1, 5, 10]

        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)
        log_ret = np.zeros(n)
        log_ret[1:] = np.diff(np.log(close))

        result = df.clone()
        window = 50

        for lag in lags:
            ac = np.full(n, np.nan)
            for i in range(window + lag, n):
                chunk = log_ret[i - window:i]
                lagged = log_ret[i - window - lag:i - lag]
                if np.std(chunk) > 1e-12 and np.std(lagged) > 1e-12:
                    corr = np.corrcoef(chunk, lagged)[0, 1]
                    ac[i] = float(corr) if np.isfinite(corr) else 0.0
            result = result.with_columns(pl.Series(f"autocorr_{lag}", ac))

        return result

    @staticmethod
    def add_variance_ratio(df: pl.DataFrame, period: int = 10, window: int = 100) -> pl.DataFrame:
        """Add variance ratio test statistic.

        VR > 1: Trending / momentum
        VR < 1: Mean-reverting
        VR ≈ 1: Random walk

        Args:
            df: Input DataFrame.
            period: Ratio period.
            window: Rolling window.

        Returns:
            DataFrame with variance_ratio column.
        """
        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)
        vr = np.full(n, np.nan)

        for i in range(window, n):
            chunk = np.log(close[i - window:i])
            ret_1 = np.diff(chunk)
            ret_k = chunk[period:] - chunk[:-period]

            var_1 = np.var(ret_1, ddof=1)
            var_k = np.var(ret_k, ddof=1)

            if var_1 > 1e-12:
                vr[i] = (var_k / period) / var_1

        return df.with_columns(pl.Series("variance_ratio", vr))
