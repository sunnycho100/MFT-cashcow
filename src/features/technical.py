"""Technical indicator features for trading models.

Computes RSI, MACD, Bollinger Bands, ATR, EMA, SMA, and other
standard technical indicators using Polars for performance.
"""

import numpy as np
import polars as pl
from typing import Optional

from ..utils.logger import get_logger

logger = get_logger("crypto_trader.features.technical")


class TechnicalFeatures:
    """Compute technical analysis indicators on OHLCV data.

    All methods return Polars DataFrames or Series for consistency
    and performance on Apple Silicon.
    """

    @staticmethod
    def compute_all(df: pl.DataFrame) -> pl.DataFrame:
        """Compute all technical indicators and add as columns.

        Args:
            df: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume.

        Returns:
            DataFrame with all technical indicator columns added.
        """
        result = df.clone()

        # Price-based
        result = TechnicalFeatures.add_returns(result)
        result = TechnicalFeatures.add_sma(result, windows=[10, 20, 50, 100])
        result = TechnicalFeatures.add_ema(result, spans=[12, 26, 50])
        result = TechnicalFeatures.add_rsi(result, period=14)
        result = TechnicalFeatures.add_rsi(result, period=28)
        result = TechnicalFeatures.add_macd(result)
        result = TechnicalFeatures.add_bollinger_bands(result, window=20)
        result = TechnicalFeatures.add_atr(result, period=14)
        result = TechnicalFeatures.add_adx(result, period=14)

        # Volume-based
        result = TechnicalFeatures.add_volume_features(result)

        # Momentum
        result = TechnicalFeatures.add_momentum_features(result)

        return result

    @staticmethod
    def add_returns(df: pl.DataFrame, periods: list[int] = None) -> pl.DataFrame:
        """Add return columns for various lookback periods.

        Args:
            df: Input DataFrame with 'close' column.
            periods: List of lookback periods. Defaults to [1, 3, 5, 10, 20].

        Returns:
            DataFrame with return columns added.
        """
        if periods is None:
            periods = [1, 3, 5, 10, 20]

        result = df.clone()
        for p in periods:
            result = result.with_columns(
                (pl.col("close") / pl.col("close").shift(p) - 1)
                .alias(f"return_{p}")
            )
        return result

    @staticmethod
    def add_sma(df: pl.DataFrame, windows: list[int] = None) -> pl.DataFrame:
        """Add Simple Moving Average columns.

        Args:
            df: Input DataFrame.
            windows: List of SMA windows. Defaults to [10, 20, 50].

        Returns:
            DataFrame with SMA columns.
        """
        if windows is None:
            windows = [10, 20, 50]

        result = df.clone()
        for w in windows:
            result = result.with_columns(
                pl.col("close").rolling_mean(window_size=w).alias(f"sma_{w}")
            )
            # Price relative to SMA
            result = result.with_columns(
                ((pl.col("close") - pl.col(f"sma_{w}")) / pl.col(f"sma_{w}"))
                .alias(f"close_rel_sma_{w}")
            )
        return result

    @staticmethod
    def add_ema(df: pl.DataFrame, spans: list[int] = None) -> pl.DataFrame:
        """Add Exponential Moving Average columns.

        Args:
            df: Input DataFrame.
            spans: List of EMA spans. Defaults to [12, 26, 50].

        Returns:
            DataFrame with EMA columns.
        """
        if spans is None:
            spans = [12, 26, 50]

        result = df.clone()
        for s in spans:
            result = result.with_columns(
                pl.col("close").ewm_mean(span=s).alias(f"ema_{s}")
            )
        return result

    @staticmethod
    def add_rsi(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """Add Relative Strength Index.

        Args:
            df: Input DataFrame.
            period: RSI period.

        Returns:
            DataFrame with RSI column.
        """
        result = df.clone()
        delta = pl.col("close") - pl.col("close").shift(1)

        result = result.with_columns([
            delta.clip(lower_bound=0).alias("_gain"),
            (-delta).clip(lower_bound=0).alias("_loss"),
        ])

        result = result.with_columns([
            pl.col("_gain").ewm_mean(span=period).alias("_avg_gain"),
            pl.col("_loss").ewm_mean(span=period).alias("_avg_loss"),
        ])

        result = result.with_columns(
            pl.when(pl.col("_avg_loss") > 0)
            .then(100 - 100 / (1 + pl.col("_avg_gain") / pl.col("_avg_loss")))
            .otherwise(100.0)
            .alias(f"rsi_{period}")
        )

        result = result.drop(["_gain", "_loss", "_avg_gain", "_avg_loss"])
        return result

    @staticmethod
    def add_macd(df: pl.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pl.DataFrame:
        """Add MACD, Signal, and Histogram.

        Args:
            df: Input DataFrame.
            fast: Fast EMA period.
            slow: Slow EMA period.
            signal: Signal line EMA period.

        Returns:
            DataFrame with MACD columns.
        """
        result = df.clone()

        # Ensure EMAs exist
        if f"ema_{fast}" not in result.columns:
            result = result.with_columns(
                pl.col("close").ewm_mean(span=fast).alias(f"ema_{fast}")
            )
        if f"ema_{slow}" not in result.columns:
            result = result.with_columns(
                pl.col("close").ewm_mean(span=slow).alias(f"ema_{slow}")
            )

        result = result.with_columns(
            (pl.col(f"ema_{fast}") - pl.col(f"ema_{slow}")).alias("macd")
        )
        result = result.with_columns(
            pl.col("macd").ewm_mean(span=signal).alias("macd_signal")
        )
        result = result.with_columns(
            (pl.col("macd") - pl.col("macd_signal")).alias("macd_histogram")
        )
        return result

    @staticmethod
    def add_bollinger_bands(df: pl.DataFrame, window: int = 20, n_std: float = 2.0) -> pl.DataFrame:
        """Add Bollinger Bands.

        Args:
            df: Input DataFrame.
            window: Lookback window.
            n_std: Number of standard deviations.

        Returns:
            DataFrame with Bollinger Band columns.
        """
        result = df.clone()

        result = result.with_columns([
            pl.col("close").rolling_mean(window_size=window).alias("bb_middle"),
            pl.col("close").rolling_std(window_size=window).alias("_bb_std"),
        ])

        result = result.with_columns([
            (pl.col("bb_middle") + n_std * pl.col("_bb_std")).alias("bb_upper"),
            (pl.col("bb_middle") - n_std * pl.col("_bb_std")).alias("bb_lower"),
        ])

        # %B indicator
        result = result.with_columns(
            pl.when(pl.col("bb_upper") - pl.col("bb_lower") > 0)
            .then(
                (pl.col("close") - pl.col("bb_lower"))
                / (pl.col("bb_upper") - pl.col("bb_lower"))
            )
            .otherwise(0.5)
            .alias("bb_pct_b")
        )

        # Bandwidth
        result = result.with_columns(
            pl.when(pl.col("bb_middle") > 0)
            .then((pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_middle"))
            .otherwise(0.0)
            .alias("bb_bandwidth")
        )

        result = result.drop(["_bb_std"])
        return result

    @staticmethod
    def add_atr(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """Add Average True Range.

        Args:
            df: Input DataFrame.
            period: ATR period.

        Returns:
            DataFrame with ATR column.
        """
        result = df.clone()

        # True Range components
        result = result.with_columns([
            (pl.col("high") - pl.col("low")).alias("_tr1"),
            (pl.col("high") - pl.col("close").shift(1)).abs().alias("_tr2"),
            (pl.col("low") - pl.col("close").shift(1)).abs().alias("_tr3"),
        ])

        # True Range = max of the three
        result = result.with_columns(
            pl.max_horizontal("_tr1", "_tr2", "_tr3").alias("true_range")
        )

        result = result.with_columns(
            pl.col("true_range").ewm_mean(span=period).alias(f"atr_{period}")
        )

        # ATR as percentage of close
        result = result.with_columns(
            (pl.col(f"atr_{period}") / pl.col("close")).alias(f"atr_pct_{period}")
        )

        result = result.drop(["_tr1", "_tr2", "_tr3"])
        return result

    @staticmethod
    def add_adx(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """Add Average Directional Index (simplified Polars version).

        Uses numpy for the Wilder smoothing, then adds result back to DataFrame.

        Args:
            df: Input DataFrame.
            period: ADX period.

        Returns:
            DataFrame with ADX column.
        """
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)
        n = len(close)

        adx = np.full(n, np.nan)

        if n < period * 2:
            return df.with_columns(pl.Series(f"adx_{period}", adx))

        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # EMA smoothing (approximating Wilder)
        alpha = 1.0 / period
        atr_s = np.zeros(len(tr))
        pdm_s = np.zeros(len(tr))
        mdm_s = np.zeros(len(tr))

        atr_s[0] = tr[0]
        pdm_s[0] = plus_dm[0]
        mdm_s[0] = minus_dm[0]

        for i in range(1, len(tr)):
            atr_s[i] = alpha * tr[i] + (1 - alpha) * atr_s[i - 1]
            pdm_s[i] = alpha * plus_dm[i] + (1 - alpha) * pdm_s[i - 1]
            mdm_s[i] = alpha * minus_dm[i] + (1 - alpha) * mdm_s[i - 1]

        pdi = 100 * pdm_s / np.where(atr_s > 0, atr_s, 1.0)
        mdi = 100 * mdm_s / np.where(atr_s > 0, atr_s, 1.0)
        di_sum = pdi + mdi
        dx = 100 * np.abs(pdi - mdi) / np.where(di_sum > 0, di_sum, 1.0)

        # Smooth DX to get ADX
        adx_vals = np.zeros(len(dx))
        adx_vals[0] = dx[0]
        for i in range(1, len(dx)):
            adx_vals[i] = alpha * dx[i] + (1 - alpha) * adx_vals[i - 1]

        adx[1:] = adx_vals

        return df.with_columns(pl.Series(f"adx_{period}", adx))

    @staticmethod
    def add_volume_features(df: pl.DataFrame) -> pl.DataFrame:
        """Add volume-based features.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with volume features.
        """
        result = df.clone()

        # Volume moving averages
        result = result.with_columns([
            pl.col("volume").rolling_mean(window_size=20).alias("volume_sma_20"),
            pl.col("volume").rolling_mean(window_size=50).alias("volume_sma_50"),
        ])

        # Volume ratio (current vs average)
        result = result.with_columns(
            pl.when(pl.col("volume_sma_20") > 0)
            .then(pl.col("volume") / pl.col("volume_sma_20"))
            .otherwise(1.0)
            .alias("volume_ratio")
        )

        # Volume trend
        result = result.with_columns(
            pl.when(pl.col("volume_sma_50") > 0)
            .then(pl.col("volume_sma_20") / pl.col("volume_sma_50"))
            .otherwise(1.0)
            .alias("volume_trend")
        )

        # On-Balance Volume (OBV) direction
        result = result.with_columns(
            pl.when(pl.col("close") > pl.col("close").shift(1))
            .then(pl.col("volume"))
            .when(pl.col("close") < pl.col("close").shift(1))
            .then(-pl.col("volume"))
            .otherwise(0)
            .cum_sum()
            .alias("obv")
        )

        return result

    @staticmethod
    def add_momentum_features(df: pl.DataFrame) -> pl.DataFrame:
        """Add momentum-based features.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with momentum features.
        """
        result = df.clone()

        # Rate of Change
        for period in [10, 20]:
            result = result.with_columns(
                ((pl.col("close") - pl.col("close").shift(period))
                 / pl.col("close").shift(period) * 100)
                .alias(f"roc_{period}")
            )

        # Williams %R
        result = result.with_columns(
            pl.when(
                pl.col("high").rolling_max(window_size=14)
                - pl.col("low").rolling_min(window_size=14) > 0
            ).then(
                (pl.col("high").rolling_max(window_size=14) - pl.col("close"))
                / (pl.col("high").rolling_max(window_size=14)
                   - pl.col("low").rolling_min(window_size=14))
                * -100
            ).otherwise(-50.0)
            .alias("williams_r")
        )

        # Stochastic %K
        result = result.with_columns(
            pl.when(
                pl.col("high").rolling_max(window_size=14)
                - pl.col("low").rolling_min(window_size=14) > 0
            ).then(
                (pl.col("close") - pl.col("low").rolling_min(window_size=14))
                / (pl.col("high").rolling_max(window_size=14)
                   - pl.col("low").rolling_min(window_size=14))
                * 100
            ).otherwise(50.0)
            .alias("stoch_k")
        )

        result = result.with_columns(
            pl.col("stoch_k").rolling_mean(window_size=3).alias("stoch_d")
        )

        return result
