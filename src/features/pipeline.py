"""Feature engineering pipeline.

Orchestrates technical and statistical feature computation,
handles feature selection, normalization, and caching.
"""

import numpy as np
import polars as pl
from typing import Optional

from .technical import TechnicalFeatures
from .statistical import StatisticalFeatures
from ..utils.logger import get_logger

logger = get_logger("crypto_trader.features.pipeline")


class FeaturePipeline:
    """End-to-end feature engineering pipeline.

    Computes technical indicators, statistical features, and derived
    features. Supports incremental updates and feature normalization.

    Args:
        config: Configuration dictionary.
        normalize: Whether to normalize features (z-score normalization).
        norm_window: Rolling window for normalization.
    """

    def __init__(
        self,
        config: dict = None,
        normalize: bool = True,
        norm_window: int = 100,
    ):
        self.config = config or {}
        self.normalize = normalize
        self.norm_window = norm_window
        self._feature_columns: list[str] = []
        self._base_columns = {"timestamp", "open", "high", "low", "close", "volume"}

    def compute(self, data: pl.DataFrame) -> pl.DataFrame:
        """Run the full feature pipeline.

        Args:
            data: Raw OHLCV DataFrame.

        Returns:
            DataFrame with all features computed.
        """
        logger.info(f"Computing features for {len(data)} rows")

        # Validate input
        missing = self._base_columns - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        result = data.clone()

        # Technical features
        result = TechnicalFeatures.compute_all(result)

        # Statistical features
        result = StatisticalFeatures.compute_all(result)

        # Derived / interaction features
        result = self._add_derived_features(result)

        # Store feature column names
        self._feature_columns = [
            c for c in result.columns if c not in self._base_columns
        ]

        # Normalize if requested
        if self.normalize:
            result = self._normalize_features(result)

        # Fill remaining nulls
        result = result.fill_null(0.0)
        result = result.fill_nan(0.0)

        logger.info(f"Feature pipeline complete: {len(self._feature_columns)} features")
        return result

    def compute_incremental(self, existing: pl.DataFrame, new_data: pl.DataFrame) -> pl.DataFrame:
        """Incrementally update features with new data.

        Appends new_data to existing, recomputes features only for
        the tail to avoid full recomputation.

        Args:
            existing: Previously computed feature DataFrame.
            new_data: New raw OHLCV rows.

        Returns:
            Updated feature DataFrame.
        """
        # We need some lookback for rolling calculations
        lookback = 200
        tail = existing.tail(lookback)

        # Extract base columns from tail
        base_cols = list(self._base_columns & set(tail.columns))
        combined = pl.concat([tail.select(base_cols), new_data.select(base_cols)])

        # Recompute features on combined
        recomputed = self.compute(combined)

        # Take only the new rows
        n_new = len(new_data)
        new_features = recomputed.tail(n_new)

        # Append to existing
        # Only keep columns that exist in both
        common_cols = list(set(existing.columns) & set(new_features.columns))
        result = pl.concat([existing.select(common_cols), new_features.select(common_cols)])

        return result

    @property
    def feature_columns(self) -> list[str]:
        """List of computed feature column names."""
        return self._feature_columns

    def get_feature_matrix(self, df: pl.DataFrame) -> np.ndarray:
        """Extract feature matrix as numpy array.

        Args:
            df: DataFrame with computed features.

        Returns:
            2D numpy array of features.
        """
        if not self._feature_columns:
            self._feature_columns = [
                c for c in df.columns if c not in self._base_columns
            ]

        available = [c for c in self._feature_columns if c in df.columns]
        return df.select(available).to_numpy().astype(np.float64)

    def _add_derived_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add derived/interaction features.

        These combine multiple indicators for richer signal information.

        Args:
            df: DataFrame with base features.

        Returns:
            DataFrame with derived features.
        """
        result = df.clone()

        # RSI regime (oversold/overbought)
        if "rsi_14" in result.columns:
            result = result.with_columns([
                (pl.col("rsi_14") < 30).cast(pl.Int8).alias("rsi_oversold"),
                (pl.col("rsi_14") > 70).cast(pl.Int8).alias("rsi_overbought"),
            ])

        # Bollinger squeeze (low volatility → breakout potential)
        if "bb_bandwidth" in result.columns:
            bw_mean = pl.col("bb_bandwidth").rolling_mean(window_size=50)
            result = result.with_columns(
                pl.when(bw_mean > 0)
                .then(pl.col("bb_bandwidth") / bw_mean)
                .otherwise(1.0)
                .alias("bb_squeeze")
            )

        # Volume-price divergence
        if "volume_ratio" in result.columns and "return_1" in result.columns:
            result = result.with_columns(
                (pl.col("volume_ratio") * pl.col("return_1").sign())
                .alias("volume_price_confirm")
            )

        # Trend consistency (multiple timeframe agreement)
        if all(f"close_rel_sma_{w}" in result.columns for w in [10, 20, 50]):
            result = result.with_columns(
                (
                    pl.col("close_rel_sma_10").sign()
                    + pl.col("close_rel_sma_20").sign()
                    + pl.col("close_rel_sma_50").sign()
                ).alias("trend_consistency")
            )

        # Mean reversion score (combo of zscore + hurst)
        if "zscore_50" in result.columns and "hurst_exponent" in result.columns:
            result = result.with_columns(
                (pl.col("zscore_50").abs() * (1 - pl.col("hurst_exponent")))
                .alias("mr_score")
            )

        # Momentum score (combo of ROC + ADX)
        if "roc_10" in result.columns and "adx_14" in result.columns:
            result = result.with_columns(
                (pl.col("roc_10") * pl.col("adx_14") / 100)
                .alias("momentum_score")
            )

        return result

    def _normalize_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply rolling z-score normalization to features.

        Args:
            df: DataFrame with raw features.

        Returns:
            DataFrame with normalized features.
        """
        result = df.clone()

        for col in self._feature_columns:
            if col not in result.columns:
                continue
            # Skip binary/categorical features
            if col in {"rsi_oversold", "rsi_overbought", "trend_consistency"}:
                continue

            mean = pl.col(col).rolling_mean(window_size=self.norm_window)
            std = pl.col(col).rolling_std(window_size=self.norm_window)
            result = result.with_columns(
                pl.when(std > 1e-10)
                .then((pl.col(col) - mean) / std)
                .otherwise(0.0)
                .alias(col)
            )

        return result
