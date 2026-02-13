"""Data fetcher for crypto market data.

Currently uses yfinance as a placeholder. Designed to be swapped
for ccxt or direct exchange APIs later.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from ..utils.logger import get_logger

logger = get_logger("crypto_trader.data.fetcher")

# Mapping from trading pair notation to yfinance tickers
_YFINANCE_TICKER_MAP = {
    "BTC/USDT": "BTC-USD",
    "ETH/USDT": "ETH-USD",
    "SOL/USDT": "SOL-USD",
    "BNB/USDT": "BNB-USD",
    "XRP/USDT": "XRP-USD",
    "ADA/USDT": "ADA-USD",
    "DOGE/USDT": "DOGE-USD",
    "AVAX/USDT": "AVAX-USD",
    "DOT/USDT": "DOT-USD",
    "MATIC/USDT": "MATIC-USD",
}

# yfinance interval mapping
_INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "1h",  # yfinance doesn't support 4h natively; we'll resample
    "1d": "1d",
}


class DataFetcher:
    """Fetch crypto OHLCV data from various sources.

    Currently supports yfinance for free historical data.
    Designed to be extended with ccxt for exchange-direct data.

    Args:
        config: Configuration dictionary with data settings.
    """

    def __init__(self, config: dict = None, use_exchange: bool = False):
        self.config = config or {}
        data_cfg = self.config.get("data", {})
        self.cache_days = data_cfg.get("cache_days", 365)
        self.storage_path = Path(data_cfg.get("storage_path", "./data"))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.use_exchange = bool(use_exchange or data_cfg.get("use_exchange", False))
        self.exchange_connector = None

    def fetch(
        self,
        pair: str,
        timeframe: str = "1h",
        start: Optional[str] = None,
        end: Optional[str] = None,
        days: int = 90,
    ) -> pl.DataFrame:
        """Fetch OHLCV data for a trading pair.

        Args:
            pair: Trading pair (e.g., 'BTC/USDT').
            timeframe: Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1d').
            start: Start date string (YYYY-MM-DD). Overrides `days`.
            end: End date string (YYYY-MM-DD). Defaults to now.
            days: Number of days to fetch if start is not specified.

        Returns:
            Polars DataFrame with columns: timestamp, open, high, low, close, volume.
        """
        logger.info(
            "Fetching %s %s data (days=%s, source=%s)",
            pair,
            timeframe,
            days,
            "exchange" if self.use_exchange else "yfinance",
        )

        df = pl.DataFrame()
        if self.use_exchange:
            try:
                df = self._fetch_exchange(pair, timeframe, start, end, days)
            except Exception as e:
                logger.warning(f"Exchange fetch failed: {e}. Falling back to yfinance.")

        if len(df) == 0:
            try:
                df = self._fetch_yfinance(pair, timeframe, start, end, days)
            except Exception as e:
                logger.warning(f"yfinance fetch failed: {e}. Generating synthetic data.")
                df = self._generate_synthetic_data(pair, timeframe, days)

        if df is not None and len(df) > 0:
            logger.info(f"Fetched {len(df)} rows for {pair}")
        else:
            logger.warning(f"No data fetched for {pair}, generating synthetic")
            df = self._generate_synthetic_data(pair, timeframe, days)

        return df

    def fetch_multiple(
        self,
        pairs: list[str],
        timeframe: str = "1h",
        days: int = 90,
    ) -> dict[str, pl.DataFrame]:
        """Fetch data for multiple pairs.

        Args:
            pairs: List of trading pairs.
            timeframe: Candle timeframe.
            days: Number of days.

        Returns:
            Dict mapping pair symbols to DataFrames.
        """
        result = {}
        for pair in pairs:
            try:
                result[pair] = self.fetch(pair, timeframe, days=days)
            except Exception as e:
                logger.error(f"Failed to fetch {pair}: {e}")
        return result

    def _fetch_yfinance(
        self,
        pair: str,
        timeframe: str,
        start: Optional[str],
        end: Optional[str],
        days: int,
    ) -> pl.DataFrame:
        """Fetch data using yfinance.

        Args:
            pair: Trading pair.
            timeframe: Timeframe.
            start: Start date.
            end: End date.
            days: Days to look back.

        Returns:
            Polars OHLCV DataFrame.
        """
        import yfinance as yf

        ticker = _YFINANCE_TICKER_MAP.get(pair, pair.replace("/", "-"))
        interval = _INTERVAL_MAP.get(timeframe, "1h")
        needs_resample = timeframe == "4h"

        if start is None:
            # yfinance has limits on intraday data
            if interval in ["1m", "5m"]:
                days = min(days, 7)
            elif interval in ["15m", "30m"]:
                days = min(days, 60)
            elif interval == "1h":
                days = min(days, 730)

            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=days)
            start = start_dt.strftime("%Y-%m-%d")
            end = end_dt.strftime("%Y-%m-%d")

        yf_ticker = yf.Ticker(ticker)
        hist = yf_ticker.history(start=start, end=end, interval=interval)

        if hist.empty:
            return pl.DataFrame()

        # Convert to Polars
        hist = hist.reset_index()
        rename_map = {
            "Date": "timestamp",
            "Datetime": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }

        # Handle different index names from yfinance
        for old_name, new_name in rename_map.items():
            if old_name in hist.columns:
                hist = hist.rename(columns={old_name: new_name})

        cols_to_keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume"]
                       if c in hist.columns]
        hist = hist[cols_to_keep]

        df = pl.from_pandas(hist)

        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            if df["timestamp"].dtype != pl.Datetime:
                df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))

        # Resample to 4h if needed
        if needs_resample and len(df) > 0:
            df = self._resample_to_4h(df)

        # Cast numeric columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64))

        return df

    def _fetch_exchange(
        self,
        pair: str,
        timeframe: str,
        start: Optional[str],
        end: Optional[str],
        days: int,
    ) -> pl.DataFrame:
        """Fetch OHLCV data from configured exchange via CCXT."""
        if self.exchange_connector is None:
            from ..execution.exchange import ExchangeConnector

            self.exchange_connector = ExchangeConnector(self.config)

        limit = self._days_to_limit(timeframe=timeframe, days=days)

        # Current implementation relies on exchange-side latest bars.
        # Start/end are reserved for future pagination support.
        _ = start
        _ = end
        return self.exchange_connector.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)

    @staticmethod
    def _days_to_limit(timeframe: str, days: int) -> int:
        periods_map = {
            "1m": 24 * 60,
            "5m": 24 * 12,
            "15m": 24 * 4,
            "30m": 24 * 2,
            "1h": 24,
            "4h": 6,
            "1d": 1,
        }
        periods_per_day = periods_map.get(timeframe, 24)
        return max(20, min(days * periods_per_day, 5000))

    def _resample_to_4h(self, df: pl.DataFrame) -> pl.DataFrame:
        """Resample 1h data to 4h candles.

        Args:
            df: 1h OHLCV DataFrame.

        Returns:
            4h resampled DataFrame.
        """
        return (
            df.group_by_dynamic("timestamp", every="4h")
            .agg([
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
            ])
        )

    @staticmethod
    def _generate_synthetic_data(
        pair: str,
        timeframe: str = "1h",
        days: int = 90,
    ) -> pl.DataFrame:
        """Generate synthetic OHLCV data for testing.

        Creates realistic-looking price data using geometric Brownian motion
        with configurable parameters based on the pair.

        Args:
            pair: Trading pair (used to set base price).
            timeframe: Timeframe for candle generation.
            days: Number of days of data.

        Returns:
            Synthetic OHLCV DataFrame.
        """
        logger.info(f"Generating synthetic data for {pair}")

        # Base prices for common pairs
        base_prices = {
            "BTC/USDT": 45000.0,
            "ETH/USDT": 2500.0,
            "SOL/USDT": 100.0,
        }
        base_price = base_prices.get(pair, 100.0)

        # Periods per day
        periods_map = {"1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1}
        periods_per_day = periods_map.get(timeframe, 24)
        total_periods = days * periods_per_day

        # GBM parameters
        np.random.seed(hash(pair) % 2**32)
        mu = 0.0001  # Slight upward drift
        sigma = 0.02  # Daily vol ~2%
        dt = 1.0 / periods_per_day

        # Generate returns
        returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), total_periods)

        # Add some mean reversion and momentum
        for i in range(1, len(returns)):
            # Mean reversion component
            returns[i] -= 0.01 * returns[i - 1]
            # Volatility clustering
            returns[i] *= 1.0 + 0.3 * abs(returns[i - 1])

        # Cumulative prices
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLCV
        spread = 0.002  # 0.2% typical spread
        timestamps = [
            datetime(2024, 1, 1) + timedelta(hours=i * 24 / periods_per_day)
            for i in range(total_periods)
        ]

        opens = prices * (1 + np.random.uniform(-spread / 2, spread / 2, total_periods))
        highs = prices * (1 + np.random.uniform(0, spread * 2, total_periods))
        lows = prices * (1 - np.random.uniform(0, spread * 2, total_periods))
        volumes = np.random.lognormal(mean=15, sigma=1.5, size=total_periods)

        # Ensure OHLC consistency
        for i in range(total_periods):
            highs[i] = max(highs[i], opens[i], prices[i])
            lows[i] = min(lows[i], opens[i], prices[i])

        df = pl.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
        })

        return df
