"""Real-time data stream interface.

Placeholder for exchange websocket connections. Currently provides
a simulated stream from historical data for paper trading.
"""

import time
from datetime import datetime, timedelta
from typing import Callable, Optional

import polars as pl
import numpy as np

from ..utils.logger import get_logger

logger = get_logger("crypto_trader.data.stream")


class DataStream:
    """Real-time data stream interface.

    In production, this connects to exchange websockets for live candles.
    Currently provides a simulated stream using historical data for
    paper trading and backtesting.

    Args:
        config: Configuration dictionary.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._callbacks: list[Callable] = []
        self._running = False
        self._simulated_data: Optional[pl.DataFrame] = None
        self._current_index = 0

    def subscribe(self, callback: Callable[[dict], None]) -> None:
        """Subscribe to data updates.

        Args:
            callback: Function called with each new candle.
                     Receives dict with keys: pair, timestamp, open, high, low, close, volume.
        """
        self._callbacks.append(callback)
        logger.info(f"Subscriber added (total: {len(self._callbacks)})")

    def unsubscribe(self, callback: Callable) -> None:
        """Unsubscribe from data updates."""
        self._callbacks = [cb for cb in self._callbacks if cb != callback]

    def set_simulated_data(self, data: pl.DataFrame, pair: str = "BTC/USDT") -> None:
        """Set historical data for simulated streaming.

        Args:
            data: OHLCV DataFrame.
            pair: Trading pair name.
        """
        self._simulated_data = data
        self._pair = pair
        self._current_index = 0
        logger.info(f"Simulated data set: {len(data)} candles for {pair}")

    def start(self, interval_seconds: float = 1.0) -> None:
        """Start streaming data.

        For simulated mode, replays historical data at the given interval.

        Args:
            interval_seconds: Delay between candles in simulated mode.
        """
        if self._simulated_data is None:
            logger.error("No data source configured. Use set_simulated_data() first.")
            return

        self._running = True
        logger.info("Data stream started (simulated mode)")

        while self._running and self._current_index < len(self._simulated_data):
            row = self._simulated_data.row(self._current_index, named=True)

            candle = {
                "pair": self._pair,
                "timestamp": row["timestamp"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }

            for callback in self._callbacks:
                try:
                    callback(candle)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            self._current_index += 1
            time.sleep(interval_seconds)

        self._running = False
        logger.info("Data stream ended")

    def stop(self) -> None:
        """Stop the data stream."""
        self._running = False
        logger.info("Data stream stop requested")

    def get_next_candle(self) -> Optional[dict]:
        """Get the next candle without blocking (for event-driven usage).

        Returns:
            Candle dict or None if no more data.
        """
        if self._simulated_data is None or self._current_index >= len(self._simulated_data):
            return None

        row = self._simulated_data.row(self._current_index, named=True)
        self._current_index += 1

        return {
            "pair": getattr(self, "_pair", "UNKNOWN"),
            "timestamp": row["timestamp"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }

    @property
    def is_running(self) -> bool:
        """Whether the stream is currently active."""
        return self._running

    @property
    def progress(self) -> float:
        """Stream progress as a fraction (0 to 1)."""
        if self._simulated_data is None or len(self._simulated_data) == 0:
            return 0.0
        return self._current_index / len(self._simulated_data)
