"""Local data storage using DuckDB.

Provides persistent storage for OHLCV data, model states,
and trading history using DuckDB for fast analytical queries.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import polars as pl

from ..utils.logger import get_logger

logger = get_logger("crypto_trader.data.store")


class DataStore:
    """DuckDB-backed data store for crypto trading data.

    Stores OHLCV candles, trade history, and model metadata
    in a local DuckDB database for fast querying.

    Args:
        config: Configuration dictionary.
        db_path: Override path to DuckDB file.
    """

    def __init__(self, config: dict = None, db_path: Optional[str] = None):
        self.config = config or {}
        data_cfg = self.config.get("data", {})

        if db_path:
            self.db_path = db_path
        else:
            self.db_path = data_cfg.get("db_path", "./data/crypto_trader.duckdb")

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(self.db_path)
        self._init_tables()
        logger.info(f"DataStore initialized at {self.db_path}")

    def _init_tables(self) -> None:
        """Create database tables if they don't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                pair VARCHAR,
                timeframe VARCHAR,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (pair, timeframe, timestamp)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                pair VARCHAR,
                side VARCHAR,
                entry_price DOUBLE,
                exit_price DOUBLE,
                quantity DOUBLE,
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                pnl DOUBLE,
                pnl_pct DOUBLE,
                model VARCHAR,
                metadata VARCHAR
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_state (
                model_name VARCHAR PRIMARY KEY,
                last_trained TIMESTAMP,
                metrics VARCHAR,
                config VARCHAR,
                version INTEGER DEFAULT 1
            )
        """)

    def save_ohlcv(self, pair: str, timeframe: str, data: pl.DataFrame) -> int:
        """Save OHLCV data to the store.

        Uses INSERT OR REPLACE to handle duplicates.

        Args:
            pair: Trading pair.
            timeframe: Candle timeframe.
            data: OHLCV DataFrame.

        Returns:
            Number of rows inserted.
        """
        if data.is_empty():
            return 0

        # Add pair and timeframe columns
        save_df = data.select([
            pl.lit(pair).alias("pair"),
            pl.lit(timeframe).alias("timeframe"),
            pl.col("timestamp"),
            pl.col("open"),
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
            pl.col("volume"),
        ])

        # Use DuckDB's INSERT OR REPLACE
        self.conn.execute("DELETE FROM ohlcv WHERE pair = ? AND timeframe = ?", [pair, timeframe])
        self.conn.execute("INSERT INTO ohlcv SELECT * FROM save_df")

        count = len(save_df)
        logger.info(f"Saved {count} OHLCV rows for {pair} {timeframe}")
        return count

    def load_ohlcv(
        self,
        pair: str,
        timeframe: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pl.DataFrame:
        """Load OHLCV data from the store.

        Args:
            pair: Trading pair.
            timeframe: Candle timeframe.
            start: Start timestamp filter.
            end: End timestamp filter.
            limit: Maximum rows to return.

        Returns:
            OHLCV DataFrame.
        """
        query = "SELECT timestamp, open, high, low, close, volume FROM ohlcv WHERE pair = ? AND timeframe = ?"
        params = [pair, timeframe]

        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)

        query += " ORDER BY timestamp ASC"

        if limit:
            query += f" LIMIT {limit}"

        result = self.conn.execute(query, params).pl()

        if result.is_empty():
            logger.debug(f"No stored data for {pair} {timeframe}")
        else:
            logger.debug(f"Loaded {len(result)} rows for {pair} {timeframe}")

        return result

    def save_trade(
        self,
        pair: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        entry_time: datetime,
        exit_time: datetime,
        pnl: float,
        pnl_pct: float,
        model: str = "",
        metadata: str = "",
    ) -> None:
        """Save a completed trade.

        Args:
            pair: Trading pair.
            side: 'long' or 'short'.
            entry_price: Entry price.
            exit_price: Exit price.
            quantity: Position size.
            entry_time: Entry timestamp.
            exit_time: Exit timestamp.
            pnl: Absolute P&L.
            pnl_pct: Percentage P&L.
            model: Model that generated the signal.
            metadata: Additional JSON metadata.
        """
        self.conn.execute("""
            INSERT INTO trades (pair, side, entry_price, exit_price, quantity,
                              entry_time, exit_time, pnl, pnl_pct, model, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [pair, side, entry_price, exit_price, quantity,
              entry_time, exit_time, pnl, pnl_pct, model, metadata])

    def get_trades(
        self,
        pair: Optional[str] = None,
        model: Optional[str] = None,
        limit: int = 100,
    ) -> pl.DataFrame:
        """Retrieve trade history.

        Args:
            pair: Filter by pair.
            model: Filter by model.
            limit: Maximum trades to return.

        Returns:
            Trades DataFrame.
        """
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if pair:
            query += " AND pair = ?"
            params.append(pair)
        if model:
            query += " AND model = ?"
            params.append(model)

        query += f" ORDER BY exit_time DESC LIMIT {limit}"
        return self.conn.execute(query, params).pl()

    def save_model_state(self, model_name: str, metrics: str, config: str) -> None:
        """Save model training state.

        Args:
            model_name: Model identifier.
            metrics: JSON string of performance metrics.
            config: JSON string of model config.
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO model_state (model_name, last_trained, metrics, config, version)
            VALUES (?, CURRENT_TIMESTAMP, ?, ?,
                    COALESCE((SELECT version FROM model_state WHERE model_name = ?), 0) + 1)
        """, [model_name, metrics, config, model_name])

    def get_model_state(self, model_name: str) -> Optional[dict]:
        """Get model training state.

        Args:
            model_name: Model identifier.

        Returns:
            Dict with model state or None.
        """
        result = self.conn.execute(
            "SELECT * FROM model_state WHERE model_name = ?", [model_name]
        ).fetchone()

        if result:
            return {
                "model_name": result[0],
                "last_trained": result[1],
                "metrics": result[2],
                "config": result[3],
                "version": result[4],
            }
        return None

    def get_pair_count(self) -> dict[str, int]:
        """Get row counts per pair in OHLCV table.

        Returns:
            Dict mapping pair names to row counts.
        """
        result = self.conn.execute(
            "SELECT pair, timeframe, COUNT(*) as cnt FROM ohlcv GROUP BY pair, timeframe"
        ).fetchall()
        return {f"{r[0]}_{r[1]}": r[2] for r in result}

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
        logger.info("DataStore closed")

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass
