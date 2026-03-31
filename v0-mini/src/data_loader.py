"""Load OHLCV from v3 Coinbase CSVs or v2 DuckDB. EMA is always computed downstream."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def _pair_to_csv_name(pair: str) -> str:
    """BTC/USDT -> ohlcv_BTC_USD_1h.csv (matches v3 venue_candles naming)."""
    sym = pair.split("/")[0].upper()
    return f"ohlcv_{sym}_USD_1h.csv"


def load_ohlcv_csv(repo_root: Path, pair: str) -> pl.DataFrame:
    path = repo_root / "v3" / "data" / "coinbase" / _pair_to_csv_name(pair)
    if not path.exists():
        raise FileNotFoundError(f"Missing OHLCV CSV: {path}")
    df = pl.read_csv(
        path,
        columns=["timestamp_ms", "open", "high", "low", "close", "volume"],
        schema_overrides={
            "timestamp_ms": pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    )
    if df.is_empty():
        return df
    return (
        df.sort("timestamp_ms")
        .unique(subset=["timestamp_ms"], keep="last")
        .with_columns(
            pl.from_epoch(pl.col("timestamp_ms"), time_unit="ms")
            .dt.replace_time_zone("UTC")
            .alias("timestamp")
        )
        .select(["timestamp", "open", "high", "low", "close", "volume", "timestamp_ms"])
    )


def load_ohlcv_duckdb(repo_root: Path, pair: str, timeframe: str = "1h") -> pl.DataFrame | None:
    """Return OHLCV from v2 DuckDB if the file exists and has rows; else None."""
    db_path = repo_root / "v2" / "data" / "v2.duckdb"
    if not db_path.is_file():
        return None
    try:
        import duckdb
    except ImportError:
        return None
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        q = """
        SELECT timestamp, open, high, low, close, volume
        FROM ohlcv
        WHERE pair = ? AND timeframe = ?
        ORDER BY timestamp ASC
        """
        df = con.execute(q, [pair, timeframe]).pl()
    finally:
        con.close()
    if df.is_empty():
        return None
    return df.with_columns(
        (pl.col("timestamp").dt.timestamp("ms")).cast(pl.Int64).alias("timestamp_ms")
    )


def load_ohlcv(repo_root: Path, pair: str, timeframe: str = "1h", prefer_duckdb: bool = True) -> pl.DataFrame:
    """Prefer v2 DuckDB when available; otherwise v3 CSV."""
    if prefer_duckdb:
        from_db = load_ohlcv_duckdb(repo_root, pair, timeframe)
        if from_db is not None and not from_db.is_empty():
            return from_db
    return load_ohlcv_csv(repo_root, pair)
