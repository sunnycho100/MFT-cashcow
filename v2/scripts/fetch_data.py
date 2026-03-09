#!/usr/bin/env python3
"""Fetch OHLCV data and store in DuckDB.

Usage:
    # Fetch all configured pairs (3 years of 1h data by default)
    python -m scripts.fetch_data

    # Fetch specific pair / days
    python -m scripts.fetch_data --pairs BTC/USDT --days 365

    # Show what's stored
    python -m scripts.fetch_data --summary
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from src.utils.config import load_config
from src.data.fetcher import DataFetcher
from src.data.store import DataStore


def main():
    parser = argparse.ArgumentParser(description="Fetch crypto OHLCV data")
    parser.add_argument("--pairs", nargs="+", help="Trading pairs (e.g. BTC/USDT ETH/USDT)")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe (default: 1h)")
    parser.add_argument("--days", type=int, help="Days of history to fetch")
    parser.add_argument("--summary", action="store_true", help="Show stored data summary")
    parser.add_argument("--config", default=str(_root / "config.yaml"), help="Config path")
    args = parser.parse_args()

    config = load_config(args.config)
    store = DataStore(config)

    if args.summary:
        print("\n=== Stored Data Summary ===")
        summary = store.summary()
        if summary.is_empty():
            print("  (empty — no data fetched yet)")
        else:
            print(summary)
        store.close()
        return

    pairs = args.pairs or config.get("trading", {}).get("pairs", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    timeframe = args.timeframe
    days = args.days or config.get("data", {}).get("fetch_days", 1095)

    print(f"\n=== Fetching {len(pairs)} pairs × {days} days × {timeframe} ===\n")

    fetcher = DataFetcher(config)
    data = fetcher.fetch_multiple(pairs, timeframe=timeframe, days=days)

    total_rows = 0
    for pair, df in data.items():
        n = store.save_ohlcv(pair, timeframe, df)
        total_rows += n

    print(f"\n=== Done: {total_rows:,} total rows saved ===\n")

    # Show summary
    print(store.summary())
    store.close()


if __name__ == "__main__":
    main()
