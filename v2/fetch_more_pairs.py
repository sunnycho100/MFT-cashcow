#!/usr/bin/env python3
"""Fetch additional pairs for diversification."""
import sys
sys.path.insert(0, ".")
from src.utils.config import load_config
from src.data.fetcher import DataFetcher
from src.data.store import DataStore

config = load_config("config.yaml")
fetcher = DataFetcher(config)
store = DataStore(config)

new_pairs = ["AVAX/USDT", "DOGE/USDT", "LINK/USDT", "ADA/USDT", "DOT/USDT", "XRP/USDT"]

for pair in new_pairs:
    try:
        print(f"Fetching {pair}...", flush=True)
        df = fetcher.fetch(pair, "1h", days=1095)
        if len(df) > 0:
            store.save_ohlcv(pair, "1h", df)
            print(f"  OK {pair}: {len(df)} rows saved", flush=True)
        else:
            print(f"  FAIL {pair}: no data", flush=True)
    except Exception as e:
        print(f"  FAIL {pair}: {e}", flush=True)

store.close()
print("Done!")
