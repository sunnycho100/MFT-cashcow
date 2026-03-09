"""Paper trading loop — runs the trained model on live data."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import polars as pl

from ..data.fetcher import DataFetcher
from ..data.store import DataStore
from ..features.pipeline import build_features, get_feature_names
from ..models.lgbm_model import LGBMModel, LABEL_MAP
from ..utils.logger import get_logger

logger = get_logger("v2.strategy.paper_trade")


class PaperTrader:
    """Simulated paper trading using the trained LightGBM model.

    Fetches latest candles, generates features, predicts direction,
    and logs signals. No real orders placed.
    """

    def __init__(self, config: dict):
        self.config = config
        self.pairs = config.get("trading", {}).get("pairs", ["BTC/USDT"])
        self.timeframe = config.get("trading", {}).get("primary_timeframe", "1h")
        self.fetcher = DataFetcher(config)
        self.store = DataStore(config)
        self.model = LGBMModel(config)

        # Position tracking
        self.positions: dict[str, dict] = {}  # pair → {side, entry_price, entry_time}
        self.trade_log: list[dict] = []
        self.running = False

    def start(self, callback=None):
        """Start paper trading loop. Calls callback(signal_dict) on each tick."""
        if not self.model.load():
            raise RuntimeError("No trained model found. Train first.")

        self.running = True
        logger.info("Paper trading started")

        while self.running:
            for pair in self.pairs:
                try:
                    signal = self._tick(pair)
                    if callback:
                        callback(signal)
                except Exception as e:
                    logger.error(f"Error on {pair}: {e}")

            # Wait for next candle (check every 60s)
            time.sleep(60)

    def stop(self):
        self.running = False
        logger.info("Paper trading stopped")

    def _tick(self, pair: str) -> dict:
        """Process one tick for a pair: fetch → features → predict → signal."""
        # Fetch last 250 candles (enough for feature warmup)
        df = self.fetcher.fetch(pair, self.timeframe, days=15)
        if len(df) < 200:
            return {"pair": pair, "error": "insufficient data"}

        # Build features
        feat_df = build_features(df)
        if len(feat_df) == 0:
            return {"pair": pair, "error": "feature build failed"}

        # Predict on latest row
        latest = feat_df.tail(1)
        pred_df = self.model.predict(latest)

        pred_class = int(pred_df["pred_class"][0])
        pred_label = LABEL_MAP[pred_class]
        prob_up = float(pred_df["pred_prob_up"][0])
        prob_down = float(pred_df["pred_prob_down"][0])
        prob_flat = float(pred_df["pred_prob_flat"][0])
        price = float(latest["close"][0])
        ts = datetime.now(timezone.utc).isoformat()

        signal = {
            "pair": pair,
            "timestamp": ts,
            "price": price,
            "prediction": pred_label,
            "prob_up": round(prob_up, 3),
            "prob_down": round(prob_down, 3),
            "prob_flat": round(prob_flat, 3),
            "confidence": round(max(prob_up, prob_down, prob_flat), 3),
        }

        # Simple position logic
        current_pos = self.positions.get(pair)
        if pred_label == "UP" and prob_up > 0.45 and current_pos is None:
            self.positions[pair] = {"side": "LONG", "entry_price": price, "entry_time": ts}
            signal["action"] = "OPEN LONG"
        elif pred_label == "DOWN" and prob_down > 0.45 and current_pos is None:
            self.positions[pair] = {"side": "SHORT", "entry_price": price, "entry_time": ts}
            signal["action"] = "OPEN SHORT"
        elif current_pos and pred_label != ("UP" if current_pos["side"] == "LONG" else "DOWN"):
            pnl = (price / current_pos["entry_price"] - 1) * (1 if current_pos["side"] == "LONG" else -1)
            signal["action"] = f"CLOSE {current_pos['side']} ({pnl:+.2%})"
            self.trade_log.append({**current_pos, "exit_price": price, "exit_time": ts, "pnl_pct": round(pnl * 100, 3)})
            del self.positions[pair]
        else:
            signal["action"] = "HOLD" if current_pos else "NO POSITION"

        logger.info(f"[{pair}] {signal['prediction']} (conf={signal['confidence']:.1%}) → {signal['action']} @ ${price:,.0f}")
        return signal

    def get_status(self) -> dict:
        """Return current state for TUI display."""
        return {
            "running": self.running,
            "positions": dict(self.positions),
            "trade_count": len(self.trade_log),
            "trade_log": self.trade_log[-10:],  # last 10
        }
