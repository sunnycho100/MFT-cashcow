#!/usr/bin/env python3
"""Paper trading loop for the crypto quantitative trading system."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import yaml
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency fallback
    def load_dotenv() -> bool:
        return False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.fetcher import DataFetcher
from src.models.ensemble import EnsembleModel
from src.strategy.signals import SignalGenerator
from src.utils.logger import get_logger

logger = get_logger("crypto_trader.scripts.run_paper")


@dataclass
class PaperPosition:
    pair: str
    side: str
    amount: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class PaperTradingEngine:
    """Run a safe paper-trading loop using existing models and signals."""

    def __init__(self, config: dict, use_exchange: bool = True):
        self.config = config
        self.use_exchange = use_exchange

        execution_cfg = config.get("execution", {})
        exchange_cfg = config.get("exchange", {})

        self.execution_mode = execution_cfg.get("mode", "paper")
        self.exchange_mode = execution_cfg.get("exchange_mode", exchange_cfg.get("mode", "paper"))
        self.dry_run = bool(execution_cfg.get("dry_run", True))
        self.max_position_pct = float(config.get("risk", {}).get("max_position_pct", 0.10))

        if self.execution_mode != "paper" or self.exchange_mode != "paper":
            raise ValueError("Live trading is disabled. Set execution/exchange mode to 'paper'.")

        paper_cfg = execution_cfg.get("paper", {})
        self.initial_capital = float(paper_cfg.get("initial_capital", 100000.0))

        self.fetcher = DataFetcher(config=config, use_exchange=use_exchange)
        self.ensemble = EnsembleModel(config)
        self.signal_generator = SignalGenerator(config)

        self.exchange = None
        if use_exchange:
            from src.execution.exchange import ExchangeConnector

            self.exchange = ExchangeConnector(config)

        self.positions: dict[str, PaperPosition] = {}
        self.realized_pnl = 0.0

    def run(self, interval_seconds: int, max_cycles: int = 0) -> None:
        """Run the paper trading loop.

        Args:
            interval_seconds: Time to sleep between cycles.
            max_cycles: 0 means infinite loop.
        """
        trading_cfg = self.config.get("trading", {})
        pairs = trading_cfg.get("pairs", ["BTC/USDT"])
        timeframe = trading_cfg.get("primary_timeframe", "1h")

        cycle = 0
        while True:
            cycle += 1
            logger.info("Paper cycle %s started (%s)", cycle, datetime.utcnow().isoformat())

            try:
                pair_data = self.fetcher.fetch_multiple(pairs=pairs, timeframe=timeframe, days=90)
                primary_pair = pairs[0]
                primary_data = pair_data.get(primary_pair)

                if primary_data is None or len(primary_data) == 0:
                    logger.warning("No primary data available for %s, skipping cycle", primary_pair)
                    self._maybe_sleep(interval_seconds, max_cycles, cycle)
                    continue

                self.ensemble.fit(primary_data, pair_data=pair_data)
                signals = self.ensemble.generate_signals(primary_data, pair_data=pair_data)

                current_prices = {
                    pair: float(df["close"][-1])
                    for pair, df in pair_data.items()
                    if df is not None and len(df) > 0
                }
                portfolio_value = self._portfolio_value(current_prices)

                decisions = self.signal_generator.process_signals(
                    signals=signals,
                    current_prices=current_prices,
                    portfolio_value=portfolio_value,
                )

                for decision in decisions:
                    self._execute_decision(decision, portfolio_value, current_prices)

                logger.info(
                    "Cycle %s complete | signals=%s decisions=%s portfolio=%.2f realized_pnl=%.2f",
                    cycle,
                    len(signals),
                    len(decisions),
                    self._portfolio_value(current_prices),
                    self.realized_pnl,
                )

            except Exception as e:
                logger.error("Paper cycle failed: %s", e, exc_info=True)

            self._maybe_sleep(interval_seconds, max_cycles, cycle)

    def _execute_decision(self, decision, portfolio_value: float, prices: dict[str, float]) -> None:
        pair = decision.pair
        price = prices.get(pair)
        if price is None or price <= 0:
            logger.warning("No price for %s, skipping decision %s", pair, decision.action)
            return

        action = decision.action
        if action in {"buy", "sell"}:
            self._open_position(decision, portfolio_value, price)
        elif action in {"close_long", "close_short"}:
            self._close_position(pair, price)

    def _open_position(self, decision, portfolio_value: float, price: float) -> None:
        pair = decision.pair
        side = "long" if decision.action == "buy" else "short"

        if pair in self.positions:
            logger.info("Position already open for %s, skipping", pair)
            return

        position_pct = min(float(decision.suggested_size), self.max_position_pct)
        if position_pct <= 0:
            logger.info("Decision size is zero for %s", pair)
            return

        notional = portfolio_value * position_pct
        amount = notional / price

        order_side = "buy" if side == "long" else "sell"
        if self.exchange is not None and not self.dry_run:
            order = self.exchange.create_order(
                symbol=pair,
                side=order_side,
                order_type="market",
                amount=amount,
            )
            logger.info("Exchange order placed: %s", order.get("id"))
        else:
            logger.info(
                "Dry-run order: %s %s %.6f @ %.2f (%.2f%%)",
                order_side,
                pair,
                amount,
                price,
                position_pct * 100,
            )

        position = PaperPosition(
            pair=pair,
            side=side,
            amount=amount,
            entry_price=price,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
        )
        self.positions[pair] = position
        self.signal_generator.update_position(
            pair,
            {
                "side": side,
                "stop_loss": decision.stop_loss,
                "take_profit": decision.take_profit,
            },
        )

    def _close_position(self, pair: str, price: float) -> None:
        position = self.positions.get(pair)
        if position is None:
            return

        if self.exchange is not None and not self.dry_run:
            order_side = "sell" if position.side == "long" else "buy"
            self.exchange.create_order(
                symbol=pair,
                side=order_side,
                order_type="market",
                amount=position.amount,
            )

        side_mult = 1.0 if position.side == "long" else -1.0
        pnl = side_mult * (price - position.entry_price) * position.amount
        self.realized_pnl += pnl

        self.positions.pop(pair, None)
        self.signal_generator.update_position(pair, None)

        logger.info(
            "Closed %s %s | entry=%.2f exit=%.2f amount=%.6f pnl=%.2f",
            position.side,
            pair,
            position.entry_price,
            price,
            position.amount,
            pnl,
        )

    def _portfolio_value(self, prices: dict[str, float]) -> float:
        unrealized = 0.0
        for position in self.positions.values():
            price = prices.get(position.pair)
            if price is None:
                continue
            side_mult = 1.0 if position.side == "long" else -1.0
            unrealized += side_mult * (price - position.entry_price) * position.amount

        return self.initial_capital + self.realized_pnl + unrealized

    @staticmethod
    def _maybe_sleep(interval_seconds: int, max_cycles: int, cycle: int) -> None:
        if max_cycles > 0 and cycle >= max_cycles:
            raise SystemExit(0)
        time.sleep(interval_seconds)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper trading loop")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--interval-seconds", type=int, default=60, help="Loop interval in seconds")
    parser.add_argument("--max-cycles", type=int, default=0, help="0 = run forever")
    parser.add_argument(
        "--no-exchange",
        action="store_true",
        help="Disable exchange connectivity and use fallback data sources",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_config(args.config)

    engine = PaperTradingEngine(config=config, use_exchange=not args.no_exchange)
    engine.run(interval_seconds=args.interval_seconds, max_cycles=args.max_cycles)


if __name__ == "__main__":
    main()
