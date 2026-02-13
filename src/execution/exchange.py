"""Exchange connector using CCXT.

Handles paper/live mode, API authentication, and order operations.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

import polars as pl
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency fallback
    def load_dotenv() -> bool:
        return False

from ..utils.logger import get_logger

logger = get_logger("crypto_trader.execution.exchange")


class ExchangeConnector:
    """CCXT-based exchange connector with paper/sandbox support."""

    _SUPPORTED_EXCHANGES = {"coinbase", "kraken", "gemini"}

    def __init__(self, config: dict):
        """Initialize exchange connection.

        Args:
            config: Exchange config with exchange id, credentials, and mode.
        """
        load_dotenv()

        self.config = config or {}
        exchange_cfg = self.config.get("exchange", self.config)

        self.exchange_id = exchange_cfg.get("id") or exchange_cfg.get("exchange_id", "coinbase")
        self.mode = str(exchange_cfg.get("mode", "paper")).lower()
        self.testnet = bool(exchange_cfg.get("testnet", self.mode == "paper"))
        self.rate_limit = bool(exchange_cfg.get("rate_limit", True))

        if self.mode != "paper":
            raise ValueError("Live trading is disabled in this version. Use exchange.mode='paper'.")

        api_key_env = exchange_cfg.get("api_key_env", "EXCHANGE_API_KEY")
        api_secret_env = exchange_cfg.get("api_secret_env", "EXCHANGE_API_SECRET")

        self.api_key = exchange_cfg.get("api_key") or os.getenv(api_key_env)
        self.api_secret = exchange_cfg.get("api_secret") or os.getenv(api_secret_env)

        if self.exchange_id not in self._SUPPORTED_EXCHANGES:
            raise ValueError(
                f"Unsupported exchange '{self.exchange_id}'. "
                f"Supported: {sorted(self._SUPPORTED_EXCHANGES)}"
            )

        try:
            import ccxt  # type: ignore
        except Exception as e:
            raise ImportError(
                "ccxt is required for ExchangeConnector. Install dependencies from requirements.txt."
            ) from e

        exchange_cls = getattr(ccxt, self.exchange_id)
        params: dict[str, Any] = {
            "enableRateLimit": self.rate_limit,
            "options": {
                "defaultType": "spot",
            },
        }

        if self.api_key:
            params["apiKey"] = self.api_key
        if self.api_secret:
            params["secret"] = self.api_secret

        self.client = exchange_cls(params)

        if self.testnet and self.exchange_id in {"coinbase", "kraken", "gemini"}:
            try:
                self.client.set_sandbox_mode(True)
                logger.info("Sandbox mode enabled for exchange connector")
            except Exception as e:
                logger.warning(f"Sandbox mode not available or failed to enable: {e}")

        self.client.load_markets()
        logger.info(
            "Exchange connector initialized: id=%s mode=%s testnet=%s",
            self.exchange_id,
            self.mode,
            self.testnet,
        )

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> pl.DataFrame:
        """Fetch OHLCV data from exchange."""
        normalized_symbol = self._normalize_symbol(symbol)
        rows = self.client.fetch_ohlcv(normalized_symbol, timeframe=timeframe, limit=limit)

        if not rows:
            return pl.DataFrame()

        df = pl.DataFrame(
            rows,
            schema=["timestamp", "open", "high", "low", "close", "volume"],
            orient="row",
        )

        df = df.with_columns(
            pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("timestamp"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
        )

        return df

    def get_balance(self) -> dict[str, Any]:
        """Get account balance."""
        self._require_credentials()
        return self.client.fetch_balance()

    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float = None,
    ) -> dict[str, Any]:
        """Place an order. Returns order object."""
        self._require_credentials()

        normalized_symbol = self._normalize_symbol(symbol)
        side = side.lower()
        order_type = order_type.lower()

        if self.mode == "paper":
            simulated = self._build_simulated_order(
                symbol=normalized_symbol,
                side=side,
                order_type=order_type,
                amount=amount,
                price=price,
            )
            logger.info(
                "Simulated %s order (%s %s %s @ %s)",
                order_type,
                side,
                amount,
                normalized_symbol,
                price,
            )
            return simulated

        if order_type == "market":
            return self.client.create_order(normalized_symbol, "market", side, amount)

        if order_type == "limit":
            if price is None:
                raise ValueError("Limit order requires price")
            return self.client.create_order(normalized_symbol, "limit", side, amount, price)

        raise ValueError(f"Unsupported order type: {order_type}")

    def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        """Cancel an open order."""
        self._require_credentials()

        if self.mode == "paper":
            return {
                "id": order_id,
                "symbol": self._normalize_symbol(symbol),
                "status": "canceled",
                "mode": "paper",
            }

        return self.client.cancel_order(order_id, self._normalize_symbol(symbol))

    def get_open_orders(self, symbol: str = None) -> list[dict[str, Any]]:
        """Get all open orders."""
        self._require_credentials()

        if self.mode == "paper":
            return []

        if symbol is None:
            return self.client.fetch_open_orders()
        return self.client.fetch_open_orders(self._normalize_symbol(symbol))

    def get_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        """Get order status."""
        self._require_credentials()

        if self.mode == "paper":
            return {
                "id": order_id,
                "symbol": self._normalize_symbol(symbol),
                "status": "closed",
                "mode": "paper",
            }

        return self.client.fetch_order(order_id, self._normalize_symbol(symbol))

    def _require_credentials(self) -> None:
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Exchange API credentials are required. Set EXCHANGE_API_KEY and "
                "EXCHANGE_API_SECRET in your environment or config."
            )

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize symbols for Coinbase spot markets.

        The system uses USDT pairs by default. Coinbase spot uses USD pairs.
        """
        if symbol.endswith("/USDT"):
            return symbol.replace("/USDT", "/USD")
        return symbol

    @staticmethod
    def _build_simulated_order(
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float],
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        order_id = f"paper-{int(now.timestamp() * 1000)}"
        return {
            "id": order_id,
            "clientOrderId": order_id,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "amount": amount,
            "price": price,
            "status": "closed" if order_type == "market" else "open",
            "filled": amount if order_type == "market" else 0.0,
            "remaining": 0.0 if order_type == "market" else amount,
            "timestamp": int(now.timestamp() * 1000),
            "datetime": now.isoformat(),
            "mode": "paper",
        }
