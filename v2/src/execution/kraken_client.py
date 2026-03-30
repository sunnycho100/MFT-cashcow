"""Kraken REST API client with HMAC-SHA512 authentication.

Phase 1 Step 1.2 — Full production client for Kraken private and public endpoints.
"""

import base64
import hashlib
import hmac
import os
import time
from typing import Optional
from urllib.parse import urlencode

import requests

from ..utils.logger import get_logger

logger = get_logger("v2.execution.kraken")


class KrakenAPIError(Exception):
    """Raised when the Kraken API returns a non-empty error list."""


# Mapping from standard pair notation to Kraken API pair codes.
_PAIR_MAP: dict[str, str] = {
    "BTC/USD": "XBTUSD",
    "ETH/USD": "ETHUSD",
    "SOL/USD": "SOLUSD",
}


class KrakenClient:
    """Full Kraken REST API client.

    Handles HMAC-SHA512 authentication for private endpoints and plain GET
    requests for public endpoints.

    Args:
        api_key: Kraken API key. Defaults to the ``KRAKEN_API_KEY`` environment
            variable.
        api_secret: Kraken API secret. Defaults to the ``KRAKEN_API_SECRET``
            environment variable.
        base_url: Base URL for the Kraken API. Defaults to
            ``https://api.kraken.com``.
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = "https://api.kraken.com",
    ) -> None:
        self.api_key: str = api_key or os.environ.get("KRAKEN_API_KEY", "")
        self.api_secret: str = api_secret or os.environ.get("KRAKEN_API_SECRET", "")
        self.base_url: str = base_url.rstrip("/")
        self._session: requests.Session = requests.Session()
        self._session.headers.update({"User-Agent": "MFT-cashcow/2.0"})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _kraken_pair(self, pair: str) -> str:
        """Convert a standard pair notation to the Kraken API format.

        Known pairs (e.g. ``BTC/USD``) are mapped via a lookup table.
        Unknown pairs fall back to removing the slash character.

        Args:
            pair: Standard pair string such as ``"BTC/USD"``.

        Returns:
            Kraken-formatted pair string such as ``"XBTUSD"``.
        """
        return _PAIR_MAP.get(pair, pair.replace("/", ""))

    def _sign(self, uri_path: str, data: dict) -> tuple[str, str]:
        """Generate the HMAC-SHA512 signature required by private endpoints.

        Args:
            uri_path: The API path, e.g. ``"/0/private/Balance"``.
            data: POST parameters (must already contain the ``"nonce"`` key).

        Returns:
            A tuple of ``(nonce, api_sign)`` where ``api_sign`` is the
            base64-encoded HMAC-SHA512 digest.
        """
        nonce = str(int(time.time() * 1000))
        data["nonce"] = nonce

        post_data = urlencode(data)
        message = (nonce + post_data).encode()
        secret_decoded = base64.b64decode(self.api_secret)
        uri_bytes = uri_path.encode()

        sig = hmac.new(
            secret_decoded,
            uri_bytes + hashlib.sha256(message).digest(),
            hashlib.sha512,
        )
        api_sign = base64.b64encode(sig.digest()).decode()
        return nonce, api_sign

    def _request(
        self,
        endpoint: str,
        data: Optional[dict] = None,
        public: bool = False,
    ) -> dict:
        """Execute a request against the Kraken API.

        For private endpoints, the request is signed with HMAC-SHA512 and sent
        as a POST.  For public endpoints, the request is sent as a plain GET.

        Args:
            endpoint: API path, e.g. ``"/0/private/Balance"``.
            data: Optional POST body parameters for private endpoints, or query
                parameters for public endpoints.
            public: If ``True``, issue an unauthenticated GET request.

        Returns:
            The ``"result"`` field of the Kraken JSON response.

        Raises:
            KrakenAPIError: If the response contains a non-empty ``"error"``
                list.
        """
        url = self.base_url + endpoint
        data = dict(data or {})

        if public:
            try:
                response = self._session.get(url, params=data, timeout=10)
                response.raise_for_status()
            except requests.RequestException as exc:
                logger.error("GET {} failed: {}", endpoint, exc)
                raise KrakenAPIError(str(exc)) from exc
        else:
            nonce, api_sign = self._sign(endpoint, data)
            headers = {
                "API-Key": self.api_key,
                "API-Sign": api_sign,
                "Content-Type": "application/x-www-form-urlencoded",
            }
            try:
                response = self._session.post(
                    url, data=data, headers=headers, timeout=10
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                logger.error("POST {} failed: {}", endpoint, exc)
                raise KrakenAPIError(str(exc)) from exc

        payload: dict = response.json()
        errors: list = payload.get("error", [])
        if errors:
            logger.error("Kraken API error on {}: {}", endpoint, errors)
            raise KrakenAPIError(f"Kraken API error: {errors}")

        return payload.get("result", {})

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def get_balance(self) -> dict:
        """Retrieve account balances for all assets.

        Returns:
            Dictionary mapping asset code to string balance, e.g.
            ``{"BTC": "0.5", "USD": "1000.0"}``.

        Raises:
            KrakenAPIError: On API or network error.
        """
        return self._request("/0/private/Balance")

    def get_open_orders(self) -> list[dict]:
        """Retrieve all currently open orders.

        Returns:
            List of open order dictionaries as returned by the Kraken API.

        Raises:
            KrakenAPIError: On API or network error.
        """
        result = self._request("/0/private/OpenOrders")
        orders_map: dict = result.get("open", {}) if isinstance(result, dict) else {}
        return list(orders_map.values())

    def get_trade_history(self, start: Optional[float] = None) -> list[dict]:
        """Retrieve the account's trade history.

        Args:
            start: Optional Unix timestamp. If provided, only trades at or after
                this time are returned.

        Returns:
            List of trade history dictionaries as returned by the Kraken API.

        Raises:
            KrakenAPIError: On API or network error.
        """
        data: dict = {}
        if start is not None:
            data["start"] = start

        result = self._request("/0/private/TradesHistory", data=data)
        trades_map: dict = result.get("trades", {}) if isinstance(result, dict) else {}
        return list(trades_map.values())

    def place_order(
        self,
        pair: str,
        side: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        leverage: Optional[int] = None,
        validate: bool = False,
    ) -> dict:
        """Place a new order on Kraken.

        Args:
            pair: Trading pair in standard notation, e.g. ``"BTC/USD"``.
            side: ``"buy"`` or ``"sell"``.
            order_type: ``"limit"`` or ``"market"``.
            volume: Order volume in the base asset.
            price: Limit price. Required for limit orders; ignored for market
                orders.
            leverage: Leverage ratio for margin orders, e.g. ``2`` or ``3``.
                Pass ``None`` for spot orders.
            validate: If ``True``, submit as a dry-run (no real order placed).

        Returns:
            Dictionary with keys ``"txid"`` (list of order IDs) and
            ``"descr"`` (order description map).

        Raises:
            KrakenAPIError: On API or network error, or if Kraken rejects the
                order.
            ValueError: If ``order_type`` is ``"limit"`` but no ``price`` is
                provided.
        """
        if order_type == "limit" and price is None:
            raise ValueError("price is required for limit orders")

        kraken_pair = self._kraken_pair(pair)
        data: dict = {
            "pair": kraken_pair,
            "type": side,
            "ordertype": order_type,
            "volume": str(volume),
        }

        if price is not None:
            data["price"] = str(price)

        if leverage is not None:
            data["leverage"] = str(leverage)

        if validate:
            data["validate"] = "true"

        logger.info(
            "Placing order: pair={} side={} type={} volume={} price={} leverage={} validate={}",
            pair,
            side,
            order_type,
            volume,
            price,
            leverage,
            validate,
        )

        return self._request("/0/private/AddOrder", data=data)

    def cancel_order(self, txid: str) -> dict:
        """Cancel an open order by transaction ID.

        Args:
            txid: The transaction ID of the order to cancel.

        Returns:
            Dictionary with cancellation details (``"count"`` of cancelled
            orders, ``"pending"`` flag).

        Raises:
            KrakenAPIError: On API or network error.
        """
        return self._request("/0/private/CancelOrder", data={"txid": txid})

    def cancel_all_after(self, timeout: int) -> dict:
        """Activate or cancel the dead-man's switch for order cancellation.

        Schedules all open orders to be cancelled after ``timeout`` seconds.
        Call with ``timeout=0`` to disable the timer.

        Args:
            timeout: Seconds until all open orders are automatically cancelled.
                Pass ``0`` to cancel the timer without cancelling orders.

        Returns:
            Dictionary with ``"currentTime"`` (current server time) and
            ``"triggerTime"`` (time at which the cancellation will trigger, or
            ``0`` if the timer was disabled).

        Raises:
            KrakenAPIError: On API or network error.
        """
        return self._request(
            "/0/private/CancelAllOrdersAfter", data={"timeout": str(timeout)}
        )

    def validate_order(
        self,
        pair: str,
        side: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        leverage: Optional[int] = None,
    ) -> dict:
        """Validate an order without actually placing it (dry run).

        Delegates to :meth:`place_order` with ``validate=True``.

        Args:
            pair: Trading pair in standard notation, e.g. ``"BTC/USD"``.
            side: ``"buy"`` or ``"sell"``.
            order_type: ``"limit"`` or ``"market"``.
            volume: Order volume in the base asset.
            price: Limit price. Required for limit orders.
            leverage: Leverage ratio, e.g. ``2`` or ``3``. ``None`` for spot.

        Returns:
            Validation result dictionary from the Kraken API.

        Raises:
            KrakenAPIError: If Kraken rejects the order parameters.
            ValueError: If ``order_type`` is ``"limit"`` but no ``price`` is
                provided.
        """
        return self.place_order(
            pair=pair,
            side=side,
            order_type=order_type,
            volume=volume,
            price=price,
            leverage=leverage,
            validate=True,
        )

    def query_orders(self, txids: str, trades: bool = False) -> dict:
        """Query info on specific orders by transaction ID.

        Args:
            txids: Comma-delimited list of transaction IDs to query.
            trades: If ``True``, include trades related to the order.

        Returns:
            Dictionary keyed by txid with order info dicts as values.

        Raises:
            KrakenAPIError: On API or network error.
        """
        data: dict = {"txid": txids}
        if trades:
            data["trades"] = "true"
        return self._request("/0/private/QueryOrders", data=data)

    def get_closed_orders(self) -> dict:
        """Retrieve recently closed orders.

        Returns:
            Dictionary with ``"closed"`` key mapping txid to order info,
            and ``"count"`` key with total number of closed orders.

        Raises:
            KrakenAPIError: On API or network error.
        """
        return self._request("/0/private/ClosedOrders")

    def get_ticker(self, pair: str) -> dict:
        """Retrieve ticker information for a trading pair.

        Args:
            pair: Trading pair in standard notation, e.g. ``"BTC/USD"``.

        Returns:
            Ticker dictionary for the pair. Key fields:

            - ``"a"``: Ask price array ``[price, whole_lot_volume, lot_volume]``
            - ``"b"``: Bid price array
            - ``"c"``: Last trade closed array ``[price, lot_volume]``

        Raises:
            KrakenAPIError: On API or network error.
        """
        kraken_pair = self._kraken_pair(pair)
        result = self._request(
            "/0/public/Ticker", data={"pair": kraken_pair}, public=True
        )
        # Kraken returns a dict keyed by the canonical pair name.
        if isinstance(result, dict) and result:
            return next(iter(result.values()))
        return result
