"""Order Router — routes and tracks orders through Kraken.

Rules:
- Limit orders for entries (maker fee 0.16% vs taker 0.26%)
- Market orders for exits (don't risk missing stop-losses)
- If limit not filled in 3 bars (config.kraken.limit_fill_timeout_bars) → convert to market
- Fill reconciliation: track expected vs actual fills
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from ..utils.logger import get_logger

logger = get_logger("v2.execution.order_router")


class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    order_id: str
    pair: str
    side: OrderSide
    order_type: str  # "limit" or "market"
    volume: float
    price: Optional[float]  # None for market orders
    leverage: Optional[int]
    status: OrderStatus = OrderStatus.PENDING
    filled_volume: float = 0.0
    avg_fill_price: Optional[float] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    bars_open: int = 0  # incremented each bar while limit order is open
    txid: Optional[str] = None  # Kraken txid after placement


class OrderRouter:
    """Routes and tracks orders through Kraken."""

    def __init__(self, config: dict, kraken_client, risk_manager):
        self.config = config
        self.kraken = kraken_client
        self.risk_manager = risk_manager
        kraken_cfg = config.get("kraken", {})
        self.limit_fill_timeout_bars: int = kraken_cfg.get("limit_fill_timeout_bars", 3)
        self.open_orders: dict[str, Order] = {}  # txid -> Order
        self.filled_orders: list[Order] = []
        self._order_counter: int = 0  # local counter for generating order_id before txid is known

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"ORD-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self._order_counter:04d}"

    def _parse_side(self, side: str) -> OrderSide:
        """Normalise a raw side string to OrderSide enum."""
        try:
            return OrderSide(side.lower())
        except ValueError as exc:
            raise ValueError(f"Invalid order side '{side}'. Expected 'buy' or 'sell'.") from exc

    def _extract_txid(self, response: dict) -> Optional[str]:
        """Extract Kraken txid from place_order response dict.

        KrakenClient._request() already unwraps the 'result' key,
        so response is the result dict directly: {"txid": [...], "descr": {...}}.
        """
        try:
            return response["txid"][0]
        except (KeyError, IndexError, TypeError):
            return None

    def _extract_fill_info(self, order_info: dict) -> tuple[float, Optional[float]]:
        """Return (filled_volume, avg_fill_price) from a Kraken order-info dict."""
        try:
            vol_exec = float(order_info.get("vol_exec", 0.0))
            price = order_info.get("price")
            avg_price = float(price) if price and float(price) > 0 else None
            return vol_exec, avg_price
        except (TypeError, ValueError):
            return 0.0, None

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_entry(
        self,
        pair: str,
        side: str,
        volume: float,
        entry_price: float,
        leverage: Optional[int] = None,
        validate: bool = False,
    ) -> Order:
        """Place a limit entry order.

        Validates with risk_manager.can_trade(), then submits a limit order
        via kraken_client. Returns an Order object and registers it in
        open_orders.
        """
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            raise RuntimeError(f"Trading halted — cannot place entry: {reason}")

        order_side = self._parse_side(side)
        order_id = self._next_order_id()

        order = Order(
            order_id=order_id,
            pair=pair,
            side=order_side,
            order_type="limit",
            volume=volume,
            price=entry_price,
            leverage=leverage,
        )

        logger.info(
            f"Placing limit entry: {side.upper()} {volume:.6f} {pair} @ {entry_price:.4f}"
            + (f" x{leverage}" if leverage else "")
            + (" [validate]" if validate else "")
        )

        try:
            response = self.kraken.place_order(
                pair=pair,
                side=side.lower(),
                order_type="limit",
                volume=volume,
                price=entry_price,
                leverage=leverage,
                validate=validate,
            )
            txid = self._extract_txid(response)

            if txid:
                order.txid = txid
                order.status = OrderStatus.OPEN
                self.open_orders[txid] = order
                logger.info(f"Limit entry placed. txid={txid}")
            elif validate:
                order.status = OrderStatus.PENDING
                logger.info(f"Limit entry validated (dry run). response={response}")
            else:
                order.status = OrderStatus.FAILED
                logger.error(f"Limit entry placement returned no txid. response={response}")

        except Exception as exc:
            order.status = OrderStatus.FAILED
            logger.error(f"Limit entry placement failed: {exc}")
            raise

        return order

    def place_exit(
        self,
        pair: str,
        side: str,
        volume: float,
        leverage: Optional[int] = None,
    ) -> Order:
        """Place a market exit order.

        Always a market order — don't risk missing stops.
        Skips the can_trade() check so exits always go through even when
        trading is halted (we still need to close existing positions).
        """
        order_side = self._parse_side(side)
        order_id = self._next_order_id()

        order = Order(
            order_id=order_id,
            pair=pair,
            side=order_side,
            order_type="market",
            volume=volume,
            price=None,
            leverage=leverage,
        )

        logger.info(
            f"Placing market exit: {side.upper()} {volume:.6f} {pair}"
            + (f" x{leverage}" if leverage else "")
        )

        try:
            response = self.kraken.place_order(
                pair=pair,
                side=side.lower(),
                order_type="market",
                volume=volume,
                leverage=leverage,
            )
            txid = self._extract_txid(response)

            if txid:
                order.txid = txid
                order.status = OrderStatus.OPEN
                self.open_orders[txid] = order
                logger.info(f"Market exit placed. txid={txid}")
            else:
                order.status = OrderStatus.FAILED
                logger.error(f"Market exit placement returned no txid. response={response}")

        except Exception as exc:
            order.status = OrderStatus.FAILED
            logger.error(f"Market exit placement failed: {exc}")
            raise

        return order

    # ------------------------------------------------------------------
    # Bar tick
    # ------------------------------------------------------------------

    def tick(self) -> list[Order]:
        """Called every bar. Checks open limit orders for fills or timeout.

        For each open limit order:
        - Increments bars_open.
        - Queries Kraken to check fill status.
        - If filled: moves to filled_orders, marks FILLED.
        - If bars_open >= limit_fill_timeout_bars and still open:
          cancels the limit order and replaces with a market order.

        Returns list of newly filled orders this tick.
        """
        if not self.open_orders:
            return []

        newly_filled: list[Order] = []
        to_remove: list[str] = []
        to_add: list[Order] = []  # market replacements

        # Fetch all currently open orders from Kraken once.
        # KrakenClient.get_open_orders() returns list[dict], but we need to
        # check by txid. Query specific orders instead for reliable lookup.
        kraken_open_txids: set[str] = set()
        try:
            open_orders_list = self.kraken.get_open_orders()
            # The list contains order dicts; we need to match by txid.
            # Kraken's OpenOrders response keys are txids in the raw "open" map,
            # but our client returns the values. For reliable matching, query our
            # specific txids directly.
            _ = open_orders_list  # consume to avoid unused warning
        except Exception as exc:
            logger.error(f"tick(): failed to fetch open orders from Kraken: {exc}")

        for txid, order in list(self.open_orders.items()):
            # Only process limit orders — market exits are fire-and-forget
            if order.order_type != "limit":
                continue

            order.bars_open += 1

            # Query this specific order's status from Kraken
            try:
                order_result = self.kraken.query_orders(txids=txid, trades=True)
                order_info = order_result.get(txid, {})
                kraken_status = order_info.get("status", "unknown")
            except Exception as exc:
                logger.error(f"tick(): failed to query order {txid}: {exc}")
                order_info = {}
                kraken_status = "unknown"

            vol_exec, avg_price = self._extract_fill_info(order_info)

            if kraken_status == "open" or kraken_status == "pending":
                # Order still open on Kraken
                if vol_exec > 0:
                    order.filled_volume = vol_exec
                    order.avg_fill_price = avg_price

                # Check timeout — convert to market if unfilled too long
                if order.bars_open >= self.limit_fill_timeout_bars:
                    logger.warning(
                        f"Limit order {txid} not filled after {order.bars_open} bars. "
                        f"Converting to market order."
                    )
                    cancelled = self.cancel_order(txid)
                    if cancelled:
                        order.status = OrderStatus.CANCELLED
                        to_remove.append(txid)
                        remaining_volume = order.volume - order.filled_volume
                        if remaining_volume > 0:
                            try:
                                market_order = self.place_exit(
                                    pair=order.pair,
                                    side=order.side.value,
                                    volume=remaining_volume,
                                    leverage=order.leverage,
                                )
                                to_add.append(market_order)
                            except Exception as exc:
                                logger.error(f"Failed to place market replacement for {txid}: {exc}")

            elif kraken_status == "closed" or vol_exec >= order.volume * 0.99:
                order.filled_volume = vol_exec if vol_exec > 0 else order.volume
                order.avg_fill_price = avg_price
                order.status = OrderStatus.FILLED
                to_remove.append(txid)
                self.filled_orders.append(order)
                newly_filled.append(order)
                logger.info(
                    f"Order {txid} FILLED: {order.filled_volume:.6f} {order.pair} "
                    f"@ avg {order.avg_fill_price}"
                )

            elif kraken_status == "canceled":
                order.status = OrderStatus.CANCELLED
                to_remove.append(txid)
                logger.info(f"Order {txid} was externally CANCELLED")

            else:
                logger.debug(f"Order {txid} status ambiguous (kraken_status={kraken_status}). Keeping open.")

        for txid in to_remove:
            self.open_orders.pop(txid, None)

        for market_order in to_add:
            if market_order.txid:
                self.open_orders[market_order.txid] = market_order

        return newly_filled

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel_order(self, txid: str) -> bool:
        """Cancel an open order. Returns True if successful."""
        try:
            self.kraken.cancel_order(txid=txid)
            order = self.open_orders.get(txid)
            if order:
                order.status = OrderStatus.CANCELLED
            logger.info(f"Cancelled order {txid}")
            return True
        except Exception as exc:
            logger.error(f"Failed to cancel order {txid}: {exc}")
            return False

    def cancel_all(self) -> int:
        """Cancel all open orders. Returns number of orders successfully cancelled."""
        cancelled_count = 0
        for txid in list(self.open_orders.keys()):
            if self.cancel_order(txid):
                cancelled_count += 1
        logger.info(f"cancel_all: cancelled {cancelled_count} orders")
        return cancelled_count

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def reconcile_fills(self) -> list[dict]:
        """Compare expected fills vs actual Kraken trade history.

        Fetches recent closed orders from Kraken and compares against the
        locally tracked filled_orders list.  Returns a list of discrepancy
        dicts, one per mismatch found.
        """
        discrepancies: list[dict] = []

        try:
            result = self.kraken.get_closed_orders()
            # KrakenClient._request() already unwraps "result", so result
            # is {"closed": {...}, "count": N} directly.
            kraken_closed: dict = result.get("closed", {})
        except Exception as exc:
            logger.error(f"reconcile_fills: failed to fetch closed orders: {exc}")
            return [{"error": str(exc)}]

        # Build lookup by txid for locally recorded fills
        local_fills: dict[str, Order] = {
            order.txid: order for order in self.filled_orders if order.txid
        }

        # Check every locally recorded fill against Kraken
        for txid, order in local_fills.items():
            if txid not in kraken_closed:
                discrepancies.append(
                    {
                        "type": "missing_on_kraken",
                        "txid": txid,
                        "pair": order.pair,
                        "expected_volume": order.volume,
                        "local_filled_volume": order.filled_volume,
                    }
                )
                continue

            kraken_info = kraken_closed[txid]
            kraken_vol_exec, kraken_avg_price = self._extract_fill_info(kraken_info)

            vol_diff = abs(order.filled_volume - kraken_vol_exec)
            if vol_diff > 1e-8:
                discrepancies.append(
                    {
                        "type": "volume_mismatch",
                        "txid": txid,
                        "pair": order.pair,
                        "local_filled_volume": order.filled_volume,
                        "kraken_vol_exec": kraken_vol_exec,
                        "diff": vol_diff,
                    }
                )

            if order.avg_fill_price and kraken_avg_price:
                price_diff = abs(order.avg_fill_price - kraken_avg_price)
                rel_diff = price_diff / kraken_avg_price if kraken_avg_price else 0
                if rel_diff > 0.001:  # >0.1% price discrepancy
                    discrepancies.append(
                        {
                            "type": "price_mismatch",
                            "txid": txid,
                            "pair": order.pair,
                            "local_avg_price": order.avg_fill_price,
                            "kraken_avg_price": kraken_avg_price,
                            "rel_diff_pct": round(rel_diff * 100, 4),
                        }
                    )

        # Check for fills that Kraken knows about but we don't
        for txid, kraken_info in kraken_closed.items():
            if txid not in local_fills:
                kraken_status = kraken_info.get("status", "")
                if kraken_status == "closed":
                    vol_exec, _ = self._extract_fill_info(kraken_info)
                    if vol_exec > 0:
                        discrepancies.append(
                            {
                                "type": "untracked_fill",
                                "txid": txid,
                                "kraken_vol_exec": vol_exec,
                                "description": "Kraken reports a closed/filled order not in local fill history",
                            }
                        )

        if discrepancies:
            logger.warning(f"reconcile_fills: found {len(discrepancies)} discrepancies")
        else:
            logger.info("reconcile_fills: no discrepancies found")

        return discrepancies

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current order state summary."""
        open_summary = [
            {
                "txid": o.txid,
                "order_id": o.order_id,
                "pair": o.pair,
                "side": o.side.value,
                "order_type": o.order_type,
                "volume": o.volume,
                "price": o.price,
                "status": o.status.value,
                "bars_open": o.bars_open,
                "filled_volume": o.filled_volume,
            }
            for o in self.open_orders.values()
        ]
        return {
            "open_order_count": len(self.open_orders),
            "filled_order_count": len(self.filled_orders),
            "open_orders": open_summary,
            "limit_fill_timeout_bars": self.limit_fill_timeout_bars,
        }
