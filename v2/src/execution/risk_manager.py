"""Position & Risk Manager — enforces all risk limits for the trading system.

Risk rules (all enforced simultaneously):
- Global halt: equity drops 30% below high-water mark → close all, disable trading, alert
- Daily loss limit: 8% → halt for 24 hours
- Per-trade risk: size = equity × risk_pct / (entry - stop), capped at max_position_pct
- Margin ceiling: never use >60% of available margin
- Dead-man heartbeat: cancel_all_after(90) every 60 seconds
- Loss-streak cooldown: 3 consecutive losses → halve position size for next 3 trades
"""

from __future__ import annotations

import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from ..utils.logger import get_logger

logger = get_logger("v2.execution.risk_manager")


class RiskManager:
    """Enforces all risk limits for the trading system."""

    def __init__(self, config: dict, kraken_client=None):
        risk_cfg = config.get("risk", {})
        margin_cfg = config.get("margin", {})
        kraken_cfg = config.get("kraken", {})

        # Risk parameters
        self.max_drawdown_pct: float = risk_cfg.get("max_drawdown_pct", 0.30)
        self.daily_loss_limit_pct: float = risk_cfg.get("daily_loss_limit_pct", 0.08)
        self.max_position_pct: float = risk_cfg.get("max_position_pct", 0.30)
        self.max_margin_usage_pct: float = margin_cfg.get("max_margin_usage_pct", 0.60)
        self.dead_man_timeout: int = kraken_cfg.get("dead_man_timeout_sec", 90)
        self.dead_man_interval: int = kraken_cfg.get("dead_man_heartbeat_sec", 60)

        # Loss-streak cooldown parameters
        self.loss_streak_threshold: int = 3
        self.loss_streak_size_factor: float = 0.5
        self.loss_streak_cooldown_trades: int = 3

        # State — initialised with placeholder equity; caller must call
        # update_equity() with starting equity before trading begins.
        self.high_water_mark: float = 0.0
        self.daily_start_equity: float = 0.0
        self.daily_loss_date: date = datetime.now(timezone.utc).date()
        self.trading_halted: bool = False
        self.halt_reason: str = ""
        self.halt_until: Optional[datetime] = None
        self.consecutive_losses: int = 0
        self.size_cooldown_trades_remaining: int = 0

        self.kraken_client = kraken_client
        self._dead_man_thread: Optional[threading.Thread] = None
        self._dead_man_stop_event: threading.Event = threading.Event()

    # ------------------------------------------------------------------
    # Equity tracking
    # ------------------------------------------------------------------

    def update_equity(self, current_equity: float) -> None:
        """Update equity tracking. Updates high water mark and checks drawdown."""
        now = datetime.now(timezone.utc)
        today = now.date()

        # Bootstrap on first call
        if self.high_water_mark == 0.0:
            self.high_water_mark = current_equity
            self.daily_start_equity = current_equity
            self.daily_loss_date = today
            logger.info(f"Risk manager bootstrapped. Starting equity: {current_equity:.2f}")
            return

        # Reset daily tracking at start of new calendar day
        if today > self.daily_loss_date:
            logger.info(
                f"New trading day. Resetting daily equity from {self.daily_start_equity:.2f} to {current_equity:.2f}"
            )
            self.daily_start_equity = current_equity
            self.daily_loss_date = today
            # Clear a 24-hour timed halt if it was issued on a previous day
            if self.halt_until is not None and now >= self.halt_until:
                self._clear_timed_halt()

        # Update high-water mark
        if current_equity > self.high_water_mark:
            self.high_water_mark = current_equity
            logger.info(f"New high-water mark: {self.high_water_mark:.2f}")

        # Check global drawdown halt
        drawdown_floor = self.high_water_mark * (1.0 - self.max_drawdown_pct)
        if current_equity < drawdown_floor:
            drawdown_pct = (self.high_water_mark - current_equity) / self.high_water_mark
            reason = (
                f"Global drawdown limit breached: equity {current_equity:.2f} is "
                f"{drawdown_pct:.1%} below high-water mark {self.high_water_mark:.2f}"
            )
            logger.critical(reason)
            self.halt_trading(reason)
            return

        # Check daily loss limit
        if self.daily_start_equity > 0:
            daily_loss_pct = (self.daily_start_equity - current_equity) / self.daily_start_equity
            if daily_loss_pct > self.daily_loss_limit_pct:
                reason = (
                    f"Daily loss limit breached: lost {daily_loss_pct:.1%} "
                    f"(limit {self.daily_loss_limit_pct:.1%}) today"
                )
                logger.error(reason)
                self.halt_trading(reason, duration_hours=24.0)

    # ------------------------------------------------------------------
    # Trading gate
    # ------------------------------------------------------------------

    def can_trade(self) -> tuple[bool, str]:
        """Returns (True, '') if trading is allowed, (False, reason) if halted."""
        if not self.trading_halted:
            return True, ""

        # Check whether a timed halt has expired
        if self.halt_until is not None:
            if datetime.now(timezone.utc) >= self.halt_until:
                self._clear_timed_halt()
                return True, ""
            remaining = (self.halt_until - datetime.now(timezone.utc)).total_seconds() / 3600
            return False, f"{self.halt_reason} (resumes in {remaining:.1f}h)"

        return False, self.halt_reason

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        risk_pct: float = 0.02,
    ) -> float:
        """Calculate position size using fixed fractional risk.

        size = equity * risk_pct / abs(entry_price - stop_price)
        Capped at equity * max_position_pct / entry_price.
        Applies loss-streak cooldown factor when active.
        Returns size in base asset units.
        """
        if equity <= 0 or entry_price <= 0:
            logger.warning("calculate_position_size called with non-positive equity or entry_price")
            return 0.0

        price_risk = abs(entry_price - stop_price)
        if price_risk == 0:
            logger.warning("entry_price == stop_price; cannot size position")
            return 0.0

        # Apply cooldown factor if a loss streak is active
        effective_risk_pct = risk_pct
        if self.size_cooldown_trades_remaining > 0:
            effective_risk_pct *= self.loss_streak_size_factor
            logger.info(
                f"Loss-streak cooldown active ({self.size_cooldown_trades_remaining} trades remaining). "
                f"risk_pct reduced to {effective_risk_pct:.4f}"
            )

        raw_size = equity * effective_risk_pct / price_risk
        max_size = equity * self.max_position_pct / entry_price
        size = min(raw_size, max_size)

        logger.debug(
            f"Position size: raw={raw_size:.6f}, cap={max_size:.6f}, final={size:.6f} "
            f"(equity={equity:.2f}, risk_pct={effective_risk_pct:.4f})"
        )
        return size

    # ------------------------------------------------------------------
    # Margin check
    # ------------------------------------------------------------------

    def check_margin_budget(self, available_margin: float, proposed_order_value: float) -> bool:
        """Returns True if order fits within margin ceiling (60% of available)."""
        ceiling = available_margin * self.max_margin_usage_pct
        fits = proposed_order_value <= ceiling
        if not fits:
            logger.warning(
                f"Margin budget exceeded: proposed {proposed_order_value:.2f} > "
                f"ceiling {ceiling:.2f} ({self.max_margin_usage_pct:.0%} of {available_margin:.2f})"
            )
        return fits

    # ------------------------------------------------------------------
    # Trade result recording
    # ------------------------------------------------------------------

    def record_trade_result(self, pnl: float) -> None:
        """Record trade P&L. Updates consecutive loss counter and cooldown."""
        if pnl < 0:
            self.consecutive_losses += 1
            logger.info(f"Loss recorded. Consecutive losses: {self.consecutive_losses}")
            if self.consecutive_losses >= self.loss_streak_threshold:
                if self.size_cooldown_trades_remaining == 0:
                    # Activate a fresh cooldown window
                    self.size_cooldown_trades_remaining = self.loss_streak_cooldown_trades
                    logger.warning(
                        f"Loss streak of {self.consecutive_losses} reached. "
                        f"Position size halved for next {self.loss_streak_cooldown_trades} trades."
                    )
                else:
                    # Extend / refresh the cooldown window
                    self.size_cooldown_trades_remaining = self.loss_streak_cooldown_trades
                    logger.warning(
                        f"Loss streak continues ({self.consecutive_losses}). "
                        f"Cooldown refreshed for {self.loss_streak_cooldown_trades} more trades."
                    )
        else:
            if self.consecutive_losses > 0:
                logger.info(f"Winning trade resets consecutive loss counter (was {self.consecutive_losses})")
            self.consecutive_losses = 0

        # Decrement cooldown counter on every trade (once activated)
        if self.size_cooldown_trades_remaining > 0:
            self.size_cooldown_trades_remaining -= 1
            logger.info(f"Cooldown trades remaining: {self.size_cooldown_trades_remaining}")

    # ------------------------------------------------------------------
    # Halt controls
    # ------------------------------------------------------------------

    def halt_trading(self, reason: str, duration_hours: Optional[float] = None) -> None:
        """Halt trading. If duration_hours given, auto-resumes after that period."""
        self.trading_halted = True
        self.halt_reason = reason
        if duration_hours is not None:
            self.halt_until = datetime.now(timezone.utc) + timedelta(hours=duration_hours)
            logger.error(
                f"Trading HALTED for {duration_hours:.1f}h. Reason: {reason}. "
                f"Resumes at {self.halt_until.isoformat()}"
            )
        else:
            self.halt_until = None
            logger.critical(f"Trading HALTED indefinitely. Reason: {reason}")

    def resume_trading(self) -> None:
        """Manually resume trading (clears halt)."""
        if not self.trading_halted:
            logger.info("resume_trading called but trading was not halted")
            return
        self._clear_timed_halt()
        logger.info("Trading manually resumed")

    def _clear_timed_halt(self) -> None:
        """Internal: clear halt state."""
        self.trading_halted = False
        self.halt_reason = ""
        self.halt_until = None

    # ------------------------------------------------------------------
    # Dead-man switch
    # ------------------------------------------------------------------

    def start_dead_man_heartbeat(self) -> None:
        """Start background thread sending cancel_all_after(90) every 60 seconds."""
        if self._dead_man_thread is not None and self._dead_man_thread.is_alive():
            logger.warning("Dead-man heartbeat already running")
            return
        if self.kraken_client is None:
            logger.warning("Cannot start dead-man heartbeat: no kraken_client configured")
            return

        self._dead_man_stop_event.clear()

        def _heartbeat() -> None:
            logger.info(
                f"Dead-man heartbeat started (timeout={self.dead_man_timeout}s, "
                f"interval={self.dead_man_interval}s)"
            )
            while not self._dead_man_stop_event.is_set():
                try:
                    self.kraken_client.cancel_all_after(self.dead_man_timeout)
                    logger.debug(f"Dead-man: cancel_all_after({self.dead_man_timeout}) sent")
                except Exception as exc:
                    logger.error(f"Dead-man heartbeat error: {exc}")
                # Use wait() so we can be interrupted promptly by stop_dead_man_heartbeat()
                self._dead_man_stop_event.wait(timeout=self.dead_man_interval)
            logger.info("Dead-man heartbeat stopped")

        self._dead_man_thread = threading.Thread(target=_heartbeat, daemon=True, name="dead-man-heartbeat")
        self._dead_man_thread.start()

    def stop_dead_man_heartbeat(self) -> None:
        """Stop the dead-man switch heartbeat."""
        self._dead_man_stop_event.set()
        if self._dead_man_thread is not None:
            self._dead_man_thread.join(timeout=self.dead_man_interval + 5)
            self._dead_man_thread = None
        logger.info("Dead-man heartbeat stopped")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return dict with current risk state for logging/alerting."""
        can, reason = self.can_trade()
        return {
            "trading_halted": self.trading_halted,
            "can_trade": can,
            "halt_reason": self.halt_reason,
            "halt_until": self.halt_until.isoformat() if self.halt_until else None,
            "high_water_mark": self.high_water_mark,
            "daily_start_equity": self.daily_start_equity,
            "daily_loss_date": self.daily_loss_date.isoformat(),
            "consecutive_losses": self.consecutive_losses,
            "size_cooldown_trades_remaining": self.size_cooldown_trades_remaining,
            "dead_man_heartbeat_alive": (
                self._dead_man_thread is not None and self._dead_man_thread.is_alive()
            ),
            "config": {
                "max_drawdown_pct": self.max_drawdown_pct,
                "daily_loss_limit_pct": self.daily_loss_limit_pct,
                "max_position_pct": self.max_position_pct,
                "max_margin_usage_pct": self.max_margin_usage_pct,
                "dead_man_timeout_sec": self.dead_man_timeout,
                "dead_man_interval_sec": self.dead_man_interval,
                "loss_streak_threshold": self.loss_streak_threshold,
                "loss_streak_cooldown_trades": self.loss_streak_cooldown_trades,
            },
        }
