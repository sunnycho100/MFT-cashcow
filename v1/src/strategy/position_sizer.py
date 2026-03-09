"""Position sizing using Kelly Criterion and risk-based methods.

Implements fractional Kelly criterion with configurable risk parameters
for optimal position sizing in crypto markets.
"""

import numpy as np
from typing import Optional

from ..models.base import Signal, SignalDirection
from ..utils.logger import get_logger

logger = get_logger("crypto_trader.strategy.position_sizer")


class PositionSizer:
    """Calculate optimal position sizes using Kelly Criterion.

    Uses fractional Kelly to balance growth rate vs. drawdown risk.
    Applies additional constraints for crypto-specific risk management.

    Args:
        config: Configuration dictionary with risk parameters.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        risk_cfg = self.config.get("risk", {})

        self.max_position_pct = risk_cfg.get("max_position_pct", 0.10)
        self.max_total_exposure_pct = risk_cfg.get("max_total_exposure_pct", 0.50)
        self.kelly_fraction = risk_cfg.get("kelly_fraction", 0.25)  # Quarter Kelly
        self.min_position_pct = 0.01  # Minimum 1% to bother

    def calculate_size(
        self,
        signal: Signal,
        portfolio_value: float,
        current_price: float,
        win_rate: float = 0.5,
        avg_win: float = 0.02,
        avg_loss: float = 0.01,
        current_exposure: float = 0.0,
    ) -> dict:
        """Calculate position size for a trading signal.

        Args:
            signal: Trading signal with confidence.
            portfolio_value: Current portfolio value in USD.
            current_price: Current asset price.
            win_rate: Historical win rate (0-1).
            avg_win: Average winning trade return.
            avg_loss: Average losing trade return (positive number).
            current_exposure: Current total portfolio exposure (0-1).

        Returns:
            Dict with:
                - size_usd: Position size in USD.
                - size_units: Position size in asset units.
                - size_pct: Position as fraction of portfolio.
                - kelly_full: Full Kelly fraction.
                - kelly_used: Fractional Kelly used.
        """
        if signal.direction == SignalDirection.FLAT:
            return self._zero_position()

        if portfolio_value <= 0 or current_price <= 0:
            return self._zero_position()

        # Full Kelly criterion
        kelly_full = self._kelly_criterion(win_rate, avg_win, avg_loss)

        # Apply fraction
        kelly_used = kelly_full * self.kelly_fraction

        # Scale by signal confidence
        confidence_scaled = kelly_used * signal.confidence

        # Apply maximum position constraint
        position_pct = min(confidence_scaled, self.max_position_pct)

        # Check total exposure limit
        remaining_exposure = self.max_total_exposure_pct - current_exposure
        if remaining_exposure <= 0:
            logger.warning(f"Max exposure reached ({current_exposure:.2%}), no new positions")
            return self._zero_position()

        position_pct = min(position_pct, remaining_exposure)

        # Apply minimum
        if position_pct < self.min_position_pct:
            logger.debug(f"Position too small ({position_pct:.4%}), skipping")
            return self._zero_position()

        size_usd = portfolio_value * position_pct
        size_units = size_usd / current_price

        result = {
            "size_usd": round(size_usd, 2),
            "size_units": size_units,
            "size_pct": position_pct,
            "kelly_full": kelly_full,
            "kelly_used": kelly_used,
        }

        logger.info(f"Position size: ${size_usd:.2f} ({position_pct:.2%}) "
                     f"kelly_full={kelly_full:.4f} kelly_frac={kelly_used:.4f}")

        return result

    def calculate_size_risk_parity(
        self,
        signal: Signal,
        portfolio_value: float,
        current_price: float,
        asset_volatility: float,
        target_risk: float = 0.01,
    ) -> dict:
        """Calculate position size using risk parity / volatility targeting.

        Sizes positions inversely proportional to volatility so each
        position contributes roughly equal risk.

        Args:
            signal: Trading signal.
            portfolio_value: Portfolio value.
            current_price: Current price.
            asset_volatility: Annualized volatility of the asset.
            target_risk: Target risk contribution per position.

        Returns:
            Position size dict (same format as calculate_size).
        """
        if signal.direction == SignalDirection.FLAT or asset_volatility <= 0:
            return self._zero_position()

        # Position size = target_risk / volatility
        vol_adjusted_pct = target_risk / asset_volatility

        # Scale by confidence
        position_pct = vol_adjusted_pct * signal.confidence

        # Apply constraints
        position_pct = min(position_pct, self.max_position_pct)
        position_pct = max(position_pct, 0.0)

        if position_pct < self.min_position_pct:
            return self._zero_position()

        size_usd = portfolio_value * position_pct
        size_units = size_usd / current_price

        return {
            "size_usd": round(size_usd, 2),
            "size_units": size_units,
            "size_pct": position_pct,
            "kelly_full": 0.0,
            "kelly_used": 0.0,
        }

    @staticmethod
    def _kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Compute the Kelly Criterion fraction.

        Kelly fraction = (W * B - L) / B
        Where:
            W = win probability
            L = loss probability (1 - W)
            B = win/loss ratio (avg_win / avg_loss)

        Args:
            win_rate: Probability of winning (0-1).
            avg_win: Average win amount.
            avg_loss: Average loss amount (positive).

        Returns:
            Optimal Kelly fraction (can be negative = don't bet).
        """
        if avg_loss <= 0:
            return 0.0

        b = avg_win / avg_loss  # Win/loss ratio
        w = win_rate
        l = 1 - w

        kelly = (w * b - l) / b

        # Clamp to [0, 1] — negative Kelly means don't trade
        return max(0.0, min(kelly, 1.0))

    @staticmethod
    def _zero_position() -> dict:
        """Return zero position."""
        return {
            "size_usd": 0.0,
            "size_units": 0.0,
            "size_pct": 0.0,
            "kelly_full": 0.0,
            "kelly_used": 0.0,
        }
