"""Base model interface for all trading models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import polars as pl


class SignalDirection(Enum):
    """Trading signal direction."""
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class Signal:
    """A trading signal produced by a model.

    Attributes:
        direction: LONG, SHORT, or FLAT.
        confidence: Confidence level between 0 and 1.
        pair: Trading pair (e.g., 'BTC/USDT').
        timestamp: Signal generation time.
        model_name: Name of the model that generated this signal.
        metadata: Additional signal metadata (e.g., z-score, feature importance).
        stop_loss: Suggested stop loss price (optional).
        take_profit: Suggested take profit price (optional).
    """
    direction: SignalDirection
    confidence: float
    pair: str
    timestamp: Any = None
    model_name: str = ""
    metadata: dict = field(default_factory=dict)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @property
    def strength(self) -> float:
        """Signal strength: direction value * confidence."""
        return self.direction.value * self.confidence

    def __repr__(self) -> str:
        return (
            f"Signal({self.direction.name}, conf={self.confidence:.3f}, "
            f"pair={self.pair}, model={self.model_name})"
        )


class BaseModel(ABC):
    """Abstract base class for all trading models.

    Every model must implement `generate_signals` and `fit`.
    This ensures a consistent interface across mean reversion,
    momentum, ML, and ensemble models.
    """

    def __init__(self, name: str, config: dict):
        """Initialize model with name and configuration.

        Args:
            name: Human-readable model name.
            config: Model-specific configuration dictionary.
        """
        self.name = name
        self.config = config
        self._is_fitted = False

    @abstractmethod
    def generate_signals(self, data: pl.DataFrame) -> list[Signal]:
        """Generate trading signals from market data.

        Args:
            data: Polars DataFrame with at minimum columns:
                  ['timestamp', 'open', 'high', 'low', 'close', 'volume'].

        Returns:
            List of Signal objects.
        """
        ...

    @abstractmethod
    def fit(self, data: pl.DataFrame) -> None:
        """Fit/train the model on historical data.

        Args:
            data: Polars DataFrame with OHLCV data.
        """
        ...

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted/trained."""
        return self._is_fitted

    def validate_data(self, data: pl.DataFrame) -> bool:
        """Validate that input data has required columns.

        Args:
            data: Input DataFrame.

        Returns:
            True if valid.

        Raises:
            ValueError: If required columns are missing.
        """
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        columns = set(data.columns)
        missing = required - columns
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, fitted={self._is_fitted})"
