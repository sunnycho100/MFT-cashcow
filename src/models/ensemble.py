"""Ensemble model combining multiple trading signal sources.

Implements weighted voting with confidence scaling and disagreement
filtering across mean reversion, momentum, and ML models.
"""

import numpy as np
import polars as pl
from typing import Optional
from collections import defaultdict

from .base import BaseModel, Signal, SignalDirection
from .mean_reversion import MeanReversionModel
from .momentum import MomentumGARCHModel
from .ml_model import MLModel
from ..utils.logger import get_logger

logger = get_logger("crypto_trader.models.ensemble")


class EnsembleModel(BaseModel):
    """Weighted ensemble of multiple trading models.

    Combines signals from mean reversion, momentum/GARCH, and ML models
    using configurable weights. Includes a disagreement filter to avoid
    taking positions when models conflict significantly.

    Parameters (via config):
        weights: Dict mapping model names to weights.
        min_agreement: Minimum weighted agreement to generate a signal.
        confidence_scaling: Whether to scale position size by confidence.
    """

    def __init__(self, config: dict):
        super().__init__("ensemble", config)
        ens_cfg = config.get("models", {}).get("ensemble", {})
        self.weights = ens_cfg.get("weights", {
            "mean_reversion": 0.33,
            "momentum": 0.33,
            "ml_model": 0.34,
        })
        self.min_agreement = ens_cfg.get("min_agreement", 0.6)
        self.confidence_scaling = ens_cfg.get("confidence_scaling", True)

        # Sub-models
        self.models: dict[str, BaseModel] = {}
        self._initialize_models(config)

    def _initialize_models(self, config: dict) -> None:
        """Initialize sub-models based on configuration."""
        model_configs = config.get("models", {})

        if model_configs.get("mean_reversion", {}).get("enabled", True):
            self.models["mean_reversion"] = MeanReversionModel(config)

        if model_configs.get("momentum", {}).get("enabled", True):
            self.models["momentum"] = MomentumGARCHModel(config)

        if model_configs.get("ml_model", {}).get("enabled", True):
            self.models["ml_model"] = MLModel(config)

        logger.info(f"Ensemble initialized with models: {list(self.models.keys())}")

    def fit(self, data: pl.DataFrame, **kwargs) -> None:
        """Fit all sub-models.

        Args:
            data: OHLCV DataFrame.
            **kwargs: Additional arguments passed to sub-model fit methods.
                      - pair_data: Dict for mean reversion pairs.
                      - features: Feature DataFrame for ML model.
        """
        self.validate_data(data)

        for name, model in self.models.items():
            try:
                if name == "mean_reversion":
                    model.fit(data, pair_data=kwargs.get("pair_data"))
                elif name == "ml_model":
                    model.fit(data, features=kwargs.get("features"))
                else:
                    model.fit(data)
                logger.info(f"Sub-model '{name}' fitted successfully")
            except Exception as e:
                logger.error(f"Failed to fit sub-model '{name}': {e}")

        self._is_fitted = any(m.is_fitted for m in self.models.values())

    def generate_signals(self, data: pl.DataFrame, **kwargs) -> list[Signal]:
        """Generate ensemble signals by combining sub-model outputs.

        Args:
            data: OHLCV DataFrame.
            **kwargs: Additional arguments for sub-models.

        Returns:
            List of ensemble Signal objects.
        """
        if not self._is_fitted:
            logger.warning("No models fitted, returning empty signals")
            return []

        self.validate_data(data)

        # Collect signals from all sub-models
        all_signals: dict[str, list[Signal]] = {}
        for name, model in self.models.items():
            if not model.is_fitted:
                continue
            try:
                if name == "mean_reversion":
                    sigs = model.generate_signals(data, pair_data=kwargs.get("pair_data"))
                elif name == "ml_model":
                    sigs = model.generate_signals(data, features=kwargs.get("features"))
                else:
                    sigs = model.generate_signals(data)
                all_signals[name] = sigs
                logger.debug(f"Model '{name}' generated {len(sigs)} signals")
            except Exception as e:
                logger.error(f"Signal generation failed for '{name}': {e}")
                all_signals[name] = []

        # Group signals by pair
        pair_signals: dict[str, dict[str, Signal]] = defaultdict(dict)
        for model_name, signals in all_signals.items():
            for sig in signals:
                pair_signals[sig.pair][model_name] = sig

        # Combine signals for each pair
        ensemble_signals = []
        for pair, model_sigs in pair_signals.items():
            combined = self._combine_signals(pair, model_sigs)
            if combined is not None:
                ensemble_signals.append(combined)

        return ensemble_signals

    def _combine_signals(self, pair: str, model_signals: dict[str, Signal]) -> Optional[Signal]:
        """Combine signals from multiple models for a single pair.

        Uses weighted voting with confidence to determine the ensemble direction.
        Applies disagreement filtering.

        Args:
            pair: Trading pair.
            model_signals: Dict mapping model names to their signals.

        Returns:
            Combined Signal or None if models disagree too much.
        """
        if not model_signals:
            return None

        # Compute weighted scores
        total_weight = 0.0
        weighted_direction = 0.0
        weighted_confidence = 0.0
        stop_losses = []
        take_profits = []
        all_metadata = {}

        for model_name, signal in model_signals.items():
            weight = self.weights.get(model_name, 0.0)
            total_weight += weight

            direction_val = signal.direction.value  # -1, 0, or 1
            conf = signal.confidence

            weighted_direction += weight * direction_val * conf
            weighted_confidence += weight * conf

            if signal.stop_loss is not None:
                stop_losses.append(signal.stop_loss)
            if signal.take_profit is not None:
                take_profits.append(signal.take_profit)

            all_metadata[model_name] = {
                "direction": signal.direction.name,
                "confidence": signal.confidence,
                **signal.metadata,
            }

        if total_weight < 1e-10:
            return None

        # Normalize
        norm_direction = weighted_direction / total_weight
        norm_confidence = weighted_confidence / total_weight

        # Determine final direction
        if norm_direction > 0.1:
            direction = SignalDirection.LONG
        elif norm_direction < -0.1:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.FLAT

        # Check agreement level
        agreement = abs(norm_direction)
        if agreement < self.min_agreement * norm_confidence:
            logger.debug(f"Pair {pair}: insufficient agreement ({agreement:.3f} < "
                          f"{self.min_agreement * norm_confidence:.3f}), skipping")
            return None

        if direction == SignalDirection.FLAT:
            return None

        # Final confidence
        final_confidence = min(agreement * norm_confidence, 1.0)
        if self.confidence_scaling:
            final_confidence *= agreement  # Scale down on disagreement

        # Average stop/take profit from models that provided them
        avg_stop = float(np.mean(stop_losses)) if stop_losses else None
        avg_tp = float(np.mean(take_profits)) if take_profits else None

        # Get timestamp from any signal
        timestamp = next(iter(model_signals.values())).timestamp

        return Signal(
            direction=direction,
            confidence=final_confidence,
            pair=pair,
            timestamp=timestamp,
            model_name="ensemble",
            stop_loss=avg_stop,
            take_profit=avg_tp,
            metadata={
                "weighted_direction": float(norm_direction),
                "agreement": float(agreement),
                "models_contributing": len(model_signals),
                "model_details": all_metadata,
            },
        )

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """Update ensemble model weights.

        Args:
            new_weights: Dict mapping model names to new weights.
        """
        self.weights.update(new_weights)
        # Normalize
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        logger.info(f"Ensemble weights updated: {self.weights}")
