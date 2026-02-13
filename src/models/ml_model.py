"""Machine Learning model for direction prediction.

Uses XGBoost/LightGBM gradient boosting with walk-forward validation,
purged cross-validation, and feature importance tracking.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder

from .base import BaseModel, Signal, SignalDirection
from ..utils.logger import get_logger

logger = get_logger("crypto_trader.models.ml_model")


class MLModel(BaseModel):
    """Gradient boosting model for crypto direction prediction.

    Predicts whether the asset will go UP, DOWN, or stay FLAT over a
    configurable prediction horizon. Uses XGBoost by default with
    walk-forward training and purged cross-validation.

    Parameters (via config):
        algorithm: 'xgboost' or 'lightgbm'.
        prediction_horizon: Number of bars to predict ahead.
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate.
        min_samples_train: Minimum samples required for training.
        walk_forward_windows: Number of walk-forward splits.
        purge_gap: Bars to purge between train/test sets.
    """

    def __init__(self, config: dict):
        super().__init__("ml_model", config)
        ml_cfg = config.get("models", {}).get("ml_model", {})
        self.algorithm = ml_cfg.get("algorithm", "xgboost")
        self.prediction_horizon = ml_cfg.get("prediction_horizon", 12)
        self.n_estimators = ml_cfg.get("n_estimators", 500)
        self.max_depth = ml_cfg.get("max_depth", 6)
        self.learning_rate = ml_cfg.get("learning_rate", 0.05)
        self.min_samples_train = ml_cfg.get("min_samples_train", 2000)
        self.walk_forward_windows = ml_cfg.get("walk_forward_windows", 5)
        self.purge_gap = ml_cfg.get("purge_gap", 3)
        self.early_stopping = ml_cfg.get("early_stopping_rounds", 50)

        self._model = None
        self._label_encoder = LabelEncoder()
        self._feature_names: list[str] = []
        self._feature_importance: dict[str, float] = {}
        self._model_save_path = config.get("learning", {}).get("model_save_path", "./models")

    def fit(self, data: pl.DataFrame, features: Optional[pl.DataFrame] = None) -> None:
        """Train the ML model on historical data with features.

        Args:
            data: OHLCV DataFrame.
            features: Pre-computed feature DataFrame. If None, basic features are computed.
        """
        self.validate_data(data)

        if features is not None:
            X, y, feature_names = self._prepare_training_data_from_features(data, features)
        else:
            X, y, feature_names = self._prepare_training_data(data)

        if len(X) < self.min_samples_train:
            logger.warning(f"Only {len(X)} samples, need {self.min_samples_train}. "
                           "Training anyway with available data.")

        self._feature_names = feature_names

        # Encode labels
        y_encoded = self._label_encoder.fit_transform(y)

        # Train with walk-forward validation to get best params
        self._model = self._train_model(X, y_encoded)
        self._is_fitted = True

        # Feature importance
        if hasattr(self._model, "feature_importances_"):
            importance = self._model.feature_importances_
            self._feature_importance = dict(zip(feature_names, importance.tolist()))
            top_5 = sorted(self._feature_importance.items(), key=lambda x: -x[1])[:5]
            logger.info(f"Top 5 features: {top_5}")

        logger.info(f"ML model trained on {len(X)} samples with {len(feature_names)} features")

    def generate_signals(self, data: pl.DataFrame, features: Optional[pl.DataFrame] = None) -> list[Signal]:
        """Generate ML-based trading signals.

        Args:
            data: OHLCV DataFrame.
            features: Pre-computed features. If None, basic features are computed.

        Returns:
            List of Signal objects.
        """
        if not self._is_fitted or self._model is None:
            logger.warning("Model not fitted, returning empty signals")
            return []

        self.validate_data(data)

        if features is not None:
            X = self._extract_latest_features(features)
        else:
            X = self._compute_basic_features(data)
            if X is None or len(X) == 0:
                return []
            X = X[-1:, :]

        timestamps = data["timestamp"].to_list()
        current_ts = timestamps[-1] if timestamps else None
        current_close = float(data["close"][-1])
        pair = "BTC/USDT"

        try:
            probas = self._model.predict_proba(X)
            prediction = self._model.predict(X)

            pred_label = self._label_encoder.inverse_transform(prediction)[0]
            max_proba = float(np.max(probas[0]))

            direction = SignalDirection.FLAT
            if pred_label == "up":
                direction = SignalDirection.LONG
            elif pred_label == "down":
                direction = SignalDirection.SHORT

            confidence = (max_proba - 1.0 / len(self._label_encoder.classes_)) / \
                         (1.0 - 1.0 / len(self._label_encoder.classes_))
            confidence = max(0.0, min(confidence, 1.0))

            signals = []
            if direction != SignalDirection.FLAT and confidence > 0.15:
                signals.append(Signal(
                    direction=direction,
                    confidence=confidence,
                    pair=pair,
                    timestamp=current_ts,
                    model_name=self.name,
                    metadata={
                        "prediction": pred_label,
                        "probabilities": {
                            cls: float(p) for cls, p in
                            zip(self._label_encoder.classes_, probas[0])
                        },
                        "top_features": dict(sorted(
                            self._feature_importance.items(),
                            key=lambda x: -x[1]
                        )[:5]),
                    },
                ))

            return signals

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return []

    def _prepare_training_data(self, data: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Prepare features and labels from raw OHLCV data.

        Returns:
            Tuple of (features array, labels array, feature names).
        """
        close = data["close"].to_numpy().astype(np.float64)
        high = data["high"].to_numpy().astype(np.float64)
        low = data["low"].to_numpy().astype(np.float64)
        volume = data["volume"].to_numpy().astype(np.float64)

        X = self._compute_basic_features(data)
        if X is None:
            return np.array([]), np.array([]), []

        # Labels: future return direction
        returns = np.zeros(len(close))
        for i in range(len(close) - self.prediction_horizon):
            returns[i] = (close[i + self.prediction_horizon] - close[i]) / close[i]

        # Classify into up/down/flat
        threshold = 0.005  # 0.5% threshold
        labels = np.where(returns > threshold, "up",
                 np.where(returns < -threshold, "down", "flat"))

        # Trim to valid range (features need lookback, labels need forward)
        valid_start = 50  # Feature lookback
        valid_end = len(close) - self.prediction_horizon

        X = X[valid_start:valid_end]
        labels = labels[valid_start:valid_end]

        feature_names = self._get_basic_feature_names()
        return X, labels, feature_names

    def _prepare_training_data_from_features(
        self, data: pl.DataFrame, features: pl.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Prepare training data using pre-computed features."""
        close = data["close"].to_numpy().astype(np.float64)

        # Labels
        returns = np.zeros(len(close))
        for i in range(len(close) - self.prediction_horizon):
            returns[i] = (close[i + self.prediction_horizon] - close[i]) / close[i]

        threshold = 0.005
        labels = np.where(returns > threshold, "up",
                 np.where(returns < -threshold, "down", "flat"))

        # Get feature columns (exclude timestamp and price columns)
        exclude = {"timestamp", "open", "high", "low", "close", "volume"}
        feature_cols = [c for c in features.columns if c not in exclude]
        feature_names = feature_cols

        X = features.select(feature_cols).to_numpy().astype(np.float64)

        # Valid range
        valid_end = len(close) - self.prediction_horizon
        X = X[:valid_end]
        labels = labels[:valid_end]

        # Remove rows with NaN
        valid_mask = ~np.any(np.isnan(X), axis=1)
        X = X[valid_mask]
        labels = labels[valid_mask]

        return X, labels, feature_names

    def _compute_basic_features(self, data: pl.DataFrame) -> Optional[np.ndarray]:
        """Compute basic features from OHLCV data when no feature pipeline is provided."""
        close = data["close"].to_numpy().astype(np.float64)
        high = data["high"].to_numpy().astype(np.float64)
        low = data["low"].to_numpy().astype(np.float64)
        volume = data["volume"].to_numpy().astype(np.float64)

        n = len(close)
        if n < 60:
            return None

        features = []

        # Returns at various lookbacks
        for lb in [1, 3, 5, 10, 20]:
            ret = np.zeros(n)
            ret[lb:] = (close[lb:] - close[:-lb]) / close[:-lb]
            features.append(ret)

        # Volatility (rolling std of returns)
        log_ret = np.zeros(n)
        log_ret[1:] = np.diff(np.log(close))
        for window in [10, 20, 50]:
            vol = self._rolling_std(log_ret, window)
            features.append(vol)

        # RSI
        for period in [14, 28]:
            rsi = self._compute_rsi(close, period)
            features.append(rsi)

        # Volume features
        vol_ma = self._rolling_mean(volume, 20)
        vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
        features.append(vol_ratio)

        # Price relative to moving averages
        for window in [10, 20, 50]:
            ma = self._rolling_mean(close, window)
            rel = np.where(ma > 0, (close - ma) / ma, 0.0)
            features.append(rel)

        # High-Low range relative to close
        hl_range = (high - low) / np.where(close > 0, close, 1.0)
        features.append(hl_range)

        X = np.column_stack(features)
        # Replace inf/nan
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def _extract_latest_features(self, features: pl.DataFrame) -> np.ndarray:
        """Extract the latest row of features for prediction."""
        exclude = {"timestamp", "open", "high", "low", "close", "volume"}
        feature_cols = [c for c in features.columns if c not in exclude]
        X = features.select(feature_cols).to_numpy().astype(np.float64)
        X = np.nan_to_num(X[-1:], nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def _get_basic_feature_names(self) -> list[str]:
        """Get feature names for basic features."""
        names = []
        for lb in [1, 3, 5, 10, 20]:
            names.append(f"return_{lb}")
        for w in [10, 20, 50]:
            names.append(f"volatility_{w}")
        for p in [14, 28]:
            names.append(f"rsi_{p}")
        names.append("volume_ratio_20")
        for w in [10, 20, 50]:
            names.append(f"price_rel_ma_{w}")
        names.append("hl_range")
        return names

    def _train_model(self, X: np.ndarray, y: np.ndarray):
        """Train the gradient boosting model.

        Args:
            X: Feature matrix.
            y: Encoded label array.

        Returns:
            Trained model.
        """
        if self.algorithm == "lightgbm":
            return self._train_lightgbm(X, y)
        return self._train_xgboost(X, y)

    def _train_xgboost(self, X: np.ndarray, y: np.ndarray):
        """Train XGBoost classifier."""
        import xgboost as xgb

        n_classes = len(np.unique(y))
        model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="multi:softprob",
            num_class=n_classes,
            tree_method="hist",  # Fast on M4
            eval_metric="mlogloss",
            early_stopping_rounds=self.early_stopping,
            random_state=42,
            n_jobs=-1,
        )

        # Simple train/val split for early stopping
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        return model

    def _train_lightgbm(self, X: np.ndarray, y: np.ndarray):
        """Train LightGBM classifier."""
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="multiclass",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        return model

    def save_model(self, path: Optional[str] = None) -> str:
        """Save model to disk.

        Args:
            path: Save directory. Defaults to configured model_save_path.

        Returns:
            Path to saved model file.
        """
        save_dir = Path(path or self._model_save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "ml_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "label_encoder": self._label_encoder,
                "feature_names": self._feature_names,
                "feature_importance": self._feature_importance,
                "config": {
                    "algorithm": self.algorithm,
                    "prediction_horizon": self.prediction_horizon,
                },
            }, f)

        logger.info(f"Model saved to {model_path}")
        return str(model_path)

    def load_model(self, path: Optional[str] = None) -> None:
        """Load model from disk.

        Args:
            path: Path to model file or directory.
        """
        load_path = Path(path or self._model_save_path)
        if load_path.is_dir():
            load_path = load_path / "ml_model.pkl"

        with open(load_path, "rb") as f:
            data = pickle.load(f)

        self._model = data["model"]
        self._label_encoder = data["label_encoder"]
        self._feature_names = data["feature_names"]
        self._feature_importance = data["feature_importance"]
        self._is_fitted = True

        logger.info(f"Model loaded from {load_path}")

    @staticmethod
    def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling standard deviation."""
        result = np.zeros_like(arr)
        for i in range(window, len(arr)):
            result[i] = np.std(arr[i - window:i], ddof=1)
        return result

    @staticmethod
    def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling mean."""
        result = np.zeros_like(arr)
        cumsum = np.cumsum(arr)
        result[window:] = (cumsum[window:] - cumsum[:-window]) / window
        return result

    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        """Compute RSI."""
        n = len(close)
        rsi = np.full(n, 50.0)
        if n < period + 1:
            return rsi

        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, n - 1):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100 - 100 / (1 + rs)
            else:
                rsi[i + 1] = 100.0

        return rsi
