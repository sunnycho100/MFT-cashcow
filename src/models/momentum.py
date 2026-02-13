"""Momentum model with GARCH volatility filter.

Combines EMA crossover signals with GARCH(1,1) volatility regime
filtering and ADX trend strength confirmation.
"""

import numpy as np
import polars as pl
from typing import Optional

from .base import BaseModel, Signal, SignalDirection
from ..utils.logger import get_logger

logger = get_logger("crypto_trader.models.momentum")


class MomentumGARCHModel(BaseModel):
    """Momentum + GARCH(1,1) volatility-filtered trading model.

    Generates signals based on EMA crossovers, filtered by:
    - GARCH(1,1) volatility regime (only trade in favorable vol)
    - ADX trend strength (only trade when trend is strong enough)
    - ATR-based dynamic stop losses

    Parameters (via config):
        fast_ema: Fast EMA period.
        slow_ema: Slow EMA period.
        signal_ema: Signal line EMA period.
        adx_threshold: Minimum ADX for trend confirmation.
        atr_multiplier: ATR multiplier for stop losses.
        garch_vol_threshold: Percentile threshold for vol regime.
    """

    def __init__(self, config: dict):
        super().__init__("momentum", config)
        mom_cfg = config.get("models", {}).get("momentum", {})
        self.fast_ema = mom_cfg.get("fast_ema", 12)
        self.slow_ema = mom_cfg.get("slow_ema", 26)
        self.signal_ema = mom_cfg.get("signal_ema", 9)
        self.adx_threshold = mom_cfg.get("adx_threshold", 25.0)
        self.atr_multiplier = mom_cfg.get("atr_multiplier", 2.0)
        self.garch_vol_threshold = mom_cfg.get("garch_vol_threshold", 0.8)

        # Fitted state
        self._garch_params: Optional[dict] = None
        self._vol_percentiles: Optional[np.ndarray] = None
        self._current_vol: float = 0.0

    def fit(self, data: pl.DataFrame) -> None:
        """Fit GARCH(1,1) model on historical returns.

        Args:
            data: OHLCV DataFrame.
        """
        self.validate_data(data)
        close = data["close"].to_numpy().astype(np.float64)
        returns = np.diff(np.log(close)) * 100  # Percentage log returns

        if len(returns) < 100:
            logger.warning("Insufficient data for GARCH fitting, using fallback")
            self._garch_params = {"omega": 0.01, "alpha": 0.1, "beta": 0.85}
            self._vol_percentiles = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
            self._is_fitted = True
            return

        try:
            from arch import arch_model

            model = arch_model(returns, vol="Garch", p=1, q=1, dist="normal")
            result = model.fit(disp="off", show_warning=False)

            self._garch_params = {
                "omega": float(result.params.get("omega", 0.01)),
                "alpha": float(result.params.get("alpha[1]", 0.1)),
                "beta": float(result.params.get("beta[1]", 0.85)),
            }

            # Store conditional volatility percentiles for regime detection
            cond_vol = result.conditional_volatility
            self._vol_percentiles = np.percentile(cond_vol, [10, 25, 50, 75, 90])
            self._current_vol = float(cond_vol[-1])

            logger.info(f"GARCH fitted: omega={self._garch_params['omega']:.6f}, "
                         f"alpha={self._garch_params['alpha']:.4f}, "
                         f"beta={self._garch_params['beta']:.4f}")

        except Exception as e:
            logger.warning(f"GARCH fitting failed: {e}, using fallback parameters")
            self._garch_params = {"omega": 0.01, "alpha": 0.1, "beta": 0.85}
            vol = np.abs(returns)
            self._vol_percentiles = np.percentile(vol, [10, 25, 50, 75, 90])

        self._is_fitted = True

    def generate_signals(self, data: pl.DataFrame) -> list[Signal]:
        """Generate momentum signals with GARCH vol filter.

        Args:
            data: OHLCV DataFrame with sufficient history.

        Returns:
            List of Signal objects.
        """
        if not self._is_fitted:
            logger.warning("Model not fitted, returning empty signals")
            return []

        self.validate_data(data)
        close = data["close"].to_numpy().astype(np.float64)
        high = data["high"].to_numpy().astype(np.float64)
        low = data["low"].to_numpy().astype(np.float64)
        timestamps = data["timestamp"].to_list()

        if len(close) < self.slow_ema + 10:
            logger.warning("Insufficient data for signal generation")
            return []

        # Compute indicators
        fast = self._ema(close, self.fast_ema)
        slow = self._ema(close, self.slow_ema)
        macd_line = fast - slow
        signal_line = self._ema(macd_line, self.signal_ema)
        adx = self._compute_adx(high, low, close, period=14)
        atr = self._compute_atr(high, low, close, period=14)

        # GARCH volatility regime check
        vol_ok = self._check_vol_regime(close)

        # Current values (latest bar)
        current_macd = macd_line[-1]
        prev_macd = macd_line[-2]
        current_signal = signal_line[-1]
        prev_signal = signal_line[-2]
        current_adx = adx[-1] if len(adx) > 0 else 0.0
        current_atr = atr[-1] if len(atr) > 0 else 0.0
        current_close = close[-1]
        current_ts = timestamps[-1] if timestamps else None

        # Determine pair from data (default)
        pair = "BTC/USDT"

        signals = []

        # EMA crossover signal
        cross_up = prev_macd <= prev_signal and current_macd > current_signal
        cross_down = prev_macd >= prev_signal and current_macd < current_signal

        # Trend strength filter
        trend_strong = current_adx >= self.adx_threshold

        # Build signal
        direction = SignalDirection.FLAT
        confidence = 0.0

        if cross_up and trend_strong and vol_ok:
            direction = SignalDirection.LONG
            confidence = self._compute_confidence(current_adx, current_macd - current_signal, close)
        elif cross_down and trend_strong and vol_ok:
            direction = SignalDirection.SHORT
            confidence = self._compute_confidence(current_adx, current_signal - current_macd, close)

        # Also detect strong trend continuation (no crossover needed)
        if direction == SignalDirection.FLAT and trend_strong and vol_ok:
            if current_macd > current_signal and current_macd > 0:
                direction = SignalDirection.LONG
                confidence = self._compute_confidence(current_adx, current_macd - current_signal, close) * 0.5
            elif current_macd < current_signal and current_macd < 0:
                direction = SignalDirection.SHORT
                confidence = self._compute_confidence(current_adx, current_signal - current_macd, close) * 0.5

        if direction != SignalDirection.FLAT and confidence > 0.1:
            # ATR-based stops
            stop_distance = current_atr * self.atr_multiplier
            if direction == SignalDirection.LONG:
                stop_loss = current_close - stop_distance
                take_profit = current_close + stop_distance * 2
            else:
                stop_loss = current_close + stop_distance
                take_profit = current_close - stop_distance * 2

            signals.append(Signal(
                direction=direction,
                confidence=min(confidence, 1.0),
                pair=pair,
                timestamp=current_ts,
                model_name=self.name,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    "macd": float(current_macd),
                    "signal": float(current_signal),
                    "adx": float(current_adx),
                    "atr": float(current_atr),
                    "vol_regime_ok": vol_ok,
                    "crossover": cross_up or cross_down,
                },
            ))

        return signals

    def _check_vol_regime(self, close: np.ndarray) -> bool:
        """Check if current volatility regime is favorable for momentum trading.

        We want moderate volatility — not too low (no movement) and not too
        high (choppy/mean-reverting).

        Returns:
            True if vol regime is favorable.
        """
        if self._vol_percentiles is None:
            return True

        returns = np.diff(np.log(close[-50:])) * 100
        current_vol = np.std(returns) if len(returns) > 5 else 0.0

        # Favorable: between 25th and 90th percentile
        vol_25 = self._vol_percentiles[1]
        vol_90 = self._vol_percentiles[4]

        return vol_25 <= current_vol <= vol_90

    def _compute_confidence(self, adx: float, macd_diff: float, close: np.ndarray) -> float:
        """Compute signal confidence based on indicator strength.

        Args:
            adx: Current ADX value.
            macd_diff: Difference between MACD and signal line.
            close: Close prices for normalization.

        Returns:
            Confidence between 0 and 1.
        """
        # ADX component (stronger trend = higher confidence)
        adx_score = min((adx - self.adx_threshold) / 30.0, 1.0) if adx > self.adx_threshold else 0.0

        # MACD divergence component (normalized by price)
        price_scale = np.mean(close[-20:])
        macd_score = min(abs(macd_diff) / (price_scale * 0.01), 1.0) if price_scale > 0 else 0.0

        return (adx_score * 0.6 + macd_score * 0.4)

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Compute Exponential Moving Average.

        Args:
            data: Input price array.
            period: EMA period.

        Returns:
            EMA array of same length as input (first `period` values are SMA).
        """
        alpha = 2.0 / (period + 1)
        ema = np.empty_like(data, dtype=np.float64)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    @staticmethod
    def _compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Compute Average Directional Index (ADX).

        Args:
            high: High prices.
            low: Low prices.
            close: Close prices.
            period: Smoothing period.

        Returns:
            ADX array.
        """
        n = len(close)
        if n < period + 1:
            return np.zeros(n)

        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Wilder smoothing
        def wilder_smooth(arr, p):
            result = np.empty_like(arr)
            result[:p] = np.nan
            result[p - 1] = np.sum(arr[:p])
            for i in range(p, len(arr)):
                result[i] = result[i - 1] - result[i - 1] / p + arr[i]
            return result

        atr_smooth = wilder_smooth(tr, period)
        plus_dm_smooth = wilder_smooth(plus_dm, period)
        minus_dm_smooth = wilder_smooth(minus_dm, period)

        # Directional Indicators
        plus_di = 100.0 * plus_dm_smooth / np.where(atr_smooth > 0, atr_smooth, 1.0)
        minus_di = 100.0 * minus_dm_smooth / np.where(atr_smooth > 0, atr_smooth, 1.0)

        # DX
        di_sum = plus_di + minus_di
        dx = 100.0 * np.abs(plus_di - minus_di) / np.where(di_sum > 0, di_sum, 1.0)

        # ADX (smoothed DX)
        adx = np.full(n, np.nan)
        valid_dx = dx[~np.isnan(dx)]
        if len(valid_dx) >= period:
            adx_vals = wilder_smooth(valid_dx, period)
            offset = n - len(adx_vals)
            adx[offset:] = adx_vals

        # Fill NaN with 0
        adx = np.nan_to_num(adx, nan=0.0)
        return adx

    @staticmethod
    def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Compute Average True Range.

        Args:
            high: High prices.
            low: Low prices.
            close: Close prices.
            period: ATR period.

        Returns:
            ATR array.
        """
        n = len(close)
        if n < 2:
            return np.zeros(n)

        tr = np.empty(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        atr = np.empty(n)
        atr[:period] = np.nan
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        atr = np.nan_to_num(atr, nan=0.0)
        return atr
