"""Feature engineering pipeline — lean, normalized features for v2.

Builds ~25 high-signal features from OHLCV data.  Every feature is either
normalised (ratio / percentage) or stationary so it generalizes across
different price levels and volatility regimes.

Features REMOVED vs v1:
 - Raw price-level indicators (EMA values, SMA_200, SAR, KC/BB bands, OBV, AD)
 - Redundant momentum (Stoch, Williams %R, CCI, ROC, MOM — all measure the
   same thing as RSI from different angles)
 - Candlestick patterns (noise on hourly timeframe)
"""

from __future__ import annotations
import numpy as np
import polars as pl
import talib

from ..utils.logger import get_logger

logger = get_logger("v2.features.pipeline")


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """Build all features from raw OHLCV data.

    Args:
        df: Must have columns [timestamp, open, high, low, close, volume]

    Returns:
        DataFrame with original columns + all engineered features.
        Rows with NaN from lookback windows are dropped.
    """
    logger.info(f"Building features from {len(df):,} rows ...")

    o = df["open"].to_numpy().astype(np.float64)
    h = df["high"].to_numpy().astype(np.float64)
    lo = df["low"].to_numpy().astype(np.float64)
    c = df["close"].to_numpy().astype(np.float64)
    v = df["volume"].to_numpy().astype(np.float64)

    features: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # 1. Trend (normalized — distance from EMA in ATR units)
    # ------------------------------------------------------------------
    atr_14 = talib.ATR(h, lo, c, timeperiod=14)
    safe_atr = np.where((atr_14 == 0) | np.isnan(atr_14), 1, atr_14)
    features["atr_14"] = atr_14  # kept for labeling + exits

    for p in [8, 21, 55]:
        ema = talib.EMA(c, timeperiod=p)
        features[f"dist_ema_{p}"] = (c - ema) / safe_atr  # normalized distance

    macd, macd_sig, macd_hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    features["macd_hist_norm"] = macd_hist / safe_atr  # normalized MACD histogram

    features["adx_14"] = talib.ADX(h, lo, c, timeperiod=14)
    features["di_spread"] = (talib.PLUS_DI(h, lo, c, timeperiod=14) -
                             talib.MINUS_DI(h, lo, c, timeperiod=14))

    # ------------------------------------------------------------------
    # 2. Momentum / oscillators (keep only non-redundant, bounded ones)
    # ------------------------------------------------------------------
    features["rsi_14"] = talib.RSI(c, timeperiod=14)
    features["mfi_14"] = talib.MFI(h, lo, c, v, timeperiod=14)
    features["ultosc"] = talib.ULTOSC(h, lo, c)

    # ------------------------------------------------------------------
    # 3. Volatility (all normalized / percentage-based)
    # ------------------------------------------------------------------
    bb_up, bb_mid, bb_low = talib.BBANDS(c, timeperiod=20)
    safe_bb_range = np.where((bb_up - bb_low) == 0, 1, bb_up - bb_low)
    features["bb_pctb"] = (c - bb_low) / safe_bb_range   # 0-1 band position
    features["bb_width"] = (bb_up - bb_low) / np.where(bb_mid == 0, 1, bb_mid)
    features["natr_14"] = talib.NATR(h, lo, c, timeperiod=14)  # % ATR

    # ------------------------------------------------------------------
    # 4. Volume (relative, not raw)
    # ------------------------------------------------------------------
    vol_sma = talib.SMA(v, timeperiod=20)
    features["relative_volume"] = v / np.where(vol_sma == 0, 1, vol_sma)

    features["adosc_norm"] = talib.ADOSC(h, lo, c, v) / (v + 1)  # normalized

    # ------------------------------------------------------------------
    # 5. Returns at multiple scales (stationary by construction)
    # ------------------------------------------------------------------
    for lag in [1, 4, 12, 24]:
        ret = np.full_like(c, np.nan)
        ret[lag:] = np.log(c[lag:] / c[:-lag])
        features[f"log_return_{lag}"] = ret

    # ------------------------------------------------------------------
    # 6. Microstructure (all 0-1 or ratio-based)
    # ------------------------------------------------------------------
    bar_range_raw = h - lo
    features["bar_range"] = bar_range_raw / np.where(c == 0, 1, c)
    features["close_in_range"] = (c - lo) / np.where(bar_range_raw == 0, 1, bar_range_raw)
    features["body_ratio"] = np.abs(c - o) / np.where(bar_range_raw == 0, 1, bar_range_raw)

    # Volume-weighted momentum
    features["vol_weighted_mom"] = features["log_return_1"] * features["relative_volume"]

    # ------------------------------------------------------------------
    # 7. Regime features (stationary / bounded)
    # ------------------------------------------------------------------
    for w in [20, 50]:
        rvol = np.full_like(c, np.nan)
        for i in range(w, len(c)):
            rvol[i] = np.std(np.diff(np.log(c[i - w : i + 1])))
        features[f"realized_vol_{w}"] = rvol

    # Trend strength: |EMA21 - EMA55| / ATR14
    ema21 = talib.EMA(c, timeperiod=21)
    ema55 = talib.EMA(c, timeperiod=55)
    features["trend_strength"] = np.abs(ema21 - ema55) / safe_atr

    # Variance ratio (20-bar vs 1-bar) — <1 = mean-reverting, >1 = trending
    log_rets = np.diff(np.log(c), prepend=np.nan)
    vr = np.full_like(c, np.nan)
    for i in range(20, len(c)):
        var1 = np.var(log_rets[i - 19 : i + 1])
        long_rets = np.log(c[i] / c[i - 20])
        var20 = (long_rets ** 2) / 20
        vr[i] = var20 / var1 if var1 > 0 else np.nan
    features["variance_ratio_20"] = vr

    # ------------------------------------------------------------------
    # Assemble into Polars DataFrame
    # ------------------------------------------------------------------
    feat_df = df.clone()
    for name, arr in features.items():
        feat_df = feat_df.with_columns(pl.Series(name, arr))

    # Drop rows with NaN (from lookback windows)
    n_before = len(feat_df)
    feat_df = feat_df.drop_nulls()
    n_after = len(feat_df)
    logger.info(
        f"Features built: {len(features)} features, "
        f"{n_before - n_after} warmup rows dropped, "
        f"{n_after:,} rows remaining"
    )
    return feat_df


def get_feature_names(df: pl.DataFrame) -> list[str]:
    """Return list of feature column names (everything except OHLCV + timestamp + label-leaking cols).

    Excludes atr_14 because the triple barrier labels are DEFINED as
    multiples of ATR — including it would leak label information.
    """
    skip = {"timestamp", "open", "high", "low", "close", "volume", "atr_14"}
    return [c for c in df.columns if c not in skip]
