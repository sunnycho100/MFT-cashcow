"""Feature engineering pipeline — the core alpha source for v2.

Assembles 100+ features from OHLCV data using TA-Lib and custom
calculations. Designed for LightGBM consumption.
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
    # 1. Trend indicators
    # ------------------------------------------------------------------
    for p in [8, 21, 55, 200]:
        features[f"ema_{p}"] = talib.EMA(c, timeperiod=p)
    features["sma_50"] = talib.SMA(c, timeperiod=50)
    features["sma_200"] = talib.SMA(c, timeperiod=200)

    macd, macd_sig, macd_hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    features["macd"] = macd
    features["macd_signal"] = macd_sig
    features["macd_hist"] = macd_hist

    features["adx_14"] = talib.ADX(h, lo, c, timeperiod=14)
    features["plus_di"] = talib.PLUS_DI(h, lo, c, timeperiod=14)
    features["minus_di"] = talib.MINUS_DI(h, lo, c, timeperiod=14)
    features["sar"] = talib.SAR(h, lo)
    features["aroon_up"], features["aroon_down"] = talib.AROON(h, lo, timeperiod=14)

    # ------------------------------------------------------------------
    # 2. Momentum / oscillators
    # ------------------------------------------------------------------
    features["rsi_14"] = talib.RSI(c, timeperiod=14)
    features["rsi_7"] = talib.RSI(c, timeperiod=7)
    features["stoch_k"], features["stoch_d"] = talib.STOCH(h, lo, c)
    # StochRSI: apply STOCHF to RSI values (use RSI as all three inputs)
    rsi_arr = talib.RSI(c, timeperiod=14)
    features["stochrsi_k"], features["stochrsi_d"] = talib.STOCHF(
        rsi_arr, rsi_arr, rsi_arr, fastk_period=14, fastd_period=3
    )
    features["willr"] = talib.WILLR(h, lo, c, timeperiod=14)
    features["cci_14"] = talib.CCI(h, lo, c, timeperiod=14)
    features["roc_10"] = talib.ROC(c, timeperiod=10)
    features["mom_10"] = talib.MOM(c, timeperiod=10)
    features["ultosc"] = talib.ULTOSC(h, lo, c)

    # ------------------------------------------------------------------
    # 3. Volatility
    # ------------------------------------------------------------------
    bb_up, bb_mid, bb_low = talib.BBANDS(c, timeperiod=20)
    features["bb_upper"] = bb_up
    features["bb_lower"] = bb_low
    features["bb_width"] = (bb_up - bb_low) / np.where(bb_mid == 0, 1, bb_mid)
    features["bb_pctb"] = (c - bb_low) / np.where((bb_up - bb_low) == 0, 1, bb_up - bb_low)

    features["atr_14"] = talib.ATR(h, lo, c, timeperiod=14)
    features["natr_14"] = talib.NATR(h, lo, c, timeperiod=14)

    # Keltner channel approximation
    kc_mid = talib.EMA(c, timeperiod=20)
    kc_atr = talib.ATR(h, lo, c, timeperiod=20)
    features["kc_upper"] = kc_mid + 2 * kc_atr
    features["kc_lower"] = kc_mid - 2 * kc_atr

    # ------------------------------------------------------------------
    # 4. Volume
    # ------------------------------------------------------------------
    features["obv"] = talib.OBV(c, v)
    features["mfi_14"] = talib.MFI(h, lo, c, v, timeperiod=14)
    features["ad_line"] = talib.AD(h, lo, c, v)
    features["adosc"] = talib.ADOSC(h, lo, c, v)

    # Relative volume (vs 20-bar SMA)
    vol_sma = talib.SMA(v, timeperiod=20)
    features["relative_volume"] = v / np.where(vol_sma == 0, 1, vol_sma)

    # ------------------------------------------------------------------
    # 5. Returns at multiple scales
    # ------------------------------------------------------------------
    for lag in [1, 2, 4, 8, 12, 24, 48]:
        ret = np.full_like(c, np.nan)
        ret[lag:] = np.log(c[lag:] / c[:-lag])
        features[f"log_return_{lag}"] = ret

    # ------------------------------------------------------------------
    # 6. Microstructure
    # ------------------------------------------------------------------
    features["bar_range"] = (h - lo) / np.where(c == 0, 1, c)
    features["close_in_range"] = (c - lo) / np.where((h - lo) == 0, 1, h - lo)
    features["body_ratio"] = np.abs(c - o) / np.where((h - lo) == 0, 1, h - lo)

    # Volume-weighted momentum
    ret1 = features["log_return_1"]
    features["vol_weighted_mom"] = ret1 * features["relative_volume"]

    # ------------------------------------------------------------------
    # 7. Regime features (let LightGBM learn regimes)
    # ------------------------------------------------------------------
    # Rolling realized volatility
    for w in [20, 50]:
        rvol = np.full_like(c, np.nan)
        for i in range(w, len(c)):
            rvol[i] = np.std(np.diff(np.log(c[i - w : i + 1])))
        features[f"realized_vol_{w}"] = rvol

    # Trend strength: |EMA21 - EMA55| / ATR14
    ema21 = features["ema_21"]
    ema55 = features["ema_55"]
    atr14 = features["atr_14"]
    with np.errstate(divide="ignore", invalid="ignore"):
        features["trend_strength"] = np.abs(ema21 - ema55) / np.where(atr14 == 0, 1, atr14)

    # Variance ratio (20-bar vs 1-bar) — <1 = mean-reverting, >1 = trending
    log_rets = np.diff(np.log(c), prepend=np.nan)
    vr = np.full_like(c, np.nan)
    for i in range(20, len(c)):
        var1 = np.var(log_rets[i - 19 : i + 1])
        long_rets = np.log(c[i] / c[i - 20])
        var20 = (long_rets ** 2) / 20
        vr[i] = var20 / var1 if var1 > 0 else np.nan
    features["variance_ratio_20"] = vr

    # ADX regime bucket
    features["adx_regime"] = np.where(features["adx_14"] > 25, 1.0,
                              np.where(features["adx_14"] < 20, -1.0, 0.0))

    # ------------------------------------------------------------------
    # 8. Candlestick patterns (encoded as -100/0/+100 by TA-Lib)
    # ------------------------------------------------------------------
    features["cdl_doji"] = talib.CDLDOJI(o, h, lo, c).astype(np.float64)
    features["cdl_engulfing"] = talib.CDLENGULFING(o, h, lo, c).astype(np.float64)
    features["cdl_hammer"] = talib.CDLHAMMER(o, h, lo, c).astype(np.float64)
    features["cdl_morningstar"] = talib.CDLMORNINGSTAR(o, h, lo, c).astype(np.float64)

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
    """Return list of feature column names (everything except OHLCV + timestamp)."""
    skip = {"timestamp", "open", "high", "low", "close", "volume"}
    return [c for c in df.columns if c not in skip]
