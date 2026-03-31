"""Feature engineering pipeline — aggressive alpha features for v2.

Builds ~50+ high-signal features from OHLCV data.  Every feature is either
normalised (ratio / percentage) or stationary so it generalizes across
different price levels and volatility regimes.

New in v2.1:
 - VWAP z-score (mean-reversion signal)
 - Keltner squeeze (volatility breakout)
 - Multi-timeframe (4H/daily via rolling windows — no lookahead)
 - Calendar (hour-of-day sin/cos)
 - Cumulative delta proxy (orderflow from OHLCV)
 - Volume profile POC distance
 - RSI divergence detection
 - ATR ratio (short/long vol compression)
 - Close-in-range persistence
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
    safe_vol_sma = np.where((vol_sma == 0) | np.isnan(vol_sma), 1, vol_sma)
    features["relative_volume"] = v / safe_vol_sma

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
    safe_bar_range = np.where(bar_range_raw == 0, 1, bar_range_raw)
    features["bar_range"] = bar_range_raw / np.where(c == 0, 1, c)
    features["close_in_range"] = (c - lo) / safe_bar_range
    features["body_ratio"] = np.abs(c - o) / safe_bar_range

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

    # ==================================================================
    # NEW ALPHA FEATURES (v2.1)
    # ==================================================================

    # ------------------------------------------------------------------
    # 8. VWAP z-score — mean reversion signal
    # ------------------------------------------------------------------
    typical_price = (h + lo + c) / 3.0
    cum_tp_vol = np.nancumsum(typical_price * v)
    cum_vol = np.nancumsum(v)
    safe_cum_vol = np.where(cum_vol == 0, 1, cum_vol)

    for lookback in [20, 48]:
        vwap = np.full_like(c, np.nan)
        vwap_std = np.full_like(c, np.nan)
        for i in range(lookback, len(c)):
            window_tp = typical_price[i - lookback + 1 : i + 1]
            window_v = v[i - lookback + 1 : i + 1]
            total_v = window_v.sum()
            if total_v > 0:
                vwap_val = (window_tp * window_v).sum() / total_v
                vwap[i] = vwap_val
                # Deviation
                deviations = window_tp - vwap_val
                vwap_std[i] = np.sqrt((deviations ** 2 * window_v).sum() / total_v)
        safe_std = np.where((vwap_std == 0) | np.isnan(vwap_std), 1, vwap_std)
        features[f"vwap_zscore_{lookback}"] = (c - vwap) / safe_std

    # ------------------------------------------------------------------
    # 9. Keltner Squeeze — BB inside KC = low vol, breakout incoming
    # ------------------------------------------------------------------
    kc_mid = talib.EMA(c, timeperiod=20)
    kc_upper = kc_mid + 1.5 * atr_14
    kc_lower = kc_mid - 1.5 * atr_14

    # Squeeze = BB inside KC (1 = squeeze on, 0 = off)
    squeeze_on = ((bb_up < kc_upper) & (bb_low > kc_lower)).astype(np.float64)
    features["squeeze_on"] = squeeze_on

    # Squeeze momentum (diff of midline)
    squeeze_val = (c - (kc_mid + bb_mid) / 2) / safe_atr
    features["squeeze_momentum"] = squeeze_val

    # Bars since last squeeze release (normalized)
    bars_since_squeeze = np.full_like(c, np.nan)
    last_squeeze_end = -999
    for i in range(1, len(c)):
        if squeeze_on[i] == 0 and (i == 0 or squeeze_on[i - 1] == 1):
            last_squeeze_end = i
        bars_since_squeeze[i] = min(i - last_squeeze_end, 50) / 50.0  # 0-1 normalized
    features["bars_since_squeeze"] = bars_since_squeeze

    # ------------------------------------------------------------------
    # 10. Multi-timeframe via rolling windows (NO lookahead)
    # ------------------------------------------------------------------
    # 4H equivalent: RSI_56 (56 1H bars ≈ 4H RSI_14)
    features["rsi_56"] = talib.RSI(c, timeperiod=56)

    # Daily EMA distance (168 bars ≈ 7 day EMA, 504 ≈ 21 day)
    for p in [168, 504]:
        ema_long = talib.EMA(c, timeperiod=p)
        features[f"dist_ema_{p}"] = (c - ema_long) / safe_atr

    # 4H ADX (56 bars)
    features["adx_56"] = talib.ADX(h, lo, c, timeperiod=56)

    # ATR ratio: short vol / long vol — <1 = compression (breakout likely)
    atr_56 = talib.ATR(h, lo, c, timeperiod=56)
    safe_atr_56 = np.where((atr_56 == 0) | np.isnan(atr_56), 1, atr_56)
    features["atr_ratio_14_56"] = atr_14 / safe_atr_56

    # Daily NATR for regime detection
    features["natr_56"] = talib.NATR(h, lo, c, timeperiod=56)

    # ------------------------------------------------------------------
    # 11. Calendar features — crypto has strong hour-of-day effects
    # ------------------------------------------------------------------
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        if ts.dtype == pl.Utf8:
            ts = ts.str.to_datetime()
        hours = ts.dt.hour().to_numpy().astype(np.float64)
        # Sin/cos encoding (captures cyclical nature)
        features["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        features["hour_cos"] = np.cos(2 * np.pi * hours / 24)
        # Day of week
        dow = ts.dt.weekday().to_numpy().astype(np.float64)
        features["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        features["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # ------------------------------------------------------------------
    # 12. Cumulative delta proxy (orderflow from OHLCV)
    # ------------------------------------------------------------------
    # CLV (Close Location Value) × Volume approximates buy/sell pressure
    clv = np.where(safe_bar_range > 0,
                   (2 * c - lo - h) / safe_bar_range,
                   0.0)
    delta_proxy = clv * v
    # Cumulative delta over rolling windows
    for w in [12, 48]:
        cum_delta = np.full_like(c, np.nan)
        for i in range(w, len(c)):
            cum_delta[i] = delta_proxy[i - w + 1 : i + 1].sum()
        # Normalize by average volume
        cum_delta_norm = cum_delta / (safe_vol_sma * w + 1)
        features[f"cum_delta_{w}"] = cum_delta_norm

    # ------------------------------------------------------------------
    # 13. Volume profile — distance from POC (Point of Control)
    # ------------------------------------------------------------------
    poc_dist = np.full_like(c, np.nan)
    for i in range(100, len(c)):
        window_c = c[i - 99 : i + 1]
        window_v = v[i - 99 : i + 1]
        # Simple POC: price level with highest volume
        n_bins = 20
        price_min, price_max = window_c.min(), window_c.max()
        if price_max > price_min:
            bins = np.linspace(price_min, price_max, n_bins + 1)
            bin_idx = np.digitize(window_c, bins) - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            bin_vol = np.zeros(n_bins)
            for bi, vi in zip(bin_idx, window_v):
                bin_vol[bi] += vi
            poc_bin = np.argmax(bin_vol)
            poc_price = (bins[poc_bin] + bins[poc_bin + 1]) / 2
            poc_dist[i] = (c[i] - poc_price) / safe_atr[i]
    features["poc_distance"] = poc_dist

    # ------------------------------------------------------------------
    # 14. RSI divergence detector (price vs RSI movement)
    # ------------------------------------------------------------------
    rsi = features["rsi_14"]
    for lookback in [14, 28]:
        price_change = np.full_like(c, np.nan)
        rsi_change = np.full_like(c, np.nan)
        price_change[lookback:] = c[lookback:] - c[:-lookback]
        rsi_change[lookback:] = rsi[lookback:] - rsi[:-lookback]
        # Divergence = price up but RSI down (bearish) or vice versa (bullish)
        # Positive = bullish divergence, negative = bearish
        safe_pc = np.where(np.abs(price_change) < 1e-10, 1e-10, price_change)
        features[f"rsi_divergence_{lookback}"] = -np.sign(safe_pc) * rsi_change

    # ------------------------------------------------------------------
    # 15. Close-in-range persistence (momentum continuation signal)
    # ------------------------------------------------------------------
    cir = features["close_in_range"]
    cir_persistence = np.full_like(c, np.nan)
    for i in range(5, len(c)):
        # Average CIR over last 5 bars — >0.7 = strong closing high, <0.3 = closing low
        cir_persistence[i] = np.nanmean(cir[i - 4 : i + 1])
    features["cir_persistence_5"] = cir_persistence

    # ------------------------------------------------------------------
    # 16. Longer-term returns for trend context
    # ------------------------------------------------------------------
    for lag in [48, 168]:
        ret = np.full_like(c, np.nan)
        ret[lag:] = np.log(c[lag:] / c[:-lag])
        features[f"log_return_{lag}"] = ret

    # ------------------------------------------------------------------
    # 17. Volume surge detector
    # ------------------------------------------------------------------
    vol_sma_5 = talib.SMA(v, timeperiod=5)
    safe_vol_sma_5 = np.where((vol_sma_5 == 0) | np.isnan(vol_sma_5), 1, vol_sma_5)
    features["vol_surge"] = v / safe_vol_sma_5  # short-term volume spike

    # ------------------------------------------------------------------
    # 18. Volume z-score (24h participation proxy, no order-book data)
    # ------------------------------------------------------------------
    vol_mean_24 = talib.SMA(v, timeperiod=24)
    vol_std_24 = talib.STDDEV(v, timeperiod=24)
    safe_vol_std = np.where((vol_std_24 == 0) | np.isnan(vol_std_24), 1.0, vol_std_24)
    features["volume_zscore_24"] = (v - vol_mean_24) / safe_vol_std

    # ------------------------------------------------------------------
    # 19. ATR ratio 14 vs 48 — shorter horizon compression vs 14/56
    # ------------------------------------------------------------------
    atr_48 = talib.ATR(h, lo, c, timeperiod=48)
    safe_atr_48 = np.where((atr_48 == 0) | np.isnan(atr_48), 1.0, atr_48)
    features["atr_ratio_14_48"] = atr_14 / safe_atr_48

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
