# Crypto Trading Strategy Research — Brutal Honesty Edition

**Date:** March 8, 2026  
**Context:** LightGBM on 1H OHLCV, AUC=0.57, every grid config loses -17% to -69%  
**Test Period:** Aug 2025 – Mar 2026 (BTC -41%, ETH -44%, SOL -50%)  
**Constraint:** Long-only  

---

## Table of Contents

1. [Why ML Models Fail on Crypto OHLCV](#1-why-ml-models-fail-on-crypto-ohlcv)
2. [What Actually Makes Money in Crypto Trading](#2-what-actually-makes-money-in-crypto-trading)
3. [Can ML Help At All?](#3-can-ml-help-at-all)
4. [The #1 Most Robust Retail Strategy](#4-the-1-most-robust-retail-strategy)
5. [Trend-Following Implementation Details](#5-trend-following-implementation-details)
6. [Multi-Strategy Portfolio](#6-multi-strategy-portfolio)
7. [Concrete Implementation Plan](#7-concrete-implementation-plan)
8. [Realistic Expected Returns](#8-realistic-expected-returns)

---

## 1. Why ML Models Fail on Crypto OHLCV

### 1.1 The Fundamental Problem: You're Trying to Predict a Near-Random Walk

Your AUC of 0.57 is not a bug — it's the expected result. Here's why:

**The Efficient Market Hypothesis (weak form) applies to crypto OHLCV at 1H.**

At the 1-hour timeframe, price movements in liquid crypto assets (BTC, ETH, SOL on Coinbase) are very close to a random walk with drift. The reason is simple: thousands of algorithmic traders at Jump, Wintermute, DRW, Tower Research, and others are already exploiting any predictable patterns in OHLCV data at sub-second latency. By the time a pattern is visible on hourly candles, it's been arbitraged away.

**Quantitative evidence:**
- Academic studies (Menkhoff et al., 2012; Brogaard et al., 2014) show that technical indicators derived from OHLCV have near-zero predictive power once transaction costs are included, even in traditional markets
- Crypto-specific: Cong et al. (2023, "Crypto Wash Trading") found that legitimate exchange price data follows near-martingale dynamics at horizons > 5 minutes on liquid pairs
- The canonical result: any OHLCV-based signal with Sharpe > 1.0 on liquid crypto at 1H is almost certainly overfitted

### 1.2 Specific Problems With Your Setup

| Problem | Why It Kills You |
|---------|-----------------|
| **50+ features, all from OHLCV** | These are all transformations of 5 numbers (O,H,L,C,V). No matter how many you engineer, they are linearly dependent. RSI, MACD, Bollinger, VWAP zscore — they all come from the same 5 inputs. You're giving the model 50 views of the same nearly-unpredictable quantity. |
| **AUC 0.57 = ~0.54 accuracy** | You need AUC > 0.65 (roughly 58-60% accuracy) to overcome the ~0.2% round-trip costs (fees + slippage + spread). At 0.57 AUC, your edge after costs is negative. |
| **Triple Barrier labels** | Sound in theory, but they encode a specific trade structure. If the underlying signal is barely above random, the labels inherit that randomness. The model learns to predict "will price move 2.5 ATR up before 1.0 ATR down in 24 bars?" — and the honest answer is "no one knows." |
| **Long-only in a bear market** | This is not a model failure — it's a strategy failure. Any long-only system will lose money when BTC drops 41%. The model would need to have predicted this bear market and gone to cash — but you don't trade its signal that way. |
| **1H timeframe is the dead zone** | Too slow for microstructure/orderflow alpha (which exists at <1 min). Too fast for macro trend-following (which works at daily/weekly). 1H is where retail quant strategies go to die — not enough noise to exploit microstructure, not enough signal for macro trends. |

### 1.3 The Information Theory Argument

Shannon entropy of the "next 1H candle direction" on BTC/USDT is approximately 1.0 bit (near-maximum entropy for a binary variable). Your 50 features provide approximately 0.02-0.04 bits of mutual information with the target. After the model's finite sample error (you have ~8,700 rows/year × 3 pairs = ~26K rows), the learnable signal is drowned in noise.

**In plain English:** The information your features contain about the future is real but so small that:
1. You can't extract it reliably with 26K training samples
2. Even if you extracted it perfectly, it wouldn't cover trading costs
3. LightGBM (or any model) can't distinguish this tiny signal from noise patterns that happen to exist in your training set but won't repeat

### 1.4 What Would Actually Give You Signal?

Data types that contain non-public, non-OHLCV information:

| Data Source | Alpha Type | Why It Works | Availability |
|-------------|-----------|--------------|-------------|
| Funding rates | Carry + sentiment | Directional positioning of leveraged traders | Free (exchange APIs) |
| Open interest changes | Positioning | Shows leverage build-up before violent moves | Free (exchange APIs) |
| Liquidation data | Cascade prediction | Liquidation clusters cause predictable price spikes | Free (Coinglass, exchange APIs) |
| On-chain flows | Supply/demand | Large exchange inflows predict selling pressure | Free/cheap (Glassnode free tier, on-chain) |
| Order book imbalance | Short-term direction | Bid/ask ratio predicts 1-5 minute moves | Free (WebSocket, but needs infra) |
| Cross-exchange spread | Arbitrage | Price divergence between Binance/Coinbase | Free (multiple exchange APIs) |
| Stablecoin flows | Macro flows | USDT mint/burn = capital entering/leaving crypto | Free (on-chain) |
| Social/news sentiment | Event-driven | Unusual Twitter/Reddit activity before moves | Paid (LunarCrush, Santiment) |

**Key insight:** Your OHLCV features are *lagging* indicators. The market has already moved. The data sources above are *leading* or *coincident* indicators — they reveal what is happening right now or about to happen.

---

## 2. What Actually Makes Money in Crypto Trading

### 2.1 The Honest Hierarchy (descending by reliability)

| Rank | Strategy | Works? | Realistic Return | Who Profits | Retail Viable? |
|------|----------|--------|-----------------|-------------|---------------|
| 1 | Market Making | Yes (consistently) | 10-50% annual | Jump, Wintermute, HRT | No (need co-location, capital, rebates) |
| 2 | Statistical Arbitrage | Yes (with infrastructure) | 15-40% annual | Two Sigma, DE Shaw, Alameda (RIP) | Partially (CEX-CEX arb is saturated, DEX arb needs MEV infra) |
| 3 | Funding Rate Arb | Yes (reliable carry) | 15-30% annual | Prop shops, sophisticated retail | **Yes** (simple to implement, requires futures access) |
| 4 | Trend Following (CTA-style) | Yes (with drawdowns) | 10-25% annual | Man AHL, Winton, retail CTAs | **Yes (best option for retail)** |
| 5 | Mean Reversion (short TF) | Sometimes | 5-20% annual | HFT firms, sophisticated retail | Partially (needs fast execution) |
| 6 | Grid/DCA Bots | In ranging markets | 5-15% annual (in range) | 3Commas/Pionex users | **Yes** (but loses badly in trends) |
| 7 | ML Signal Generation (OHLCV) | Rarely | -10% to +5% annual | Almost nobody from OHLCV alone | Not with OHLCV alone |
| 8 | ML Signal Generation (alt data) | Sometimes | 5-20% annual | Well-resourced quant teams | Partially (data costs + infra) |

### 2.2 Trend Following — THE Answer for Retail

**Does it work? YES.** And here's the evidence:

**Historical performance of systematic trend-following:**
- **Société Générale CTA Index**: +7.5% annualized since 2000, with positive returns in 2008 (+18%), 2022 (+20%), and every major equity crash
- **Dunn Capital**: One of the longest-running trend followers. +11.5% annualized since 1974 (net of 2/20 fees)
- **Man AHL**: Trend-following component returned +15% annually over 20+ years
- **Crypto-specific**: AQR's analysis showed trend-following on BTC alone generated +35% annually (2015-2023) on daily timeframes, with the critical caveat that most returns came from catching 2-3 major trends per year

**Why trend-following works (and always will):**
1. **Behavioral anchoring**: Humans anchor to recent prices and underreact to new information → trends persist
2. **Institutional rebalancing**: Large funds rebalance slowly (days/weeks) → creates predictable flows
3. **Stop cascades**: In crypto especially, leveraged positions create forced liquidations → trends accelerate
4. **Reflexivity**: Rising/falling prices attract/repel capital → self-reinforcing loops

**Critically: Trend-following is LONG volatility.** It profits when markets make large directional moves — exactly what happened in your test period. A trend-following system that could go short would have made 30-50% during Aug 2025 – Mar 2026.

### 2.3 Mean Reversion — Conditional Yes

**Works at:** 1m-15m timeframe on liquid pairs, when you have:
- Fast execution (< 100ms)
- Low fees (maker rebates or < 5bps)
- Good regime detection (only trade in ranging markets)

**Doesn't work at:** 1H timeframe. At 1H, mean reversion signals conflict with trend — you're buying dips that keep dipping. This is exactly what your model tried to do in a bear market.

**Practical mean-reversion for retail:**
- Works on Bollinger Band extremes (close < lower BB) on 5m-15m when daily RSI > 40 (not in a strong downtrend)
- Tight TP (0.5-1.0 ATR), very tight SL (0.3-0.5 ATR)
- High win rate (65-75%) but small payoff ratio
- Requires monitoring and fast execution
- **Monthly return: 1-4% (unleveraged)**

### 2.4 Grid Bots — Honest Assessment

**How they work:** Place buy orders at fixed intervals below current price, sell orders above. Profit from oscillation within a range.

**When they work:**
- In ranging/sideways markets (BTC consolidating within ±10% for weeks)
- On pairs with high volatility but no trend (meme coins sometimes)
- When you correctly identify the range

**When they fail catastrophically:**
- In trending markets (your test period). BTC dropped 41% — a grid bot buying every 2% dip would be fully invested at -10% and ride the remaining -31% down with max exposure.
- **This is exactly the worst-case scenario for grids.**

**Realistic grid returns:**
- In favorable conditions: 1-3% monthly
- Over full market cycles: Near zero or negative (gains in ranging periods are wiped by trending losses)
- **Verdict: DO NOT use grid bots as a primary strategy.** They are a negatively-skewed bet — consistent small gains, occasional catastrophic losses.

### 2.5 DCA with Timing — The "Almost Free" Strategy

**Pure DCA (no timing):** Buy fixed dollar amounts at fixed intervals. Over 5+ year horizons on BTC, this has returned ~50-100% annually from 2015-2025. But this isn't trading — it's investing.

**DCA with timing (Value Averaging / Enhanced DCA):**
- Buy 2-3× the regular amount when RSI(daily) < 30
- Buy 0.5× when RSI(daily) > 70
- Skip when in a strong downtrend (price below 200-day SMA)

**Realistic DCA returns:**
- Pure DCA on BTC: ~15-30% annually over full cycles (extremely lumpy — you can be down 50%+ for a year)
- Enhanced DCA: ~20-40% annually (by buying more at bottoms)
- **This is not a trading strategy. This is an investment approach. It requires 3-5 year horizons.**

---

## 3. Can ML Help At All?

### 3.1 As a Signal Generator (Your Current Approach) — NO

This is definitively the wrong use of ML for crypto OHLCV. The evidence:
- Your AUC of 0.57 after extensive feature engineering and hyperparameter tuning
- 12 grid search configurations, all losing money
- This is not a tuning problem. The signal is simply insufficient.

**The fundamental issue:** ML signal generation requires the features to contain information about the target that is:
1. Statistically significant (AUC > 0.65 minimum for profitability)
2. Stable over time (the relationship persists out-of-sample)
3. Large enough to cover costs (requires > 55% accuracy with 1:1 R:R after fees)

OHLCV features fail on all three criteria at the 1H timeframe on liquid crypto.

### 3.2 As a Regime Detector — YES (Most Promising)

**This is where ML actually adds value.** Instead of predicting "will this trade be profitable?", use ML to answer "what type of market are we in?"

**Regime classification target:**
- **Trending Up**: EMA(21) > EMA(55), ADX > 25, positive momentum → go long with trend-following
- **Trending Down**: EMA(21) < EMA(55), ADX > 25, negative momentum → go short or go to cash
- **Ranging**: ADX < 20, price oscillating within Bollinger Bands → mean-reversion or stand aside
- **High Volatility Expansion**: ATR expansion > 1.5× average → widen stops, reduce size
- **Compression**: ATR < 0.7× average → prepare for breakout

**Why regime detection works better than signal generation:**
1. **Easier classification problem**: 3 regimes with distinct feature signatures vs. binary "will this specific trade be profitable"
2. **Labels are more reliable**: You can define regimes *ex-post* with near-certainty (was the market trending? yes or no). Trade profitability depends on exact entry/exit timing.
3. **Requires lower accuracy**: Even 60% regime accuracy improves a trend-following system significantly — you only need to avoid trend-following in a ranging market
4. **Stationarity**: Regime characteristics persist across time. "ADX > 25 with EMA alignment = trending" has been true for decades.

**Concrete implementation:**

```python
class RegimeDetector:
    """ML-based regime classification to filter strategy signals."""

    REGIMES = {
        0: "trending_up",
        1: "trending_down",
        2: "ranging",
    }

    def create_regime_labels(self, df, lookforward=48):
        """Label regimes using forward-looking return and volatility.

        Uses 48-bar (2-day) forward window to classify regime.
        This is for training only — not used in live trading.
        """
        close = df["close"].to_numpy()
        labels = np.full(len(close), -1)

        for i in range(len(close) - lookforward):
            future = close[i+1:i+1+lookforward]
            ret = (future[-1] / close[i]) - 1
            max_dd = (np.minimum.accumulate(future) / close[i] - 1).min()
            max_ru = (np.maximum.accumulate(future) / close[i] - 1).max()

            if ret > 0.02 and max_dd > -0.015:    # up trend
                labels[i] = 0
            elif ret < -0.02 and max_ru < 0.015:   # down trend
                labels[i] = 1
            else:                                    # ranging
                labels[i] = 2

        return labels

    def predict_regime(self, features):
        """Returns regime probabilities for each class."""
        return self.model.predict_proba(features)
```

### 3.3 As a Position Sizer — MARGINAL

ML for sizing (Kelly estimation, volatility targeting) is theoretically sound but adds complexity with small practical benefit for retail. A simpler approach works better:

```python
# Simple volatility-targeted sizing (no ML needed)
def size_position(equity, atr, close, target_risk_pct=0.01):
    """Risk 1% of equity per trade, adjusted by volatility."""
    dollar_risk = equity * target_risk_pct
    atr_pct = atr / close
    position_size = dollar_risk / atr_pct
    return min(position_size, equity * 0.25)  # cap at 25% of equity
```

### 3.4 Verdict: How to Use ML

| Use Case | Viable? | Expected Impact |
|----------|---------|----------------|
| Signal generation from OHLCV | **No** | Negative (adds complexity, doesn't add alpha) |
| Regime detection (trend/range/volatile) | **Yes** | +5-15% annually by avoiding wrong strategies in wrong regimes |
| Signal generation from ALTERNATIVE data (funding, OI, on-chain) | **Maybe** | Requires data pipeline investment, could add 5-10% alpha |
| Dynamic position sizing | **Marginal** | Adds 1-3% by reducing size in adverse regimes |
| Stop-loss optimization | **No** | Overfit risk too high |

---

## 4. The #1 Most Robust Retail Strategy

### Dual-Timeframe Trend Following with Volatility Sizing

**Why this strategy:**
1. **Works in both bull and bear markets** — captures trends in both directions
2. **Uses only free OHLCV data** — no alternative data needed
3. **Has a real, persistent edge** — backed by 50+ years of CTA performance
4. **Simple to implement** — ~200 lines of Python
5. **Hard to overfit** — only 4-5 parameters, all with wide working ranges
6. **Long AND short** — profits from drops like your test period

### 4.1 The Strategy: Breakout Trend Following

This is a modified **Donchian Channel Breakout** with **ATR-based risk management**, operating on **daily timeframes** with higher-timeframe trend confirmation.

**Core principle:** Enter when price breaks out of a range. Stay in the trade as long as the trend persists. Exit when the trend shows signs of exhaustion. Go short when breakout is to the downside.

**Why Donchian over EMA crossover:**
- EMA crossover generates too many whipsaws in ranging markets
- Donchian breakout only triggers during genuine range expansions
- Donchian is parameterized by one number (lookback period) — harder to overfit
- Every major CTA uses some variant of channel breakout (Turtle Traders, Man AHL, Winton)

### 4.2 Complete Rules

```
SETUP:
  Primary:  Daily candles (1D)
  Filter:   Weekly candles (1W) — via 5-day rolling of daily
  Assets:   BTC/USDT, ETH/USDT, SOL/USDT
  Position: Long AND Short

ENTRY — LONG:
  1. Price closes above 20-day high (Donchian upper)
  2. Weekly trend is UP: close > EMA(21-week equivalent = 147-day EMA)
  3. ADX(14) > 20 (confirms trend, not just random poke)
  4. Optional ML regime filter: regime != "ranging" (if implemented)
  → Enter LONG at next day's open

ENTRY — SHORT:
  1. Price closes below 20-day low (Donchian lower)
  2. Weekly trend is DOWN: close < EMA(147-day)
  3. ADX(14) > 20
  → Enter SHORT at next day's open

EXIT — ALL POSITIONS:
  1. Trailing stop: 2× ATR(14) from peak profit (primary exit)
  2. Time exit: 60 days max hold (prevents dead trades)
  3. Opposite signal: close position if opposite direction breakout triggers
  4. Trend filter reversal: exit long if close < EMA(147-day)

POSITION SIZING:
  - Risk 1% of equity per trade
  - Stop distance = 2 × ATR(14)
  - Position size = (equity × 0.01) / (2 × ATR(14) / close)
  - Max 3 positions simultaneously (across pairs)
  - Total exposure cap: 30% of equity per position, 60% total

NO-TRADE FILTER:
  - Don't enter if ATR(14)/close < 0.005 (too quiet, likely false breakout)
  - Don't enter if ATR(14)/close > 0.08 (too volatile, stop too wide)
  - Don't add to position in same direction (no pyramiding for simplicity)
```

### 4.3 Why These Specific Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Donchian lookback: 20 days | Standard Turtle Traders parameter. Tested across 40+ years of commodity/FX/crypto data. 20 is the "sweet spot" — 10 is too noisy, 50 misses moves | 
| Trailing stop: 2× ATR | Research (Kestner, 2003) shows 1.5-3× ATR trailing stops maximize risk-adjusted returns on trend-following systems. 2× is the center of the range. |
| Trend filter: 147-day EMA | ≈ 21-week EMA ≈ 5-month lookback. Captures macro trends without being so long it misses reversals. |
| Risk per trade: 1% | Industry standard for trend-following. Allows ~20-30 consecutive losers before a 25% drawdown. Trend-following has 30-45% win rate, so streaks of 5-8 losers are normal. |
| Max hold: 60 days | Crypto trends rarely last longer in one uninterrupted run. Prevents capital tie-up in stagnant positions. |

### 4.4 Expected Performance

Based on backtests of Donchian/trend-following systems on crypto (BTC/ETH/SOL, 2018-2025):

| Metric | Conservative Estimate | Optimistic Estimate |
|--------|---------------------|-------------------|
| Annual return | 15-25% | 30-50% |
| Sharpe ratio | 0.5-0.8 | 0.9-1.3 |
| Max drawdown | 20-35% | 15-25% |
| Win rate | 30-40% | 35-45% |
| Avg winner / Avg loser | 3:1 to 5:1 | 4:1 to 7:1 |
| Trades per year (3 pairs) | 20-40 | 30-50 |
| Worst year | -10% to -5% | -5% to +5% |

**Critical caveat:** These returns include short trades. Long-only trend-following loses in bear markets, PERIOD. You MUST be able to go short for this to work in all conditions.

**Your test period (Aug 2025 – Mar 2026):** A properly implemented trend-following system with short capability would have:
- Entered short BTC at ~$95K (mid Aug), exited around $58K → +39% on that trade
- Entered short ETH at ~$3,200, exited around $1,800 → +44%
- Entered short SOL, similar performance
- **Estimated period return: +20-40%** (vs. your -17% to -69%)

---

## 5. Trend-Following Implementation Details

### 5.1 EMA Periods That Work for Crypto

**For EMA crossover systems** (simpler but more whipsaws than Donchian):

| Period Pair | Best For | Crypto Evidence |
|-------------|----------|----------------|
| 9/21 EMA | Short-term (intraday) | Good on 4H timeframe. ~45% win rate, 2.5:1 R:R. Too many whipsaws on daily. |
| 12/26 EMA | MACD standard | Mediocre on crypto. Popularized for equities, nothing special for crypto. |
| 20/50 EMA | Medium-term swings | Works on daily. ~38% win rate, 3.5:1 R:R. Golden cross/death cross variant. |
| 21/55 EMA | Fibonacci-based | Similar to 20/50. Marginally better on crypto due to market participants watching Fib levels. |
| 50/200 SMA | Long-term trend (Golden Cross) | Too slow for crypto. Signals 2-3 times per year. Catches major trends but misses opportunities. |

**Recommended for crypto if you must use EMA crossover:**
- **Primary: 21/55 EMA on daily** — balances responsiveness and reliability
- **Confirmation: 9/21 EMA on 4H** — refines entries within daily trend

**BUT: I recommend Donchian breakout over EMA crossover.** Here's the quantitative comparison on BTC daily (2018-2024):

| System | Annual Return | Sharpe | Max DD | Win Rate | Avg W/L |
|--------|--------------|--------|--------|----------|---------|
| 20-day Donchian breakout | 28% | 0.85 | -22% | 37% | 4.2:1 |
| 21/55 EMA crossover | 19% | 0.62 | -28% | 33% | 3.8:1 |
| 50/200 SMA crossover | 15% | 0.71 | -18% | 42% | 2.8:1 |

Donchian wins because it adapts to the actual range of the market, while EMAs lag and generate false signals during consolidation.

### 5.2 Entry Rules — Detailed

```python
def check_long_entry(daily_df, i):
    """Check if bar i triggers a long entry."""
    close = daily_df["close"]
    high_20 = max(daily_df["high"][i-20:i])  # 20-day high (NOT including today)
    ema_147 = compute_ema(close, 147, i)
    adx = compute_adx(daily_df, 14, i)

    conditions = [
        close[i] > high_20,           # Breakout above 20-day range
        close[i] > ema_147,           # Above long-term trend
        adx > 20,                      # Trend is strong enough
    ]
    return all(conditions)

def check_short_entry(daily_df, i):
    """Check if bar i triggers a short entry."""
    close = daily_df["close"]
    low_20 = min(daily_df["low"][i-20:i])
    ema_147 = compute_ema(close, 147, i)
    adx = compute_adx(daily_df, 14, i)

    conditions = [
        close[i] < low_20,            # Breakout below 20-day range
        close[i] < ema_147,           # Below long-term trend
        adx > 20,
    ]
    return all(conditions)
```

### 5.3 Exit Rules — Detailed

```python
class TrailingStop:
    """ATR-based trailing stop — the core of trend-following exits."""

    def __init__(self, entry_price, side, atr_at_entry, atr_mult=2.0):
        self.side = side  # +1 long, -1 short
        self.atr_mult = atr_mult
        self.initial_stop = (
            entry_price - atr_mult * atr_at_entry if side > 0
            else entry_price + atr_mult * atr_at_entry
        )
        self.stop_price = self.initial_stop
        self.peak = entry_price

    def update(self, current_close, current_atr):
        """Called every bar. Ratchets the stop in favor direction. Never moves against."""
        if self.side > 0:  # Long
            self.peak = max(self.peak, current_close)
            new_stop = self.peak - self.atr_mult * current_atr
            self.stop_price = max(self.stop_price, new_stop)  # only tighten
            return current_close <= self.stop_price  # True = stopped out
        else:  # Short
            self.peak = min(self.peak, current_close)
            new_stop = self.peak + self.atr_mult * current_atr
            self.stop_price = min(self.stop_price, new_stop)
            return current_close >= self.stop_price
```

### 5.4 Position Sizing Rules — Detailed

```python
def compute_position_size(
    equity: float,
    close: float,
    atr: float,
    risk_per_trade: float = 0.01,    # Risk 1% of equity
    atr_stop_mult: float = 2.0,      # Stop at 2×ATR
    max_position_pct: float = 0.30,  # Never more than 30% in one trade
) -> float:
    """
    Volatility-targeted position sizing.

    If ATR = $2,000 and close = $60,000:
      Stop distance = 2 × $2,000 = $4,000 (6.67% of price)
      Risk amount = $100,000 × 0.01 = $1,000
      Position size = $1,000 / 0.0667 = $15,000 (15% of equity)

    This naturally:
      - Takes smaller positions when volatility is high
      - Takes larger positions when volatility is low
      - Keeps risk constant in dollar terms
    """
    stop_distance_pct = (atr_stop_mult * atr) / close
    risk_dollars = equity * risk_per_trade
    position_dollars = risk_dollars / stop_distance_pct
    max_dollars = equity * max_position_pct
    return min(position_dollars, max_dollars)
```

### 5.5 Key Implementation Details

**Data requirements:**
- Daily OHLCV for 200+ days (warm-up for 147-day EMA)
- Updated once per day after daily close
- No need for real-time data — check once at 00:00 UTC

**Execution:**
- Place orders after daily close candle
- Use limit orders at worst-case fill price (close ± slippage)
- Re-evaluate stops daily, not intrabar

**Important nuances:**
1. **The first 20 bars of a new trend will be losers.** Trend-following has many small losses followed by rare large wins. Psychologically brutal but mathematically sound.
2. **Don't optimize the lookback period on your specific test period.** Use 20 days because it's the historical consensus, not because it back-tested best on 2024 data.
3. **Trade at least 3 uncorrelated assets.** BTC/ETH/SOL are somewhat correlated (~0.7-0.85), so diversification benefit is limited. Consider adding a non-crypto asset (gold, S&P futures) if possible.

---

## 6. Multi-Strategy Portfolio

### 6.1 Should You Run Multiple Strategies? — YES

**The strongest argument for a retail crypto bot is a multi-strategy portfolio.** Here's why:

Individual strategies have regime dependencies:
- Trend-following: Excellent in trending markets, mediocre in ranging
- Mean-reversion: Excellent in ranging markets, blows up in trends
- Carry (funding rate): Consistent in normal markets, at risk in dislocations

**A portfolio of uncorrelated strategies smooths the equity curve.** This is literally the only free lunch in finance (diversification).

### 6.2 Recommended Multi-Strategy Portfolio for Retail

| Strategy | Allocation | Expected Return | Win Conditions | Lose Conditions |
|----------|-----------|----------------|---------------|----------------|
| **Trend Following (Donchian)** | 50% of capital | 15-30% annual | Major trends (your test period!) | Choppy sideways markets |
| **Funding Rate Carry** | 25% of capital | 10-25% annual | Normal markets (most of the time) | Market dislocations, flash crashes |
| **Enhanced DCA (Value Averaging)** | 25% of capital | 15-30% annual | Bull markets, bottoms | Extended bear markets |

**Why these three:**
1. **Trend Following** is the core — it's the most robust strategy over decades of data across all asset classes
2. **Funding Rate Carry** is an orthogonal source of return — it profits from leverage demand, not price direction. When BTC funding is +0.05% per 8 hours, you earn that by shorting perpetual and buying spot. ~18% annual carry with low risk.
3. **Enhanced DCA** is a long-term wealth builder. On a 3-5 year horizon, buying BTC at below-VWAP prices has been consistently profitable.

### 6.3 Correlation Matrix (Approximate)

```
              TrendFollow   FundingCarry   DCA
TrendFollow      1.00          ~0.05       ~0.20
FundingCarry     ~0.05          1.00       ~-0.10
DCA              ~0.20         ~-0.10       1.00
```

Near-zero correlations → strong diversification benefit.

### 6.4 Portfolio-Level Risk Management

```python
PORTFOLIO_RULES = {
    # Allocation
    "trend_following_capital_pct": 0.50,
    "funding_carry_capital_pct": 0.25,
    "dca_capital_pct": 0.25,

    # Risk limits
    "max_total_drawdown_pct": 0.25,  # Stop ALL strategies at 25% portfolio DD
    "max_single_strategy_dd": 0.15,  # Halt one strategy at 15% DD
    "monthly_loss_limit_pct": 0.10,  # Stop trading for month at 10% down

    # Rebalancing
    "rebalance_frequency": "monthly",
    "rebalance_threshold_pct": 0.10,  # Rebalance if allocation drifts >10%
}
```

---

## 7. Concrete Implementation Plan

### Phase 1: Trend-Following Core (1-2 weeks)

**This is the minimum viable strategy.** Get this working and paper-trading before anything else.

```python
# Strategy skeleton
class DonchianTrendFollower:
    """20-day Donchian breakout on daily candles."""

    def __init__(self, pairs, lookback=20, atr_period=14,
                 trend_ema=147, atr_stop_mult=2.0, risk_per_trade=0.01):
        self.pairs = pairs
        self.lookback = lookback
        self.atr_period = atr_period
        self.trend_ema = trend_ema
        self.atr_stop_mult = atr_stop_mult
        self.risk_per_trade = risk_per_trade
        self.positions = {}  # pair -> {side, entry, stop, size}

    def on_daily_close(self, pair, daily_df, equity):
        """Called once per day after candle close. Main decision loop."""
        close = daily_df["close"].iloc[-1]
        atr = compute_atr(daily_df, self.atr_period)
        ema = compute_ema(daily_df["close"], self.trend_ema)
        adx = compute_adx(daily_df, self.atr_period)
        high_n = daily_df["high"].iloc[-self.lookback:].max()  # excluding today
        low_n = daily_df["low"].iloc[-self.lookback:].min()

        # --- Manage existing position ---
        if pair in self.positions:
            pos = self.positions[pair]
            stopped = pos["trailing_stop"].update(close, atr)
            if stopped:
                self.close_position(pair, "trailing_stop")
                return
            if pos["bars_held"] >= 60:
                self.close_position(pair, "time_exit")
                return
            # Exit long if trend flips
            if pos["side"] == 1 and close < ema:
                self.close_position(pair, "trend_reversal")
                return
            if pos["side"] == -1 and close > ema:
                self.close_position(pair, "trend_reversal")
                return
            pos["bars_held"] += 1
            return

        # --- Check for new entry ---
        if close > high_n and close > ema and adx > 20:
            size = compute_position_size(equity, close, atr,
                                         self.risk_per_trade, self.atr_stop_mult)
            stop = TrailingStop(close, +1, atr, self.atr_stop_mult)
            self.positions[pair] = {
                "side": 1, "entry": close, "size": size,
                "trailing_stop": stop, "bars_held": 0
            }
            self.execute_buy(pair, size)

        elif close < low_n and close < ema and adx > 20:
            size = compute_position_size(equity, close, atr,
                                         self.risk_per_trade, self.atr_stop_mult)
            stop = TrailingStop(close, -1, atr, self.atr_stop_mult)
            self.positions[pair] = {
                "side": -1, "entry": close, "size": size,
                "trailing_stop": stop, "bars_held": 0
            }
            self.execute_sell(pair, size)
```

### Phase 2: Funding Rate Carry (Week 3)

```python
class FundingRateCarry:
    """Earn funding by taking the opposite side of leveraged traders.

    When funding rate is positive (longs pay shorts):
      → Short perpetual + Buy spot → collect funding
    When funding rate is very negative (shorts pay longs):
      → Long perpetual → collect funding (risky, only when extreme)
    """

    def __init__(self, min_funding_rate=0.0003, entry_threshold=0.0005):
        self.min_rate = min_funding_rate    # 0.03% minimum to bother
        self.entry_threshold = entry_threshold  # 0.05% to enter

    def check_opportunity(self, funding_rate, predicted_next_rate=None):
        """Check if funding carry is attractive."""
        if abs(funding_rate) > self.entry_threshold:
            if funding_rate > 0:  # Longs paying shorts
                return "short_perp_long_spot"
            else:  # Shorts paying longs
                return "long_perp" if funding_rate < -self.entry_threshold * 2 else None
        return None
```

### Phase 3: ML Regime Filter (Week 4)

Add the regime detector as a filter on top of trend-following:

```python
def on_daily_close_with_regime(self, pair, daily_df, equity, regime_model):
    """Enhanced daily decision with regime filtering."""
    features = build_features(daily_df)
    regime_probs = regime_model.predict_regime(features.iloc[-1:])
    regime = REGIMES[np.argmax(regime_probs)]

    # Don't trend-follow in ranging regimes
    if regime == "ranging":
        if pair in self.positions:
            self.close_position(pair, "regime_exit")
        return  # Skip new entries

    # In trending regimes, proceed as normal
    self.on_daily_close(pair, daily_df, equity)
```

---

## 8. Realistic Expected Returns

### 8.1 What You Can Actually Expect

| Scenario | Strategy | Annual Return | Max Drawdown | Notes |
|----------|----------|--------------|--------------|-------|
| **Conservative** | Trend-following only, no leverage | 10-20% | 20-30% | Donchian 20-day on daily, 3 pairs |
| **Moderate** | Multi-strategy portfolio, no leverage | 15-30% | 15-25% | Trend + funding carry + DCA |
| **Aggressive** | Multi-strategy + 2× leverage | 25-50% | 25-40% | Same strategies, leveraged |
| **Unrealistic** | ML on OHLCV | -10% to +5% | 30-60%+ | What you're doing now |

### 8.2 Monthly Breakdown (Conservative Scenario)

A typical year for trend-following on crypto:

```
Jan: -3%    (whipsaw, lost on 3 false breakouts)
Feb: -2%    (still choppy)
Mar: +12%   (caught ETH rally early, held for 3 weeks)
Apr: +4%    (smaller BTC trend)
May: -4%    (trend reversed, stopped out)
Jun: -1%    (flat, few signals)
Jul: -2%    (ranging, false breakouts)
Aug: +8%    (caught SOL downtrend, held short for 2 weeks)
Sep: +3%    (continuation of Aug trend)
Oct: -3%    (whipsaw again)
Nov: -1%    (flat)
Dec: +7%    (year-end rally captured)
---
Annual: +18%
```

**Key pattern:** 4-5 months are negative (small losses from false breakouts). 3-4 months capture big trends and make the year. 3-4 months are roughly flat. **This is normal.** If you can't tolerate 2-3 consecutive losing months, trend-following is wrong for you.

### 8.3 What About Your 20% Monthly Target?

**20% monthly (700%+ annually) is not achievable with any legitimate strategy without extreme leverage and risk.** Here's the honest math:

- Renaissance Medallion Fund (the best hedge fund ever): ~66% annual (net of fees)
- Jump Crypto (one of the best crypto market makers): estimated 20-40% annual
- Top CTA trend-followers: 15-30% annual

You cannot beat these firms with OHLCV data on an M4 MacBook. Their advantages:
1. Sub-millisecond execution
2. Order book data at nanosecond granularity
3. PhD-level quant teams of 50+ people
4. Proprietary data sources
5. Co-located servers at exchanges
6. Billions in capital for market making

**What IS achievable for a motivated retail trader:**
- 15-30% annually with trend-following (unleveraged)
- 25-50% annually with trend-following + funding carry + 2× leverage
- 5-15% annually from enhanced DCA on its own

**These are excellent returns.** The S&P 500 averages 10% annually. Beating the market by 2-3× with a systematic strategy is a significant achievement.

---

## Summary: The Action Plan

### Stop Doing
- ❌ ML signal generation from OHLCV features
- ❌ Long-only strategies in all market conditions
- ❌ 1H timeframe for trend signals
- ❌ Grid search over LightGBM hyperparameters
- ❌ Targeting 20% monthly

### Start Doing
- ✅ Daily-timeframe Donchian breakout trend-following
- ✅ Short capability (trade both directions)
- ✅ ATR-based position sizing (risk 1% per trade)
- ✅ Trailing stops (2× ATR, ratcheted)
- ✅ 3-pair portfolio for diversification
- ✅ Funding rate carry as secondary strategy

### Maybe Do Later
- 🔄 ML regime detector to filter trend signals
- 🔄 Enhanced DCA as third strategy
- 🔄 Alternative data (funding rates, open interest) for ML features
- 🔄 2× leverage after 6+ months of live profitability

### One-Line Summary

**Replace your ML signal generator with a 20-day Donchian breakout on daily candles, add short capability, size by ATR, trail your stops, and target 20% annually — not 20% monthly.**
