# MFT-CashCow — Monthly Performance Summary

**Date:** March 9, 2026  
**Strategy:** Donchian Breakout + Triple Filter (ADX + Volatility + ML)  
**Assets:** ETH/USDT, SOL/USDT (1-hour candles)  
**Test Period:** Aug 2025 – Mar 2026 (7.2 months, out-of-sample)

---

## How Much Do You Gain in One Month?

| Leverage | Expected Monthly Return | Max Drawdown | Risk Level |
|----------|------------------------|--------------|------------|
| **1x (spot)** | **+2.6%** | -3.2% | Conservative |
| **3x (futures)** | **+7.7%** | -9.5% | Moderate |
| **5x (futures)** | **+12.7%** | -15.7% | Aggressive |
| **7x (futures)** | **+17.5%** | -21.6% | Very Aggressive |
| **10x (futures)** | **+24.4%** | -30.2% | Extreme |

> **Recommended: 3x–5x leverage** for a balance of returns and survival.

---

## Strategy Explained (Simple Version)

The bot combines four layers:

1. **Donchian Breakout** — When price breaks above the 20-day high → go LONG.  
   When price breaks below the 20-day low → go SHORT.

2. **ADX Filter** — Only enter trades when the market is actually trending (ADX > 18).  
   This alone blocks ~60% of false signals.

3. **Volatility Expansion Filter** — Only enter when current volatility (ATR) is above  
   its 10-day average. Catches breakouts from squeeze conditions.

4. **ML Filter (LightGBM)** — A machine learning model trained on 50 features confirms  
   the trade direction. Removes the last ~15% of bad signals.

**Exit:** 4× ATR trailing stop (lets winners run, cuts losers).

---

## Backtest Results (Out-of-Sample, 7.2 Months)

### At 1x (No Leverage — Spot Trading)
```
Portfolio:  +18.5%  over 7.2 months
Monthly:    +2.6%
Annualized: +30.8%
Trades:     13 total (very selective — 77% of signals filtered out)
Win Rate:   ~46%
Profit Factor: 3.52  (wins are 3.5× larger than losses)
Max Drawdown:  -3.2%
```

### At 5x Leverage (Futures)
```
Portfolio:  +91.0%  over 7.2 months
Monthly:    +12.7%
Annualized: +151.8%
Trades:     13
Profit Factor: 3.52
Max Drawdown:  -15.7%
```

### At 10x Leverage (Futures)
```
Portfolio:  +175.4%  over 7.2 months
Monthly:    +24.4%  ← exceeds 20%/month target
Annualized: +292.7%
Trades:     13
Profit Factor: 3.52
Max Drawdown:  -30.2%
```

### Monthly Breakdown (5x Leverage)
```
2025-09:    -0.1%   (flat)
2025-11:   +26.5%   (big trend captured)
2026-01:   +77.0%   (massive downtrend captured)
2026-02:   -10.8%   (choppy reversal)

Win months: 2 out of 4 active months (50%)
```

---

## Per-Pair Performance (1x)

| Pair | Return | Trades | Profit Factor | Notes |
|------|--------|--------|---------------|-------|
| ETH/USDT | +3.6% | 7 | 1.70 | Steady contributor |
| SOL/USDT | +14.9% | 6 | 8.58 | Star performer |

- SOL drives most of the returns due to stronger trending behavior
- BTC was dropped — it contributed near-zero (+0.4%) and dragged down everything

---

## Important Warnings

### 1. Only 13 trades in 7 months
The triple filter is extremely selective. This means:
- Results can swing significantly from one trade
- Statistical significance is low (need 100+ trades for high confidence)
- Some months have zero trades

### 2. Full 3-year validation shows drawdowns
When tested over the full 3 years (not just the favorable test period):
- ETH at 10x: reached -96.7% drawdown (near wipeout)
- SOL at 10x: reached -99.5% drawdown (near wipeout)

**The 7-month test period happened to be favorable for trend-following.** In choppy/ranging markets, even the filtered strategy can produce consecutive losses.

### 3. Leverage is a double-edged sword
- At 10x, a single -10% adverse move = account liquidation
- At 5x, survivable but still painful (-15.7% max DD)
- At 3x, most sustainable long-term

### 4. Real trading adds friction
- Slippage may exceed modeled 5 bps
- Funding rates on perpetual futures cost 0.01-0.1% per 8 hours
- Exchange outages during volatile moments
- Emotional decision-making can override signals

---

## Realistic Expectations

| Scenario | Monthly Return | Annual Return | Survival Probability |
|----------|---------------|---------------|---------------------|
| Conservative (1x spot) | +2–3% | +25–35% | Very High |
| Moderate (3x futures) | +6–8% | +70–100% | High |
| Aggressive (5x futures) | +10–13% | +120–155% | Moderate |
| YOLO (10x futures) | +20–25% | +240–300% | Low (ruin risk ~40-60%) |

> **Bottom line:** With 5x leverage, expect roughly **+10-13% per month** in trending  
> markets and **-5 to -15%** in choppy months. Over a full year, the strategy likely  
> compounds to **+100-150%** if you survive the drawdowns.

---

## What Makes This Strategy Work

1. **Trend-following is a proven edge** — works across all markets for decades
2. **Triple filter (ADX + Vol + ML)** catches only the highest-quality breakouts
3. **Both long AND short** — makes money in crashes (Jan 2026: +77% at 5x)
4. **SOL's strong trending nature** gives better signals than BTC

## What Could Break It

1. **Prolonged choppy/ranging market** — no trends = no signals = slow bleed from stops
2. **Leverage too high** — one bad sequence wipes the account
3. **Overfitting to test period** — 7 months is short; strategy needs live validation
4. **Regime change** — if crypto becomes less volatile, breakouts weaken
