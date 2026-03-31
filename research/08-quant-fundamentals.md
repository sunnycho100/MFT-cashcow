# Quant Fundamentals for V3

Research date: 2026-03-30

## Executive Summary

If the goal is to move from the current robust `v3` result toward a much higher-return system, the research does **not** support "just add more ML" as the main answer.

The highest-confidence path is:
- keep trend / regime structure as the execution backbone
- add stronger market-structure data
- use derivatives-state as a positioning layer
- treat attention and sentiment as secondary context
- optimize for return **with** drawdown and live-execution constraints, not raw backtest return alone

My current inference:
- `10%` monthly unlevered is not a realistic research-backed base target for a robust BTC/ETH/SOL hourly system
- `3%` to `5%` monthly unlevered is already ambitious
- `6%` to `10%` monthly with disciplined `2x` leverage is a more credible stretch target than `10%` monthly unlevered

That inference is based on the evidence below plus our own stricter walk-forward results.

## 1. Trend and Momentum Are Real in Crypto

High-confidence evidence:
- Liu and Tsyvinski show a strong time-series momentum effect in crypto and find that crypto returns are predicted by crypto-specific factors rather than standard stock / macro factors.
- In their Bitcoin weekly grouping, the top momentum quintile earned `11.22%` per week with Sharpe `0.45`, versus `2.60%` per week with Sharpe `0.19` for the bottom quintile.
- They also report out-of-sample momentum effects and attention effects, which is exactly why `v2` and `v3` work better when trend logic is the backbone instead of pure classification.

Implication for us:
- the system should remain trend-first
- ML should confirm or size trend trades, not replace them

## 2. Market Microstructure Matters More Than Fancy Model Architecture

High-confidence evidence:
- Albers et al. use fragmented order book and trade data across Bitcoin venues and show that microstructure features explain between `10%` and `37%` of the variation in `500ms` future returns, depending on venue.
- They also test maker and taker strategies and explicitly account for trading realities such as fees and execution style.
- Lim and Gorse show that order-flow-based modelling can remain temporally stable even through the 2017 Bitcoin bubble regime shift.

Implication for us:
- if we want materially better returns, order book, trade flow, imbalance, spread, and liquidity regime are more promising than "another feature stack on candles"
- this is especially true if we want to improve shorts, which usually require more precise entry timing than longs

## 3. Funding and Basis Are Not Just Features, They Proxy Crowd Positioning

High-confidence evidence:
- The Yale / Cowles paper on leverage and stablecoin pegs argues that perpetual futures funding rates are a useful proxy for speculative demand and expected crypto returns.
- Their appendix reports Bitcoin futures-implied expected returns averaging `5.0%` from December 2017 to November 2022, ranging from `-10.8%` to `23.5%`.
- They also show that funding rates are strongly related to stablecoin lending rates and are highly correlated across venues and contracts.

Implication for us:
- funding should stay in the model
- funding is not a standalone trade signal
- funding works best as a crowding / leverage / positioning input combined with trend and breakout context

## 4. Attention Helps More Than Raw Social Sentiment

Evidence is mixed and horizon-dependent:
- Liu and Tsyvinski find that investor attention predicts future returns. For Bitcoin, a one-standard-deviation increase in Google searches leads to `1.84%` and `2.30%` increases in 1-week and 2-week ahead returns; higher Twitter post counts also predict higher future returns.
- But another paper using Twitter and Reddit sentiment with a LASSO-VAR approach reports **no causality from social media sentiment to cryptocurrency returns**.

Implication for us:
- "people are talking about BTC more" is not enough as a primary signal
- attention proxies may be useful as a higher-level regime or participation filter
- social sentiment should be researched later, not elevated above order flow, funding, or trend structure

## 5. Risk Management and Costs Are Part of the Alpha

High-confidence evidence:
- AQR's managed-futures commentary emphasizes that trend-following performs best in persistent trends and struggles in sharp reversals / chop.
- The same discussion cites research placing transaction costs for sophisticated managed-futures managers in the order of `1%` to `4%` per year.
- Kraken's trading docs still make live safety design mandatory: validate-safe order flow, authenticated websocket tokens, and the dead-man switch / cancel-all-after mechanism.

Implication for us:
- a strategy that looks amazing before cost and stale-data controls is not ready
- "return-max" work still needs drawdown and execution penalties
- we should optimize with explicit caps, not maximize gross return blindly

## 6. What This Means for Our Current System

Best interpretation of the evidence:
- `v2` had higher upside in looser and more aggressive tests
- `v3` currently has a lower but more believable result because the validation is stricter
- the next jump in returns probably comes from:
  - porting the strongest `v2` aggressive logic into `v3`
  - adding market-structure data
  - optimizing for return under a drawdown cap
  - using `2x` leverage only after the unlevered core is stronger

## 7. Data Priority Ranking for a Higher-Return V3

Priority 1:
- Kraken order book
- Kraken trades
- spread, depth, imbalance, sweep and liquidity features

Priority 2:
- Deribit funding
- open interest
- futures basis / premium

Priority 3:
- cross-exchange premium and relative strength
- Coinbase / Kraken / other venue divergence

Priority 4:
- liquidation / volatility regime signals
- volatility scaling and adaptive sizing

Priority 5:
- attention and social overlays
- Polymarket event overlays
- Reddit / X / news sentiment

## 8. Bottom Line

Research-backed conclusion:
- do **not** abandon the hybrid path
- do **not** prioritize generic sentiment before market structure
- do **not** expect a robust `10%` monthly unlevered result from the current architecture

Best next research-backed development goal:
- build a return-max `v3` branch targeting roughly `3%` to `5%` unlevered monthly in robust testing
- then evaluate whether `2x` leverage gets us close to the `10%` monthly target without unacceptable drawdown

## Sources

- NBER, *Risks and Returns of Cryptocurrency*: https://www.nber.org/papers/w24877 and https://www.nber.org/papers/w24877.pdf
- arXiv, *Fragmentation, Price Formation, and Cross-Impact in Bitcoin Markets*: https://arxiv.org/abs/2108.09750
- arXiv, *Deep Recurrent Modelling of Stationary Bitcoin Price Formation Using the Order Flow*: https://arxiv.org/abs/2004.01499
- Cowles / Yale, *Leverage and Stablecoin Pegs*: https://cowles.yale.edu/sites/default/files/2023-04/Leverage%20and%20Stablecoin%20Pegs_April%202023.pdf
- ScienceDirect, *Trend-based forecast of cryptocurrency returns*: https://www.sciencedirect.com/science/article/pii/S0264999323001359
- ScienceDirect, *Cryptocurrency market risk-managed momentum strategies*: https://www.sciencedirect.com/science/article/abs/pii/S1544612325011377
- arXiv, *Forecasting Cryptocurrencies Log-Returns: a LASSO-VAR and Sentiment Approach*: https://arxiv.org/abs/2210.00883
- Financial Advisor / AQR managed-futures discussion: https://www.fa-mag.com/news/trends-with-benefits-20669.html
- Kraken REST / WS trading docs: https://docs.kraken.com/api/docs/category/rest-api/trading/ and https://docs.kraken.com/api/docs/websocket-v2/add_order/
