# Strategy Map For V3

## Executive Summary

`v3` should not be a single ML model that tries to predict BTC direction from candles alone.

Best fit for this repo:
- Core edge: trend-following plus regime filtering.
- High-priority alpha inputs: order flow, spread/depth, cross-exchange premium, funding/open-interest.
- Secondary modules: pair trading and carry.
- Experimental overlays: Polymarket, news sentiment, Reddit attention.

---

## 1. Trend-Following + Regime Filter

### Why it belongs in `v3`

- It matches the strongest direction already emerging from `v2`.
- It is robust to both bull and bear phases.
- It is easier to govern with risk rules than pure prediction models.

### What we could build

- 4H or daily regime classifier:
  - trending up,
  - trending down,
  - range / no-trade.
- 1H setup quality score:
  - breakout strength,
  - volume confirmation,
  - volatility expansion,
  - trend alignment.
- 15m entry timing:
  - tighter entries once the higher-timeframe regime is already known.

### Paper / research benchmarks

- `AdaptiveTrend` across 150+ cryptocurrency pairs over a 36-month evaluation window reported:
  - annualized Sharpe `2.41`,
  - max drawdown `-12.7%`,
  - Calmar `3.18`.
  - Source: <https://arxiv.org/abs/2602.11708>

- A paper titled `A Decade of Evidence of Trend Following Investing in Cryptocurrencies` reported:
  - `255%` walkforward annualized returns on Bitcoin.
  - This is a very aggressive result and should be treated as an upper-bound backtest claim, not an expected live outcome.
  - Source: <https://arxiv.org/abs/2009.12155>

### V3 recommendation

- Make this the backbone of `v3`.
- Let ML filter or rank trades inside this framework instead of replacing it.

---

## 2. Order Flow / Microstructure

### Why it belongs in `v3`

- This is closer to the actual mechanism that moves price than OHLCV transforms.
- It improves both signal quality and execution quality.
- It is especially useful for deciding when not to trade.

### What we could build

- Real-time L2 depth imbalance:
  - bid depth / ask depth at top N levels.
- Spread and spread expansion features:
  - current spread,
  - spread z-score,
  - spread widening before volatility bursts.
- Aggressor-flow proxies:
  - trade imbalance,
  - buy-initiated vs sell-initiated flow,
  - signed trade volume.
- Book resiliency:
  - how quickly the book refills after sweeps.

### Paper / research benchmarks

- `Order Flow and Cryptocurrency Returns` reports that order flow materially improves cross-sectional crypto return prediction.
- Search summary from the 2026 paper reports:
  - best model with order flow: annualized Sharpe `3.63`,
  - same model class without order flow benchmarked lower,
  - average full-sample crypto return `0.21%` per day or about `52.9%` per year in the sample.
  - Source: <https://www.sciencedirect.com/science/article/pii/S1386418126000029>
  - Supporting PDF snippet: <https://www.efmaefm.org/0EFMAMEETINGS/EFMA%20ANNUAL%20MEETINGS/2025-Greece/papers/OrderFlowpaper.pdf>

- `Deep Recurrent Modelling of Stationary Bitcoin Price Formation Using the Order Flow` argues order flow models stay temporally stable through very different Bitcoin regimes, including the 2017 bubble period.
  - This is more about robustness than a quoted % return, but it strongly supports collecting order flow data early.
  - Source: <https://arxiv.org/abs/2004.01499>

### V3 recommendation

- Make this a phase-1 data source.
- Even if we do not trade ultra-short-term, order flow can gate bad entries and detect fragile liquidity.

---

## 3. Derivatives State / Carry / Positioning

### Why it belongs in `v3`

- Crypto-specific positioning signals often matter more than generic technical indicators.
- Funding and open interest help separate:
  - healthy trend continuation,
  - crowded trend exhaustion,
  - squeeze conditions,
  - liquidation cascades.

### What we could build

- Funding features:
  - current funding,
  - funding trend,
  - funding z-score,
  - cross-venue funding divergence.
- Open interest features:
  - OI change,
  - OI + price joint state,
  - OI z-score.
- Basis features:
  - spot vs futures basis,
  - basis slope across maturities.

### Paper / research benchmarks

- `Carry Trade` results from a CMU paper on crypto perpetual funding report for BTC tether contracts:
  - annualized return `21.84%`,
  - annualized standard deviation `1.90%`,
  - Sharpe `11.47`.
- The same paper reports for ETH tether contracts:
  - annualized return `27.4%`,
  - standard deviation `2.7%`,
  - Sharpe `10.25`.
- In a higher-leverage era subsample, the same paper reports:
  - BTC carry `33.03%` annualized,
  - ETH carry `44.60%` annualized.
- Source: <https://www.andrew.cmu.edu/user/azj/files/CarryTrade.v1.0.pdf>

### V3 recommendation

- Treat derivatives-state features as phase-1 or phase-1.5, depending on data access.
- Carry itself can become a separate module later, but the features are immediately useful for directional filtering.

---

## 4. Pair Trading / Market-Neutral Relative Value

### Why it belongs in `v3`

- It can diversify the directional trend module.
- It is useful when outright directional conditions are weak but relative dislocations are strong.

### What we could build

- Cointegration / spread z-score monitoring for:
  - BTC-ETH,
  - ETH-SOL,
  - BTC basket vs alt basket.
- Relative value baskets:
  - long strongest, short weakest,
  - or multivariate mean-reverting spread baskets.

### Paper / research benchmarks

- `Optimal Market-Neutral Multivariate Pair Trading on the Cryptocurrency Platform` reports:
  - annualized profit `15.49%` in 2020-2022 crypto experiments.
  - Source: <https://www.mdpi.com/2227-7072/12/3/77>

- `Reinforcement Learning Pair Trading: A Dynamic Scaling Approach` reports:
  - traditional non-RL pair trading annualized profit `8.33%`,
  - RL-based pair trading annualized profits from `9.94%` to `31.53%`.
  - Source: <https://www.mdpi.com/1911-8074/17/12/555>

### V3 recommendation

- Keep this as a phase-2 module.
- It is attractive, but it adds state, hedge-ratio, and execution complexity before the core server pipeline is battle-tested.

---

## 5. Text / Sentiment / Attention

### Why it may help

- Can capture narrative shifts before they fully appear in price.
- More useful at swing horizons than at very short horizons.

### What we could build

- News headline sentiment:
  - crypto-native sources,
  - ETF / macro headlines,
  - earnings or macro event overlays when relevant.
- Search / attention:
  - CoinGecko trending,
  - BTC dominance / market-cap rotation,
  - Google Trends if we decide to add it later.
- Social:
  - Reddit mention velocity,
  - subreddit-level polarity,
  - only as a low-weight feature.

### Paper / research benchmarks

- `Sentiment-Aware Mean-Variance Portfolio Optimization for Cryptocurrencies` reports:
  - cumulative return `38.72`,
  - Bitcoin benchmark `8.85`,
  - equal-weight crypto benchmark `21.65`,
  - Sharpe `1.1093`.
  - Source: <https://arxiv.org/abs/2508.16378>

- `Risks and Returns of Cryptocurrency` finds investor attention helps explain crypto returns and is more relevant than standard stock-style factors.
  - This is more structural support than a direct trading % benchmark, but it is one of the clearest primary-source arguments for keeping attention features as optional overlays.
  - Source: <https://www.nber.org/papers/w24877>

### Caveat

- I did not find strong evidence that raw Reddit word frequency alone is a robust standalone trading edge.
- The stronger results tend to come from structured news or better sentiment extraction, not naive keyword counting.

### V3 recommendation

- Keep this as phase-2 or phase-3.
- Prefer curated news and event text before Reddit count features.

---

## 6. Prediction Markets / Polymarket

### Why it may help

- Useful for event risk and macro odds:
  - Fed,
  - elections,
  - ETF approvals,
  - recession or inflation narratives.

### What we could build

- Event overlay features:
  - market-implied probability of a catalyst,
  - change in probability over 24h / 7d,
  - disagreement between event markets and price action.

### Evidence / calibration notes

- Prediction-market research supports calibration usefulness in general:
  - Source: <https://academic.oup.com/ej/article/123/568/491/5079498>
- Independent dashboards currently track Polymarket Brier performance:
  - example references: <https://brier.fyi/charts/polymarket/>
  - official accuracy page: <https://polymarket.com/accuracy/>

### V3 recommendation

- Use this as an event overlay only.
- Do not make it the main BTC direction engine.

---

## Final Strategy Ranking For V3

1. Trend-following + regime filter
2. Order flow / microstructure
3. Derivatives state / funding / OI / basis
4. Pair trading / market-neutral relative value
5. News sentiment / attention
6. Polymarket event overlay
7. Reddit-specific features

## Final Build Recommendation

- Phase 1:
  - trend + regime,
  - Kraken execution ownership,
  - order flow and spread/depth,
  - derivatives-state features.
- Phase 2:
  - pair trading,
  - curated news sentiment,
  - event overlays.
- Phase 3:
  - Reddit and broader social attention experiments.
