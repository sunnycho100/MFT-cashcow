# Return-Max V3 Roadmap

Research date: 2026-03-30

## Goal

Translate the stronger `v2` upside into a stricter `v3` system without abandoning realistic validation.

Working target:
- primary target: `3%` to `5%` monthly unlevered in robust walk-forward testing
- stretch target: approach `10%` monthly with maximum `2x` leverage

This is an ambitious target, not a guaranteed outcome.

## Core Design Principle

Do not chase the target by loosening standards.

We should only try to raise returns by improving:
- signal quality
- entry timing
- sizing
- regime selection

We should **not** raise returns by:
- removing cost assumptions
- reducing out-of-sample rigor
- weakening short controls just to force more trades

## Development Priorities

### 1. Port the Strongest Aggressive V2 Logic into V3

Why:
- `v2` aggressive hybrid / trend configurations produced meaningfully better monthly return numbers than current robust `v3`
- right now `v3` is more robust, but not yet more profitable

What to port:
- strongest Donchian / ATR / risk-per-trade profile
- best pair-selection logic
- aggressive but still bounded exposure logic
- same exit behavior that made the `v2` return-max tests work

### 2. Keep the Winning External Layers

Keep:
- Deribit funding
- Coinbase premium

Why:
- they already improved the stronger hybrid path
- they are not the bottleneck right now

### 3. Add the Next High-Value Market-Structure Layer

Priority order:
1. Kraken order book
2. Kraken trade flow
3. open interest or liquidation pressure

Why:
- this is the strongest research-backed way to improve timing
- especially useful for shorts and breakout filtering

### 4. Optimize for Return Under a Drawdown Cap

New objective should not be "best AUC" or "best average validation score."

It should be something like:
- maximize monthly return
- while max drawdown stays below a cap
- while positive window ratio stays acceptable
- while trade count is high enough to matter

Recommended caps:
- unlevered max drawdown target: better than `-12%`
- `2x` max drawdown target: better than `-20%`

### 5. Test Explicitly at 1x and 2x

Do not optimize only unlevered and then assume leverage works.

Evaluate both:
- `1x` realistic baseline
- `2x` capped-risk target profile

And report:
- monthly return
- max drawdown
- positive window ratio
- trade count
- long / short contribution

## Concrete Research-to-Code Sequence

### Phase A: Return-Max Baseline

Build a `v3` branch that:
- mirrors the strongest `v2` aggressive config
- runs inside the strict `v3` walk-forward harness
- reports both `1x` and `2x`

Output:
- a clean apples-to-apples answer on whether `v2` upside survives stricter `v3` evaluation

### Phase B: Market-Structure Upgrade

Add:
- Kraken book snapshots
- trade-flow imbalance features
- breakout quality filters

Output:
- test whether market-structure data improves trend continuation entries and short timing

### Phase C: Return-Max Optimization

Tune:
- entry filters
- short gates
- risk-per-trade
- max position
- confidence-weighted sizing
- volatility targeting

Objective:
- highest return subject to drawdown and stability constraints

### Phase D: Paper-Trade the Return-Max Candidate

Only after the return-max candidate looks good in walk-forward:
- run continuous paper
- evaluate round-trip trade quality
- confirm that the live path still behaves as expected

## Success Criteria

Minimum acceptable:
- beats current `v3` robust baseline
- stays profitable after cost assumptions
- positive paper log behavior

Strong result:
- `3%+` monthly unlevered walk-forward
- positive round-trip replay
- acceptable drawdown

Stretch result:
- `5%+` monthly unlevered or close
- `2x` path approaches `10%` monthly without blowing up the drawdown profile

## Red Flags

Reject or pause if:
- return improves only because one or two windows dominate
- short profits come from looser gates rather than better timing
- the model looks good only in backtest but produces poor replay / paper round trips
- drawdown rises faster than return

## Bottom Line

The research supports a return-max `v3` branch.

But the credible path is:
- stronger trend configuration
- better market-structure data
- stricter short timing
- drawdown-capped optimization

Not:
- more generic ML
- more noisy sentiment
- or pretending the current `+1.02%` monthly robust figure is already enough
