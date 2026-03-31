# 3-30-2026 v3 return max development plan

## Objective

Make `v3` more aggressive and more profitable without faking the validation.

Working target:
- primary target: `3%` to `5%` average monthly return at `1x`
- stretch target: approach `10%` average monthly return at maximum `2x`

This plan assumes:
- we keep `v3` as the main system
- we preserve strict walk-forward evaluation
- we do not loosen short rules just to inflate trade count

## Core Principles

1. Keep trend-first structure.
2. Use ML as a filter / confidence layer, not the whole strategy.
3. Increase returns by improving timing, sizing, and market selection.
4. Penalize drawdown, not just reward return.
5. Compare every aggressive upgrade at both `1x` and `2x`.

## Success Metrics

Minimum success:
- beats current `v3` `funding_premium` walk-forward baseline
- profitable round-trip replay
- stable paper behavior

Target success:
- `3%+` average monthly return at `1x`
- positive window ratio of at least `50%`
- max drawdown better than `-12%`

Stretch success:
- `5%+` average monthly return at `1x`
- `2x` path close to `10%` monthly
- `2x` max drawdown better than `-20%`

## Phase 1: Aggressive V2 Logic Inside V3

Goal:
- port the strongest aggressive `v2` hybrid / trend config into the strict `v3` walk-forward harness

Why:
- `v2` had stronger upside than current robust `v3`
- we need an apples-to-apples test under the same `v3` validation standards

Tasks:
- identify the best aggressive `v2` config from the return-max tests
- move its Donchian / ATR / sizing logic into reusable `v3` strategy code
- support return reporting at `1x` and `2x`
- compare against current `funding_premium`

Deliverable:
- a new `v3` walk-forward report showing whether aggressive `v2` logic survives stricter validation

## Phase 2: Return-Max Objective Function

Goal:
- stop optimizing only for AUC or loose ranking metrics

New optimization objective:
- maximize average monthly return
- while keeping drawdown below a cap
- while keeping trade count meaningful
- while keeping positive window ratio acceptable

Tasks:
- build a score that combines:
  - average monthly return
  - max drawdown penalty
  - positive window ratio
  - trade count floor
- run optimizer for both `1x` and `2x`

Deliverable:
- ranked aggressive profiles by real trading objective, not classifier quality alone

## Phase 3: Market-Structure Upgrade

Goal:
- improve entry quality and especially short timing

Priority data:
1. Kraken order book
2. Kraken trades
3. derivatives open interest or liquidation stress

Features to build:
- spread regime
- top-of-book imbalance
- depth imbalance
- aggressive buyer / seller dominance
- breakout quality filters
- failed breakout / absorption filters

Deliverable:
- market-structure overlay that can be tested on top of the aggressive `v3` baseline

## Phase 4: Better Short Logic

Goal:
- make shorts rarer but more profitable

Requirements for short entries:
- bear regime
- downside breakout
- weak premium / basis context
- hostile funding / positioning context
- negative flow or failed rally confirmation

Tasks:
- convert current short logic from threshold-only gating into confluence gating
- report long and short contribution separately

Deliverable:
- a short module that is stricter but more intentional

## Phase 5: Aggressive Sizing and Risk Controls

Goal:
- increase returns through better sizing, not just more trades

Tasks:
- test confidence-weighted sizing
- test volatility-targeted sizing
- test pair-level exposure caps
- keep hard stop on total exposure
- compare `1x` and `2x` explicitly

Deliverable:
- sizing layer that increases return without letting drawdown explode

## Phase 6: Validation Stack

Goal:
- make sure higher returns are real

Validation sequence:
1. walk-forward
2. replay with round-trip trade metrics
3. continuous paper log
4. only then live-safe deployment path

Required reports:
- average monthly return
- total out-of-sample return
- positive window ratio
- max drawdown
- long / short contribution
- trade count
- replay round-trip return

Deliverable:
- one clear scorecard for every aggressive candidate

## Build Order

Immediate next step:
1. Phase 1

After that:
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5
6. Phase 6

## What We Should Not Do

- do not start with Reddit or generic sentiment
- do not loosen short thresholds just to create more trades
- do not optimize on one split and call it done
- do not rely on classifier AUC as the main target metric

## Immediate Coding Task

The next concrete implementation task is:
- build a `return-max v3` branch of the current strategy stack
- port the strongest aggressive `v2` config into `v3`
- run strict walk-forward at `1x` and `2x`
- compare it directly to current `funding_premium`
