# 3-30-2026 v3 pivot plan (frequency + scaling)

## Objective

Pivot `v3` from low-frequency, high-variance return paths toward a more scalable return engine.

Working target:
- primary target: `3%` to `5%` average monthly return with robust validation
- stretch target: approach `10%` monthly only as a high-risk upside path, not as baseline expectation

This plan assumes:
- we keep strict walk-forward and replay discipline
- we keep trend/regime-first structure
- we do not loosen short logic just to inflate trade count

## Why We Are Pivoting

Current issue:
- robust variants are still too sparse in trade count for stable high monthly compounding
- sparse trading forces oversized per-trade edge assumptions
- current ML edge quality is not yet strong enough to justify heavy filtering by itself

Practical implication:
- we need more opportunities, better sizing logic, and stronger cross-pair breadth

## Core Principles

1. Increase opportunity count without dropping validation quality.
2. Keep trend as backbone, but test faster variants and complementary entry logic.
3. Use ML confidence for sizing and gating only when it proves additive.
4. Score candidates by return and drawdown and trade count together.
5. Treat 10% monthly as stretch, not as guaranteed target.

## Success Metrics

Minimum success:
- beats current `funding_premium` at `1x` and `2x`
- positive window ratio at least `50%`
- no collapse in replay quality

Target success:
- `3%+` average monthly at `1x`
- max drawdown better than `-12%` at `1x`
- meaningful trade count per window

Stretch success:
- `5%+` average monthly at `1x`
- `2x` path near `10%` monthly with max drawdown better than `-20%`

## Phase 1: Frequency and Sizing Pivot (Immediate)

Goal:
- increase trade frequency while keeping risk bounded

Tasks:
- add confidence-weighted position sizing (ML confidence scales risk budget)
- test faster trend variants (`14-15d` entry range) in strict walk-forward
- update objective with a trade-count floor so "good but too sparse" profiles are penalized
- compare all candidates directly to `funding_premium`

Deliverable:
- updated return-max summary with frequency-aware and sizing-aware rankings

## Phase 2: ML Utility Gate

Goal:
- decide whether ML filter is helping or hurting each profile

Tasks:
- run each candidate with and without ML filter under same windows
- report delta in monthly return, drawdown, and trade count
- keep ML only where it adds net benefit

Deliverable:
- explicit keep/drop decision for ML on each major profile

## Phase 3: Multi-Asset Breadth Upgrade

Goal:
- diversify opportunity set beyond current concentration

Tasks:
- expand tradable universe to additional liquid alts with adequate history
- reuse funding and premium overlays on added pairs
- enforce pair-level caps and portfolio exposure constraints

Deliverable:
- walk-forward scorecard on expanded pair universe

## Phase 4: Execution Alpha Layer

Goal:
- improve realized edge without changing raw signal quality

Tasks:
- add limit-first order policy in paper simulation path
- evaluate slippage reduction and fill quality by regime
- compare realized returns vs market-style execution assumptions

Deliverable:
- execution-impact report with basis-point savings and return lift

## Phase 5: Market-Structure Timing Layer

Goal:
- improve entry quality, especially for shorts

Tasks:
- add Kraken order-book and trade-flow imbalance features
- add breakout quality and failed-breakout filters
- integrate into strict short confluence gates

Deliverable:
- market-structure-augmented candidate with robust walk-forward report

## Validation Rules

- walk-forward first, replay second, paper third
- no cherry-picked window claims
- no AUC-only optimization claims
- no short-rule loosening solely for higher trade count

## Immediate Coding Task

Start now with:
- confidence-weighted sizing in trend backtest
- frequency-aware objective function in return-max walk-forward
- new candidate profiles that test the frequency/sizing pivot at `1x` and `2x`
