# 3-31-2026 v3 active work checklist

Step-by-step backlog for return-max `v3`. Status is a snapshot of what is implemented in-repo.

## Completed (this iteration or prior)

1. Return-max walk-forward harness with funding+premium overlays, `1x`/`2x`, ML on/off gate, pair selection (`top1`/`top2`), execution fee/slippage flags, optional execution stress JSON.
2. Confidence-weighted sizing optional path in `v2` trend backtest.
3. Paper runtime: return-max trend profile resolution, drawdown guardrail, `ml_filter: false` edge-gate alignment, `pair_selection` + `active_pairs` in artifacts.
4. Execution stress artifact: `v3/data/walkforward/return_max_walkforward_execution_stress.json`.

## In progress / next (ordered)

1. ~~**One-command evaluation**~~ — `v3/scripts/run_evaluation_suite.py` runs hybrid walk-forward comparison + return-max (+ stress) and prints output paths. *(done)*
2. ~~**Feature layer (OHLCV proxies)**~~ — `volume_zscore_24`, `atr_ratio_14_48` added in `v2/src/features/pipeline.py`. *(done; retrain implied on next full run)*
3. **Market-structure (real)** — Kraken REST depth or WS book/trade collectors; persist snapshots; join features (separate milestone).
4. **Multi-asset breadth** — add liquid alts to `v2/config.yaml` after confirming history in DuckDB; extend overlay maps in `hybrid_stack`.
5. **Replay + paper scoring** — automate replay baseline + paper log eval after each major change.
6. **Production policy** — freeze default profile + caps from latest JSON; version artifacts in git or dated copies.

## Anti-goals

- Do not loosen short rules for trade count.
- Do not ship order-book claims without stored history and alignment tests.
