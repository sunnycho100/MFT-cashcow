# MFT-Cashcow V3

`v3` is the current hybrid trading system under active development.

It keeps Kraken as the execution venue, but moves trade decisions onto a server-owned path that combines:
- `v2` market features and local 3-year candle history
- external overlays that actually help
- calibrated ML probabilities
- trend-following entry and exit rules
- paper / replay / live-safe execution interfaces that share the same core flow

## What V3 Is

`v2` showed that pure ML was not enough on its own. The stronger path was:
- trend and regime structure first
- ML as a filter, not the whole strategy
- better market-structure data
- stricter execution and risk control

`v3` turns that into a system:
- collectors generate normalized signals
- the decision engine owns the final trade action
- the execution layer translates those actions into Kraken orders
- paper and replay modes use the same decision shape as future live trading

## Current Champion

As of March 30, 2026, the current leading variant is `funding_premium`.

It combines:
- base `v2` features
- Deribit funding-rate overlay
- Coinbase cross-exchange premium overlay
- calibrated probabilities
- Donchian breakout entry gating
- ATR-aware exit / trailing-stop state

Current benchmark snapshot:
- 12-window walk-forward: `+1.02%` average monthly return, `0.5717` average AUC
- threshold optimization winner: `sigmoid` calibration, `0.30` long threshold, `0.35` short score threshold
- 180-day replay: `20` closed trades, `60.0%` win rate, `+3.1454%` average round-trip return
- 365-day replay: `27` closed trades, `44.44%` win rate, `+1.2754%` average round-trip return

Important caveat:
- this is still a research and paper-trading system
- replay and walk-forward results are encouraging, but not enough to justify unrestricted live trading yet

## Data Sources

Currently integrated:
- Kraken spot candles for the local market-data base
- Deribit funding history
- Coinbase spot candles for premium / cross-venue features

Research or experimental:
- Polymarket history downloader exists, but it has not yet proven positive trading impact
- Reddit / social overlays are not part of the champion path

## Main Modules

- `src/data/`: external data collectors and overlay joins
- `src/strategy/`: hybrid stack, calibration, and trend-runtime helpers
- `src/server/`: decision engine, paper runtime, and paper metrics
- `src/execution/`: Kraken execution gateway
- `src/transport/`: normalized signal transport
- `src/utils/`: config and logging helpers

## Key Workflows

Run a one-shot paper cycle:

```bash
python3 v3/main.py --mode paper-once
```

Run continuous paper trading:

```bash
python3 v3/scripts/run_paper_loop.py --iterations 0
```

Run walk-forward comparison:

```bash
python3 v3/scripts/run_walkforward_hybrid_comparison.py \
  --days 1095 \
  --train-days 365 \
  --test-days 60 \
  --step-days 60
```

Run the full evaluation stack (hybrid walk-forward + return-max + execution stress):

```bash
python3 v3/scripts/run_evaluation_suite.py
```

Skip the slower hybrid step if you only need return-max:

```bash
python3 v3/scripts/run_evaluation_suite.py --skip-hybrid
```

Optimize calibration and thresholds:

```bash
python3 v3/scripts/optimize_funding_premium_thresholds.py
```

Replay the current champion and score trade quality:

```bash
python3 v3/scripts/replay_paper_baseline.py --replay-hours 4320
```

Evaluate a saved paper log:

```bash
python3 v3/scripts/evaluate_paper_log.py
```

Download Polymarket history for research:

```bash
python3 v3/scripts/download_polymarket_history.py \
  --event-slug microstrategy-sell-any-bitcoin-in-2025 \
  --interval 1d \
  --fidelity 1440
```

## Current Behavior

The paper path now:
- refreshes Kraken candles
- loads the optimized `funding_premium` profile
- auto-selects a return-max trend profile from `return_max_walkforward_summary` based on leverage
- applies a drawdown guardrail fallback when the preferred profile exceeds cap
- calibrates probabilities before decision-making
- respects breakout and exit gates instead of firing on probability alone
- persists paper position state across cycles
- skips duplicate-candle cycles much earlier
- writes artifacts and paper logs for later evaluation

Shorting is intentionally strict:
- shorts only pass when the system is in bear regime
- a downside breakout is present
- the short gate is strong enough to pass the configured threshold

## Status

This is no longer just a scaffold. `v3` has a working integrated research loop:
- collect data
- build overlays
- run walk-forward tests
- tune thresholds
- replay decisions
- paper trade through the server-owned Kraken-safe path

It is still not fully production-ready.

## Next

- accumulate a longer real paper-trading log
- score that log with round-trip trade metrics
- add one more strong market-structure input, likely open interest or order-book imbalance
- only then consider a tighter live deployment path

## Return-Max Walk-Forward Results (Mar 30, 2026)

New return-max comparison script:

```bash
python3 v3/scripts/run_walkforward_return_max.py
```

Optional execution stress (writes a second JSON with `1.5x` fee and `2x` slippage vs your `--fee-rate` / `--slippage-bps`):

```bash
python3 v3/scripts/run_walkforward_return_max.py --stress-execution
```

Output summaries:
- `v3/data/walkforward/return_max_walkforward_summary.json` (default fee `0.001` per side, slippage `5` bps)
- `v3/data/walkforward/return_max_walkforward_execution_stress.json` (when `--stress-execution` is used)

Key results (12 windows, strict walk-forward, funding+premium overlays kept):

- `1x` baseline (`funding_premium`): about `+0.37%` avg monthly, max DD about `-7.16%`, positive windows `5/12`
- `1x` best return profile (`aggressive_v2_trend`): about `+1.71%` avg monthly, max DD about `-11.78%`, positive windows `7/12`
- `1x` top-2 pair selection (`aggressive_v2_trend_mid_top2`): about `+1.70%` avg monthly with fewer trades per window

- `2x` baseline (`funding_premium`): about `+0.70%` avg monthly, max DD about `-13.98%`, positive windows `5/12`
- `2x` objective winner (`aggressive_v2_trend_mid_top2`): about `+3.37%` avg monthly, max DD about `-21.37%`, positive windows `7/12`
- `2x` drawdown-cap-friendly top-2 (`aggressive_v2_trend_balanced_top2`): about `+3.15%` avg monthly, max DD about `-20.04%`, positive windows `7/12`

Execution stress snapshot (`fee` `0.0015`, slippage `10` bps): top profiles remain positive but monthly returns compress (for example `2x` winner near `+2.17%` on `aggressive_v2_trend_balanced_top2`).

Interpretation:
- return-max `v2` trend logic materially outperforms the current robust baseline
- pair selection and execution-cost stress tests are now part of the standard evaluation loop
- `1x` still needs another alpha layer to reach a sustained `3-5%` monthly target range
