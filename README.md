# Crypto Quantitative Trading System

A modular crypto trading system for research, model-driven signal generation, and paper execution.  
Current scope is paper trading only. Live execution is intentionally disabled.

## Features
- Multi-model signal stack: mean reversion, momentum/GARCH, and gradient boosting ML.
- Ensemble signal aggregation with confidence weighting.
- Configurable risk controls (position caps, exposure caps, stop-loss/take-profit guidance).
- Exchange integration via CCXT (Coinbase sandbox first, with Kraken/Gemini compatibility paths).
- Fallback market data pipeline (exchange -> yfinance -> synthetic data).

## Architecture
```
src/
  data/        fetch market data (exchange + fallback sources)
  models/      alpha models (MR, momentum, ML, ensemble)
  strategy/    signal-to-decision logic and position sizing
  execution/   exchange connector and order operations
scripts/
  run_paper.py paper trading loop
```

Execution flow:
1. Fetch OHLCV data for configured pairs/timeframe.
2. Fit/update models and generate ensemble signals.
3. Convert signals to trading decisions using risk constraints.
4. Execute in paper mode (dry-run by default).

## Installation
```bash
git clone <repo-url>
cd crypto-trader
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration
1. Create local environment file from template:
```bash
cp .env.example .env
```

2. Set exchange credentials in `.env`:
```bash
EXCHANGE_API_KEY=...
EXCHANGE_API_SECRET=...
```

3. Review `config.yaml`:
- `exchange.id`: `coinbase` (default)
- `exchange.mode`: `paper`
- `exchange.testnet`: `true`
- `execution.dry_run`: `true` (recommended starting point)

`EXCHANGE_API_KEY` and `EXCHANGE_API_SECRET` are loaded from environment variables defined in `config.yaml`.

## Usage
Run the dashboard test pipeline:
```bash
python3 test_dashboard.py
```

Run paper trading loop:
```bash
python3 scripts/run_paper.py --interval-seconds 60
```

Run one cycle for verification:
```bash
python3 scripts/run_paper.py --max-cycles 1
```

Disable exchange connectivity and force fallback data sources:
```bash
python3 scripts/run_paper.py --no-exchange --max-cycles 1
```

## Models
- Mean Reversion: z-score and half-life based spread reversion logic.
- Momentum/GARCH: trend-following with volatility regime filtering.
- ML Model: gradient boosting directional classifier.
- Ensemble: weighted signal voting and confidence scaling.

## Safety and Risk
- Paper trading only; do not enable live execution in this version.
- Keep `execution.dry_run: true` until sandbox behavior is validated.
- Never commit API credentials. `.env` is gitignored.
- Model output is probabilistic and can be wrong; this is not investment advice.

## Roadmap
- Exchange-specific order normalization and better historical pagination.
- Persistent portfolio ledger and fills reconciliation.
- End-to-end tests for execution and risk constraints.
- Controlled live-trading gate with explicit operator confirmation.

## License
Internal project codebase. Add a formal license before external distribution.
