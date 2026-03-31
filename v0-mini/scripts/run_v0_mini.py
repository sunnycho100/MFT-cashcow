#!/usr/bin/env python3
"""Run v0-mini: price-vs-EMA and EMA(30/50/120) crossover backtests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

MINI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MINI_ROOT.parent
sys.path.insert(0, str(MINI_ROOT))

from src.data_loader import load_ohlcv  # noqa: E402
from src.ema_backtest import (  # noqa: E402
    BacktestResult,
    compute_ema_columns,
    run_ema_cross_backtest,
    run_ema50_vs_ema120_backtest,
    run_price_vs_ema_backtest,
)


def _load_config() -> dict:
    import yaml

    cfg_path = MINI_ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _print_result_block(title: str, res: BacktestResult) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    s = res.summary
    if "error" in s:
        print(json.dumps(s, indent=2))
        return
    print(json.dumps({k: v for k, v in s.items() if k != "pair"}, indent=2))
    print(f"\n{res.pair}: final_equity={s.get('final_equity'):.2f}  total_return_pct={s.get('total_return_pct'):.4f}%  trades={s.get('n_trades')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="v0-mini: price vs EMA + EMA cross backtests")
    parser.add_argument("--csv-only", action="store_true", help="Skip v2 DuckDB; use v3 CSV only")
    parser.add_argument("--no-export", action="store_true", help="Do not write parquet/json to v0-mini/data/")
    args = parser.parse_args()

    cfg = _load_config()
    pairs: list[str] = cfg["pairs"]
    ema_cfg = cfg["ema"]
    strat = cfg.get("strategies", {})
    price_cfg = strat.get("price_vs_ema", {"signal": 50, "trend": 120})
    triple_cfg = strat.get("ema_triple", {"fast": 30, "mid": 50, "slow": 120})
    bt = cfg["backtest"]
    out_dir = REPO_ROOT / cfg["paths"]["derived_output_dir"]
    if not args.no_export:
        out_dir.mkdir(parents=True, exist_ok=True)

    prefer_duckdb = not args.csv_only

    print(
        "v0-mini — three modes:\n"
        "  (1) PRICE: golden/dead = close crosses above/below EMA(signal); "
        f"long only when close > EMA({price_cfg['trend']}) on entry.\n"
        "  (2) EMA 30×50: golden/dead = EMA(30) vs EMA(50); "
        f"long only when close > EMA({ema_cfg['trend']}).\n"
        "  (3) EMA 50×120: golden/dead = EMA(50) vs EMA(120); "
        f"long only when close > EMA({triple_cfg['fast']}).\n"
    )

    all_outputs: list[dict] = []

    for pair in pairs:
        raw = load_ohlcv(REPO_ROOT, pair, timeframe=cfg.get("timeframe", "1h"), prefer_duckdb=prefer_duckdb)

        enriched = compute_ema_columns(
            raw,
            fast=ema_cfg["fast"],
            slow=ema_cfg["slow"],
            trend=ema_cfg["trend"],
        )
        if not args.no_export and not enriched.is_empty():
            slug = pair.replace("/", "_")
            pq = out_dir / f"ohlcv_with_ema_{slug}.parquet"
            enriched.write_parquet(pq)

        r1 = run_price_vs_ema_backtest(
            raw,
            pair=pair,
            signal=price_cfg["signal"],
            trend=price_cfg["trend"],
            initial_capital=bt["initial_capital"],
            fee_rate=bt["fee_rate"],
            slippage_bps=bt["slippage_bps"],
        )
        _print_result_block(
            f"(1) Price vs EMA({price_cfg['signal']}) — {pair}",
            r1,
        )

        r2 = run_ema_cross_backtest(
            raw,
            pair=pair,
            fast=ema_cfg["fast"],
            slow=ema_cfg["slow"],
            trend=ema_cfg["trend"],
            initial_capital=bt["initial_capital"],
            fee_rate=bt["fee_rate"],
            slippage_bps=bt["slippage_bps"],
        )
        _print_result_block(
            f"(2) EMA({ema_cfg['fast']}) × EMA({ema_cfg['slow']}) trend EMA({ema_cfg['trend']}) — {pair}",
            r2,
        )

        r3 = run_ema50_vs_ema120_backtest(
            raw,
            pair=pair,
            fast=triple_cfg["fast"],
            mid=triple_cfg["mid"],
            slow=triple_cfg["slow"],
            initial_capital=bt["initial_capital"],
            fee_rate=bt["fee_rate"],
            slippage_bps=bt["slippage_bps"],
        )
        _print_result_block(
            f"(3) EMA({triple_cfg['mid']}) × EMA({triple_cfg['slow']}) filter close>EMA({triple_cfg['fast']}) — {pair}",
            r3,
        )

        for label, res in (
            ("price_vs_ema", r1),
            ("ema_30_50_120trend", r2),
            ("ema_50_120_30filter", r3),
        ):
            row = {"pair": pair, "strategy": label, **res.summary}
            if not res.monthly.is_empty():
                row["monthly"] = res.monthly.to_dicts()
            all_outputs.append(row)

    if not args.no_export:
        summary_path = out_dir / "v0_mini_comparison.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_outputs, f, indent=2)
        print(f"\nWrote {summary_path.relative_to(REPO_ROOT)}")

    # Compact comparison table
    print("\n" + "=" * 72 + "\nSUMMARY (total_return_pct %)\n" + "=" * 72)
    rows = []
    for o in all_outputs:
        if "total_return_pct" in o:
            rows.append(
                {
                    "pair": o["pair"],
                    "strategy": o["strategy"],
                    "total_return_pct": round(o["total_return_pct"], 4),
                    "n_trades": o.get("n_trades"),
                }
            )
    print(pl.DataFrame(rows).sort(["strategy", "pair"]))


if __name__ == "__main__":
    main()
