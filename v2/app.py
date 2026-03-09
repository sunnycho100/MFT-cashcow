#!/usr/bin/env python3
"""MFT-cashcow V2 — Terminal UI Dashboard.

A minimalist black-and-white TUI for training, backtesting,
and paper trading the LightGBM model.

Usage:
    python3 v2/app.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project paths
_root = Path(__file__).resolve().parent
_project = _root.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_project))

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import (
    Header,
    Footer,
    Static,
    Button,
    DataTable,
    Log,
    Label,
    Rule,
)
from textual.screen import Screen

from src.utils.config import load_config
from src.data.store import DataStore
from src.features.pipeline import build_features, get_feature_names
from src.models.lgbm_model import LGBMModel, LABEL_MAP


# ======================================================================
# Helper: load model metadata without loading full model
# ======================================================================

def get_model_status(config: dict) -> dict:
    """Check if a trained model exists and return its metadata."""
    ckpt = Path(config.get("data", {}).get("storage_path", "./v2/data")).parent / "checkpoints" / "lgbm_meta.json"
    if not ckpt.exists():
        return {"trained": False}
    with open(ckpt) as f:
        meta = json.load(f)
    return {"trained": True, **meta}


def get_data_status(config: dict) -> dict:
    """Check stored data summary."""
    try:
        store = DataStore(config)
        summary = store.summary()
        store.close()
        if summary.is_empty():
            return {"has_data": False, "pairs": []}
        return {
            "has_data": True,
            "pairs": summary["pair"].to_list(),
            "rows": summary["rows"].to_list(),
            "first": str(summary["first_candle"][0]),
            "last": str(summary["last_candle"][0]),
        }
    except Exception:
        return {"has_data": False, "pairs": []}


# ======================================================================
# Main Dashboard Screen
# ======================================================================

class DashboardScreen(Screen):
    """Main menu / status screen."""

    DEFAULT_CSS = """
    DashboardScreen {
        background: $surface;
        layout: vertical;
    }

    /* ── pinned top section ── */
    #top-panel {
        height: auto;
        max-height: 60%;
    }
    #title-box {
        height: auto;
        padding: 0 2;
        text-align: center;
    }
    #summary-row {
        height: auto;
        padding: 0 2;
        margin: 0 2;
    }
    #data-status {
        height: auto;
        width: 1fr;
        padding: 0 1;
        border: solid $primary;
    }
    #model-status {
        height: auto;
        width: 1fr;
        padding: 0 1;
        border: solid $primary;
    }
    #backtest-status {
        height: auto;
        width: 1fr;
        padding: 0 1;
        border: solid $primary;
    }
    #button-row {
        height: auto;
        padding: 0 2;
        align: center middle;
    }
    #button-row Button {
        margin: 0 1;
        min-width: 20;
    }

    /* ── log fills remaining space ── */
    #log-box {
        height: 1fr;
        min-height: 6;
        margin: 0 2;
        border: solid $primary;
    }
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._last_backtest: dict | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="top-panel"):
            yield Static(
                "[bold]MFT-CASHCOW  ·  V2[/bold]  —  Feature-First Gradient Boosting",
                id="title-box",
            )
            with Horizontal(id="summary-row"):
                yield Static(id="data-status")
                yield Static(id="model-status")
                yield Static(id="backtest-status")
            yield Horizontal(
                Button("Train", id="btn-train", variant="default"),
                Button("Backtest", id="btn-backtest", variant="default"),
                Button("Paper Trade", id="btn-paper", variant="default"),
                Button("Data Info", id="btn-data", variant="default"),
                Button("Quit", id="btn-quit", variant="error"),
                id="button-row",
            )
        yield Log(id="log-box", highlight=True, auto_scroll=True)
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_status()

    def _refresh_status(self) -> None:
        model = get_model_status(self.config)
        data = get_data_status(self.config)

        # ── Data panel ──
        dl = ["[bold]DATA[/bold]"]
        if data["has_data"]:
            for p, r in zip(data["pairs"], data["rows"]):
                dl.append(f"  {p}: {r:,}")
            dl.append(f"  {data['first'][:10]} → {data['last'][:10]}")
        else:
            dl.append("  [red]No data[/red]")
        self.query_one("#data-status", Static).update("\n".join(dl))

        # ── Model panel ──
        ml = ["[bold]MODEL[/bold]"]
        if model["trained"]:
            acc = model.get("metrics", {}).get("accuracy", "?")
            n_feat = model.get("metrics", {}).get("n_features", "?")
            best_iter = model.get("metrics", {}).get("best_iteration", "?")
            ml.append(f"  [green]Trained[/green]  Acc: {acc}")
            ml.append(f"  Features: {n_feat}  Iter: {best_iter}")
            per_class = model.get("metrics", {}).get("per_class", {})
            if per_class:
                for k, v in per_class.items():
                    ml.append(f"  {k}: P={v['precision']} R={v['recall']} F1={v['f1']}")
        else:
            ml.append("  [yellow]Not trained[/yellow]")
        self.query_one("#model-status", Static).update("\n".join(ml))

        # ── Backtest panel ──
        bl = ["[bold]BACKTEST[/bold]"]
        bt = self._last_backtest
        if bt:
            ret = bt["total_return_pct"]
            bh = bt["buy_hold_return_pct"]
            ret_color = "green" if ret >= 0 else "red"
            bl.append(f"  Return: [{ret_color}]{ret:+.2f}%[/{ret_color}]  vs B&H: {bh:+.2f}%")
            bl.append(f"  Sharpe: {bt['sharpe_ratio']:.2f}  MaxDD: {bt['max_drawdown_pct']:.2f}%")
            bl.append(f"  Trades: {bt['total_trades']}  Win: {bt['win_rate_pct']:.1f}%")
            bl.append(f"  PF: {bt['profit_factor']:.2f}  Equity: ${bt['final_equity']:,.0f}")
        else:
            bl.append("  [dim]Not run yet[/dim]")
        self.query_one("#backtest-status", Static).update("\n".join(bl))

    def _log(self, msg: str) -> None:
        self.query_one("#log-box", Log).write_line(msg)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    @on(Button.Pressed, "#btn-train")
    def handle_train(self) -> None:
        self._run_training()

    @on(Button.Pressed, "#btn-backtest")
    def handle_backtest(self) -> None:
        self._run_backtest()

    @on(Button.Pressed, "#btn-paper")
    def handle_paper(self) -> None:
        self._run_paper_trade()

    @on(Button.Pressed, "#btn-data")
    def handle_data(self) -> None:
        self._show_data_info()

    @on(Button.Pressed, "#btn-quit")
    def handle_quit(self) -> None:
        self.app.exit()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @work(thread=True)
    def _run_training(self) -> None:
        log = self._log

        # Check if already trained
        status = get_model_status(self.config)
        if status["trained"]:
            log(f"⚠  Model already trained at {status['trained_at']}")
            log(f"   Accuracy: {status['metrics']['accuracy']}")
            log("   To retrain, delete v2/checkpoints/lgbm_model.pkl")
            return

        log("━" * 50)
        log("TRAINING STARTED")
        log("━" * 50)

        try:
            store = DataStore(self.config)
            train_days = self.config.get("data", {}).get("train_days", 365)
            pairs = self.config.get("trading", {}).get("pairs", ["BTC/USDT"])

            log(f"Loading data for {pairs[0]} (last {train_days} days) ...")
            df = store.load_ohlcv(pairs[0], "1h", last_n_days=train_days)
            store.close()

            if len(df) < 500:
                log(f"✗ Not enough data: {len(df)} rows (need 500+)")
                return

            log(f"  {len(df):,} rows loaded")

            log("Building features ...")
            feat_df = build_features(df)
            feature_names = get_feature_names(feat_df)
            log(f"  {len(feature_names)} features built, {len(feat_df):,} rows after warmup")

            model = LGBMModel(self.config)
            log("Creating labels ...")
            labeled = model.create_labels(feat_df)
            log(f"  {len(labeled):,} labeled rows")

            log("Training LightGBM ...")
            metrics = model.train(labeled, feature_names, train_days=train_days)

            log("━" * 50)
            log(f"✓ TRAINING COMPLETE")
            log(f"  Accuracy:       {metrics['accuracy']:.2%}")
            log(f"  Train rows:     {metrics['train_rows']:,}")
            log(f"  Test rows:      {metrics['test_rows']:,}")
            log(f"  Best iteration: {metrics['best_iteration']}")
            log(f"  Features:       {metrics['n_features']}")

            for cls, vals in metrics.get("per_class", {}).items():
                log(f"  {cls:>5}: P={vals['precision']:.3f}  R={vals['recall']:.3f}  F1={vals['f1']:.3f}")

            log("")
            log("Top 10 features:")
            for name, imp in model.feature_importance(10):
                log(f"  {name:<25} {imp:.0f}")

            log("━" * 50)

            # Refresh status panel
            self.app.call_from_thread(self._refresh_status)

        except Exception as e:
            log(f"✗ Training failed: {e}")

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    @work(thread=True)
    def _run_backtest(self) -> None:
        log = self._log

        model = LGBMModel(self.config)
        if not model.load():
            log("✗ No trained model. Train first.")
            return

        log("━" * 50)
        log("BACKTEST STARTED")
        log("━" * 50)

        try:
            from src.validation.backtest import run_backtest

            store = DataStore(self.config)
            pairs = self.config.get("trading", {}).get("pairs", ["BTC/USDT"])
            pair = pairs[0]

            log(f"Loading full data for {pair} ...")
            df = store.load_ohlcv(pair, "1h")
            store.close()

            log("Building features ...")
            feat_df = build_features(df)

            log("Running predictions ...")
            pred_df = model.predict(feat_df)

            # Use last 20% as out-of-sample backtest
            test_start = int(len(pred_df) * 0.8)
            test_df = pred_df[test_start:]

            log(f"Backtesting on {len(test_df):,} bars (last 20%) ...")
            result = run_backtest(test_df, confidence_threshold=0.45)
            m = result["metrics"]

            log("━" * 50)
            log("BACKTEST RESULTS")
            log("━" * 50)
            log(f"  Total Return:    {m['total_return_pct']:+.2f}%")
            log(f"  Buy & Hold:      {m['buy_hold_return_pct']:+.2f}%")
            log(f"  Sharpe Ratio:    {m['sharpe_ratio']:.2f}")
            log(f"  Max Drawdown:    {m['max_drawdown_pct']:.2f}%")
            log(f"  Total Trades:    {m['total_trades']}")
            log(f"  Win Rate:        {m['win_rate_pct']:.1f}%")
            log(f"  Avg Win:         {m['avg_win_pct']:+.3f}%")
            log(f"  Avg Loss:        {m['avg_loss_pct']:+.3f}%")
            log(f"  Profit Factor:   {m['profit_factor']:.2f}")
            log(f"  Final Equity:    ${m['final_equity']:,.2f}")
            log("━" * 50)

            # Update top-panel backtest summary
            self._last_backtest = m
            self.app.call_from_thread(self._refresh_status)

        except Exception as e:
            log(f"✗ Backtest failed: {e}")
            import traceback
            log(traceback.format_exc())

    # ------------------------------------------------------------------
    # Paper trading
    # ------------------------------------------------------------------

    @work(thread=True)
    def _run_paper_trade(self) -> None:
        log = self._log

        model = LGBMModel(self.config)
        if not model.load():
            log("✗ No trained model. Train first.")
            return

        log("━" * 50)
        log("PAPER TRADING — fetching latest data ...")
        log("━" * 50)

        try:
            from src.data.fetcher import DataFetcher
            fetcher = DataFetcher(self.config)
            pairs = self.config.get("trading", {}).get("pairs", ["BTC/USDT"])

            for pair in pairs:
                log(f"\n[{pair}]")
                df = fetcher.fetch(pair, "1h", days=15)
                if len(df) < 200:
                    log(f"  ✗ Not enough data ({len(df)} rows)")
                    continue

                feat_df = build_features(df)
                if len(feat_df) == 0:
                    log(f"  ✗ Feature build failed")
                    continue

                latest = feat_df.tail(1)
                pred_df = model.predict(latest)

                price = float(latest["close"][0])
                pred_label = pred_df["pred_label"][0]
                prob_up = float(pred_df["pred_prob_up"][0])
                prob_down = float(pred_df["pred_prob_down"][0])
                prob_flat = float(pred_df["pred_prob_flat"][0])
                confidence = max(prob_up, prob_down, prob_flat)

                log(f"  Price:      ${price:,.2f}")
                log(f"  Prediction: {pred_label}")
                log(f"  Confidence: {confidence:.1%}")
                log(f"  Probs:      UP={prob_up:.3f}  DOWN={prob_down:.3f}  FLAT={prob_flat:.3f}")

                if pred_label == "UP" and prob_up > 0.45:
                    log(f"  → Signal: [bold]BUY[/bold]")
                elif pred_label == "DOWN" and prob_down > 0.45:
                    log(f"  → Signal: [bold]SELL[/bold]")
                else:
                    log(f"  → Signal: HOLD (low confidence)")

            log("\n━ Paper trade snapshot complete ━")
            log("(In production, this runs continuously every hour)")

        except Exception as e:
            log(f"✗ Paper trade failed: {e}")
            import traceback
            log(traceback.format_exc())

    # ------------------------------------------------------------------
    # Data info
    # ------------------------------------------------------------------

    @work(thread=True)
    def _show_data_info(self) -> None:
        log = self._log
        log("━" * 50)
        log("DATA SUMMARY")
        log("━" * 50)

        try:
            store = DataStore(self.config)
            summary = store.summary()

            if summary.is_empty():
                log("  No data. Run: python3 v2/scripts/fetch_data.py")
            else:
                for row in summary.iter_rows(named=True):
                    log(f"  {row['pair']} ({row['timeframe']}): {row['rows']:,} rows")
                    log(f"    {row['first_candle']}  →  {row['last_candle']}")

            store.close()
        except Exception as e:
            log(f"✗ {e}")

        log("━" * 50)


# ======================================================================
# App
# ======================================================================

class CashcowApp(App):
    """MFT-cashcow V2 TUI."""

    TITLE = "MFT-CASHCOW V2"
    CSS = """
    Screen {
        background: $surface;
    }
    """
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("t", "train", "Train"),
        ("b", "backtest", "Backtest"),
        ("p", "paper", "Paper Trade"),
        ("d", "data", "Data Info"),
    ]

    def __init__(self):
        super().__init__()
        self.config = load_config(str(_root / "config.yaml"))

    def on_mount(self) -> None:
        self.push_screen(DashboardScreen(self.config))

    def action_quit(self) -> None:
        self.exit()

    def action_train(self) -> None:
        screen = self.screen
        if isinstance(screen, DashboardScreen):
            screen.handle_train()

    def action_backtest(self) -> None:
        screen = self.screen
        if isinstance(screen, DashboardScreen):
            screen.handle_backtest()

    def action_paper(self) -> None:
        screen = self.screen
        if isinstance(screen, DashboardScreen):
            screen.handle_paper()

    def action_data(self) -> None:
        screen = self.screen
        if isinstance(screen, DashboardScreen):
            screen.handle_data()


if __name__ == "__main__":
    CashcowApp().run()
