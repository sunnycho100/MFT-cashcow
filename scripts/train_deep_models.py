#!/usr/bin/env python3
"""Main training pipeline for Option A deep learning models."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.fetcher import DataFetcher
from src.models.lstm_cnn_hybrid import LSTMCNNHybrid
from src.models.reinforcement_learning import PPOTradingAgent
from src.models.temporal_fusion_transformer import TemporalFusionTransformer


def setup_logging() -> logging.Logger:
    """Configure console and file logging for training jobs."""
    log_dir = Path("logs/training")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger("Main")
    logger.info("Training log file: %s", log_file)
    return logger


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML config with helpful error messages."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file is empty or invalid.")

    return config


def load_training_data(config: dict) -> dict[str, pl.DataFrame]:
    """Load historical data from disk, falling back to DataFetcher."""
    logger = logging.getLogger("Main")

    hist_dir = Path("data/historical")
    data: dict[str, pl.DataFrame] = {}

    if hist_dir.exists():
        for fp in sorted(hist_dir.glob("*.parquet")):
            try:
                df = pl.read_parquet(fp)
                pair = fp.stem.replace("_", "/").upper()
                data[pair] = df
                logger.info("Loaded %s rows from %s", len(df), fp)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", fp, exc)

        for fp in sorted(hist_dir.glob("*.csv")):
            pair = fp.stem.replace("_", "/").upper()
            if pair in data:
                continue
            try:
                df = pl.read_csv(fp, try_parse_dates=True)
                data[pair] = df
                logger.info("Loaded %s rows from %s", len(df), fp)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", fp, exc)

    if data:
        return data

    logger.info("No local historical files found. Fetching via DataFetcher.")
    fetcher = DataFetcher(config)

    pairs = config.get("trading", {}).get("pairs", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    timeframe = config.get("trading", {}).get("primary_timeframe", "1h")
    days = int(config.get("data", {}).get("cache_days", 365))

    fetched = fetcher.fetch_multiple(pairs=pairs, timeframe=timeframe, days=days)
    if not fetched:
        raise RuntimeError("No training data available after fetch attempt.")

    return fetched


def _ensure_checkpoint_dirs() -> None:
    Path("checkpoints/tft").mkdir(parents=True, exist_ok=True)
    Path("checkpoints/lstm_cnn").mkdir(parents=True, exist_ok=True)
    Path("checkpoints/rl_agent").mkdir(parents=True, exist_ok=True)


def train_tft(model: TemporalFusionTransformer, data: dict[str, pl.DataFrame], epochs: int) -> TemporalFusionTransformer:
    logger = logging.getLogger("TFT")
    logger.info("Starting TFT training...")

    for epoch in range(epochs):
        loss = model.train_epoch(data)
        logger.info("Epoch %s/%s | Loss: %.6f", epoch + 1, epochs, loss)

        if (epoch + 1) % 10 == 0:
            ckpt = f"checkpoints/tft/epoch_{epoch + 1}.pt"
            model.save_checkpoint(ckpt)

    model._is_fitted = True
    logger.info("TFT training complete")
    return model


def train_lstm_cnn(model: LSTMCNNHybrid, data: dict[str, pl.DataFrame], epochs: int) -> LSTMCNNHybrid:
    logger = logging.getLogger("LSTM_CNN")
    logger.info("Starting LSTM-CNN training...")

    for epoch in range(epochs):
        metrics = model.train_epoch(data)
        logger.info(
            "Epoch %s/%s | Loss: %.6f | Accuracy: %.4f",
            epoch + 1,
            epochs,
            metrics["loss"],
            metrics["accuracy"],
        )

        if (epoch + 1) % 10 == 0:
            ckpt = f"checkpoints/lstm_cnn/epoch_{epoch + 1}.pt"
            model.save_checkpoint(ckpt)

    model._is_fitted = True
    logger.info("LSTM-CNN training complete")
    return model


def train_rl_agent(
    model: PPOTradingAgent,
    data: dict[str, pl.DataFrame],
    tft_model: TemporalFusionTransformer,
    lstm_model: LSTMCNNHybrid,
    timesteps: int,
) -> PPOTradingAgent:
    logger = logging.getLogger("RL_Agent")
    logger.info("Starting RL agent training...")

    logger.info("Generating predictions for RL training environment...")
    tft_preds = tft_model.predict_all(data)
    lstm_preds = lstm_model.predict_all(data)

    model.fit(
        historical_data=data,
        tft_predictions=tft_preds,
        lstm_predictions=lstm_preds,
        total_timesteps=timesteps,
    )

    model.save_checkpoint("checkpoints/rl_agent/final_model")
    logger.info("RL agent training complete")
    return model


def maybe_resume(models: dict[str, object], checkpoint: Optional[str]) -> None:
    """Resume one of the models from a checkpoint path if provided."""
    if not checkpoint:
        return

    logger = logging.getLogger("Main")
    ckpt = checkpoint.lower()

    if "tft" in ckpt:
        models["tft"].load_checkpoint(checkpoint)
        logger.info("Resumed TFT from %s", checkpoint)
    elif "lstm" in ckpt:
        models["lstm_cnn"].load_checkpoint(checkpoint)
        logger.info("Resumed LSTM-CNN from %s", checkpoint)
    elif "rl" in ckpt:
        models["rl_agent"].load_checkpoint(checkpoint)
        logger.info("Resumed RL agent from %s", checkpoint)
    else:
        raise ValueError("Could not infer model type from checkpoint path.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deep learning models for crypto trading")
    parser.add_argument("--mode", choices=["full", "quick", "resume"], default="full")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging()

    logger.info("Training mode: %s", args.mode)
    logger.info("Device: %s", torch.device("mps" if torch.backends.mps.is_available() else "cpu"))

    config = load_config("config.yaml")
    _ensure_checkpoint_dirs()

    logger.info("Loading training data...")
    data = load_training_data(config)
    logger.info("Loaded %s total rows", sum(len(df) for df in data.values()))

    if args.mode == "quick":
        tft_epochs = 10
        lstm_epochs = 5
        rl_timesteps = 50000
    else:
        tft_epochs = args.epochs
        lstm_epochs = max(args.epochs // 2, 1)
        rl_timesteps = int(config.get("deep_learning", {}).get("rl_agent", {}).get("total_timesteps", 500000))

    models = {
        "tft": TemporalFusionTransformer(config),
        "lstm_cnn": LSTMCNNHybrid(config),
        "rl_agent": PPOTradingAgent(config),
    }

    if args.mode == "resume":
        maybe_resume(models, args.checkpoint)

    logger.info("\n%s", "=" * 60)
    logger.info("PHASE 1: Training Temporal Fusion Transformer")
    logger.info("%s", "=" * 60)
    models["tft"] = train_tft(models["tft"], data, tft_epochs)

    logger.info("\n%s", "=" * 60)
    logger.info("PHASE 2: Training LSTM-CNN Hybrid")
    logger.info("%s", "=" * 60)
    models["lstm_cnn"] = train_lstm_cnn(models["lstm_cnn"], data, lstm_epochs)

    logger.info("\n%s", "=" * 60)
    logger.info("PHASE 3: Training RL Agent")
    logger.info("%s", "=" * 60)
    models["rl_agent"] = train_rl_agent(
        models["rl_agent"],
        data,
        models["tft"],
        models["lstm_cnn"],
        rl_timesteps,
    )

    logger.info("\n%s", "=" * 60)
    logger.info("Training Complete! Models saved to checkpoints/")
    logger.info("%s", "=" * 60)


if __name__ == "__main__":
    main()
