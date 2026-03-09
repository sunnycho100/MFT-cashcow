"""Logging configuration for the crypto trading system."""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

import yaml


_loggers: dict[str, logging.Logger] = {}


def setup_logger(
    config_path: str = "config.yaml",
    name: str = "crypto_trader",
) -> logging.Logger:
    """Set up and return a configured logger.

    Args:
        config_path: Path to the YAML configuration file.
        name: Logger name.

    Returns:
        Configured logging.Logger instance.
    """
    # Load config
    log_cfg = {
        "level": "INFO",
        "file": "./logs/crypto_trader.log",
        "max_bytes": 10485760,
        "backup_count": 5,
        "format": "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    }

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            if cfg and "logging" in cfg:
                log_cfg.update(cfg["logging"])

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_cfg["level"].upper(), logging.INFO))

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_cfg["format"])
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    log_path = Path(log_cfg["file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        str(log_path),
        maxBytes=log_cfg["max_bytes"],
        backupCount=log_cfg["backup_count"],
    )
    file_handler.setLevel(getattr(logging, log_cfg["level"].upper(), logging.INFO))
    file_formatter = logging.Formatter(log_cfg["format"])
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


def get_logger(name: str = "crypto_trader") -> logging.Logger:
    """Get an existing logger or create a child logger.

    Args:
        name: Logger name (dot-separated for hierarchy).

    Returns:
        logging.Logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    # Create child of root crypto_trader logger
    base = _loggers.get("crypto_trader")
    if base is None:
        base = setup_logger()

    logger = logging.getLogger(name)
    _loggers[name] = logger
    return logger
