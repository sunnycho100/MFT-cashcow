"""Lightweight structured logger for v3."""

import sys

from loguru import logger as _logger

_logger.remove()
_logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <cyan>{name}</cyan> | {message}",
)


def get_logger(name: str):
    """Return a named logger."""
    return _logger.bind(name=name)
