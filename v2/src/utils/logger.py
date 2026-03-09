"""Lightweight structured logger for v2."""

import sys
from loguru import logger as _logger

# Remove default handler
_logger.remove()

# Console handler: INFO+
_logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <cyan>{name}</cyan> | {message}",
)


def get_logger(name: str):
    """Return a named logger."""
    return _logger.bind(name=name)
