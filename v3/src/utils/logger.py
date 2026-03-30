"""Lightweight structured logger for v3."""

import logging
import sys

try:
    from loguru import logger as _logger
except ModuleNotFoundError:
    _logger = None


class StdlibLoggerAdapter:
    """Small compatibility adapter when loguru is unavailable."""

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def info(self, message: str, *args) -> None:
        self._logger.info(message.format(*args))

    def warning(self, message: str, *args) -> None:
        self._logger.warning(message.format(*args))

    def error(self, message: str, *args) -> None:
        self._logger.error(message.format(*args))


if _logger is not None:
    _logger.remove()
    _logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <cyan>{name}</cyan> | {message}",
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def get_logger(name: str):
    """Return a named logger."""
    if _logger is not None:
        return _logger.bind(name=name)
    return StdlibLoggerAdapter(name)
