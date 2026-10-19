"""Logging conventions for command-line tools and future API services."""

from __future__ import annotations

import logging


LOGGER_NAME = "kline_retrieval"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: int | str = logging.INFO) -> logging.Logger:
    """Configure one stderr handler and return the project logger."""

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    return logger
