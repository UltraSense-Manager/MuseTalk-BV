"""
Central logging configuration. Import and call configure_logging() at app startup.
"""
import logging
import os
import sys

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv(
    "LOG_FORMAT",
    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)


def configure_logging() -> None:
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("main").setLevel(level)
    logging.getLogger("auth").setLevel(level)
    logging.getLogger("audio").setLevel(level)
    logging.getLogger("cloner").setLevel(level)
