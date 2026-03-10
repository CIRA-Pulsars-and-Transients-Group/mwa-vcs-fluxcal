########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging
from typing import cast

__all__ = ["log_levels", "setup_logger"]

log_levels: dict[str, int] = dict(
    DEBUG=logging.DEBUG,
    INFO=logging.INFO,
    WARNING=logging.WARNING,
    ERROR=logging.ERROR,
    CRITICAL=logging.CRITICAL,
)


def setup_logger(name: str | None = None, log_level: str | int = "INFO") -> None:
    """Clear all previous handlers and add a new custom stream handler.

    Parameters
    ----------
    name : str or None, default: None
        The name of the logger. Note that `None` returns the root logger.
    log_level : str or int, default: "INFO"
        The name of the logging level or the effective logging level.
    """
    # If a string was provided, map it to the effective level
    if isinstance(log_level, str):
        log_level = log_levels[log_level.upper()]
    cast(int, log_level)

    # Get the logger
    logger = logging.getLogger(name)

    # Remove any previous handlers
    logger.handlers.clear()

    # Set the verbosity level of the logger
    logger.setLevel(log_level)

    # Get channel handler
    ch = logging.StreamHandler()

    # Set the verbosity level of ch
    ch.setLevel(log_level)

    # Set the formatter of ch
    formatter = logging.Formatter(
        fmt="[%(asctime)s %(levelname)-8s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)

    # Add ch to logger
    logger.addHandler(ch)

    # Do not propagate to other packages
    logger.propagate = False
