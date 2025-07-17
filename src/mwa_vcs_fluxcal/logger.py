########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging

__all__ = ["log_levels", "setup_logger"]

log_levels = dict(
    DEBUG=logging.DEBUG,
    INFO=logging.INFO,
    WARNING=logging.WARNING,
    ERROR=logging.ERROR,
    CRITICAL=logging.CRITICAL,
)


# TODO: Make docstring
def setup_logger(name: str | None = None, log_level: str | int = "INFO") -> None:
    if type(log_level) is str:
        # Convert from str to int representation
        log_level = log_levels.get(log_level.upper())

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
