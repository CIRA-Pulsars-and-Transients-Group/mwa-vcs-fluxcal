########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging

__all__ = ["get_log_levels", "get_logger"]


def get_log_levels() -> dict:
    """Get all available logger verbosity levels.

    Returns
    -------
    log_levels : `dict`
        A dictionary containing all available logger verbosity levels.
    """
    return dict(
        DEBUG=logging.DEBUG,
        INFO=logging.INFO,
        ERROR=logging.ERROR,
        CRITICAL=logging.CRITICAL,
    )


def get_logger(name: str = __name__, log_level: int = logging.INFO) -> logging.Logger:
    """Initialise the custom logger.

    Parameters
    ----------
    name : `str`, optional
        The name of the logger. Default: `__name__`.
    log_level : `int`, optional
        The logging level. Default: `logging.INFO`.

    Returns
    -------
    logger : `logging.Logger`
        The custom logger.
    """
    logger = logging.getLogger(name)
    formatter = logging.Formatter("[%(levelname)7s - %(asctime)s] %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.setLevel(log_level)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
