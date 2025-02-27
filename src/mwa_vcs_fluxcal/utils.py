########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging

import numpy as np
import psrchive

import mwa_vcs_fluxcal

__all__ = ["read_archive", "log_nan_zeros"]


def read_archive(filename: str, logger: logging.Logger = None) -> psrchive.Archive:
    """Read a PSRCHIVE Archive, check that the data is in Stokes format,
    dedisperse, and subtract the baseline.

    Parameters
    ----------
    filename : `str`
        The path to the archive file to load.
    logger : `logging.Logger`, optional
        A logger to use. Default: None.

    Returns
    -------
    archive : `psrchive.Archive`
        The data stored in an Archive object.
    """
    if logger is None:
        logger = mwa_vcs_fluxcal.get_logger()

    logger.info(f"Loading archive: {filename}")
    try:
        archive = psrchive.Archive_load(filename)
    except AttributeError:
        archive = psrchive.Archive.load(filename)

    if archive.get_state() != "Stokes" and archive.get_npol() == 4:
        try:
            archive.convert_state("Stokes")
            logger.debug("Successfully converted to Stokes.")
        except RuntimeError:
            logger.error("Could not convert to Stokes.")

    if not archive.get_dedispersed():
        try:
            archive.dedisperse()
            logger.debug("Successfully dedispersed.")
        except RuntimeError:
            logger.error("Could not dedisperse.")

    try:
        archive.remove_baseline()
        logger.debug("Successfully removed baseline.")
    except RuntimeError:
        logger.error("Could not remove baseline.")

    return archive


def log_nan_zeros(arr: np.ndarray) -> np.ndarray:
    """Compute the base 10 logarithm of the input array, but input array elements
    that are equal to zero are replaced with `np.nan`.

    Parameters
    ----------
    arr : `np.ndarray`
        A numpy array.

    Returns
    -------
    log_nan_arr : `np.ndarray`
        The log of the input array with zeros replaced by `np.nan`.
    """
    return np.log10(np.where(arr > 0, arr, np.nan))
