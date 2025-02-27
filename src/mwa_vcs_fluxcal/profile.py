########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging

import numpy as np
import psrchive
from scipy import integrate

import mwa_vcs_fluxcal

__all__ = ["get_profile_from_archive", "get_offpulse_region"]


def get_profile_from_archive(archive: psrchive.Archive) -> np.ndarray:
    """Get the Stokes I profile from a PSRCHIVE Archive object.

    Parameters
    ----------
    archive : psrchive.Archive
        An archive in Stokes I format.

    Returns
    -------
    profile : `np.ndarray`
        The Stokes I profile.
    """
    tmp_archive = archive.clone()
    tmp_archive.fscrunch()
    tmp_archive.tscrunch()
    tmp_archive.pscrunch()
    profile = tmp_archive.get_data()[0, 0, 0, :]
    return profile


def get_offpulse_region(
    data: np.ndarray, windowsize: int | None = None, logger: logging.Logger | None = None
) -> np.ndarray[bool]:
    """Determine the off-pulse window by minimising the integral over a range.
    i.e., because noise should integrate towards zero, finding the region that
    minimises the area mean it is representative of the noise level.

    Method taken from PyPulse (Lam, 2017. https://ascl.net/1706.011).

    Parameters
    ----------
    data : `np.ndarray`
        The original pulse profile.
    windowsize : `np.ndarray`, optional
        Window width (in bins) defining the trial regions to integrate. Default: `None`
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.

    Returns
    -------
    offpulse_win : `np.ndarray`
        An array of bins corresponding to the off-pulse region.
    offpulse_mask : `np.ndarray[bool]`
        A mask which is True for offpulse bins and False for onpulse bins.
    """
    if logger is None:
        logger = mwa_vcs_fluxcal.get_logger()

    nbins = len(data)

    if windowsize is None:
        logger.debug("No off-pulse window size set, assuming 1/8 of profile.")
        windowsize = nbins // 8

    integral = np.zeros_like(data)
    for i in range(nbins):
        win = np.arange(i - windowsize // 2, i + windowsize // 2) % nbins
        integral[i] = integrate.trapezoid(data[win])

    minidx = np.argmin(integral)
    offpulse_win = np.arange(minidx - windowsize // 2, minidx + windowsize // 2) % nbins

    offpulse_mask = np.full(data.size, False, dtype=bool)
    for bin_idx in offpulse_win:
        offpulse_mask[bin_idx] = True

    return offpulse_win, offpulse_mask
