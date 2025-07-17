########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging

import numpy as np
import psrchive
from scipy import integrate

import mwa_vcs_fluxcal

__all__ = ["get_profile_from_archive", "get_offpulse_region", "get_snr_profile"]

logger = logging.getLogger(__name__)


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


def get_offpulse_region(data: np.ndarray, windowsize: int | None = None) -> np.ndarray[bool]:
    """Determine the off-pulse window by minimising the integral over a range.
    i.e., because noise should integrate towards zero, finding the region that
    minimises the area mean it is representative of the noise level.

    Method taken from PyPulse (Lam, 2017. https://ascl.net/1706.011).

    Parameters
    ----------
    data : `np.ndarray`
        The original pulse profile.
    windowsize : `np.ndarray`, optional
        Window width (in bins) defining the trial regions to integrate. Default: `None`.

    Returns
    -------
    offpulse_win : `np.ndarray`
        An array of bins corresponding to the off-pulse region.
    offpulse_mask : `np.ndarray[bool]`
        A mask which is True for offpulse bins and False for onpulse bins.
    """
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


def get_snr_profile(
    archive: psrchive.Archive,
    noise_archive: str | None = None,
    windowsize: int | None = None,
    plot_profile: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the S/N profile from a PSRCHIVE archive.

    Parameters
    ----------
    archive : `psrchive.Archive`
        The detection archive.
    noise_archive : `psrchive.Archive`, optional
        An archive of a fully dispersed observation. Default: `None`.
    windowsize : `int`, optional
        The window size (in bins) to use to find the offpulse. Default: `None`.
    plot_profile : `bool`, optional
        Plot the pulse profile. Default: `False`.

    Returns
    -------
    snr_profile : `np.ndarray`
        The integrated pulse profile in S/N units.
    std_noise : `np.ndarray`
        The standard deviation of the offpulse noise in the uncalibrated profile.
    """
    # Get the Stokes I profile as a numpy array
    profile = mwa_vcs_fluxcal.get_profile_from_archive(archive)

    # Get the offpulse region of the profile
    op_idx, op_mask = mwa_vcs_fluxcal.get_offpulse_region(profile, windowsize=windowsize)
    offpulse = profile[op_mask]

    if noise_archive is not None:
        # Get the noise as a numpy array
        noise = mwa_vcs_fluxcal.get_profile_from_archive(noise_archive)
    else:
        noise = offpulse

    # Correct the profile baseline
    profile -= np.mean(offpulse)
    offpulse -= np.mean(offpulse)

    # Convert the profile to S/N
    std_noise = np.std(noise)
    snr_profile = profile / std_noise

    if noise_archive is not None:
        noise_snr_profile = noise / np.std(noise)
    else:
        noise_snr_profile = None

    if plot_profile:
        mwa_vcs_fluxcal.plot_pulse_profile(
            snr_profile,
            noise_profile=noise_snr_profile,
            offpulse_win=op_idx,
            offpulse_std=1,
            ylabel="S/N",
        )

    return snr_profile, std_noise
