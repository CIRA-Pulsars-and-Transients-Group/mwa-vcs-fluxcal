########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging

import numpy as np
import rtoml
from astropy.coordinates import Angle, Latitude, Longitude, SkyCoord
from astropy.units import Quantity
from psrutils import StokesCube, pythonise

__all__ = [
    "log_nan_zeros",
    "pythonise",
    "qty_dict_to_toml",
    "get_flux_density_uncertainty",
    "get_offpulse_stats",
]

logger = logging.getLogger(__name__)


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


def qty_dict_to_toml(qty_dict: dict, savename="qty_dict.toml") -> None:
    """Write a dictionary of astropy Quantities to a TOML file.

    Parameters
    ----------
    qty_dict : `dict`
        A dictionary where each entry is an astropy Quantity.
    savename : `str`, optional
        The name of the file to write the TOML-encoded string to. Default: "qty_dict.toml".
    """
    vals_dict = dict()
    for key in qty_dict:
        # Store in [value, "unit"] format and ensure all types are native
        if type(qty_dict[key]) in [Quantity, Angle, Longitude, Latitude]:
            vals_dict[key] = [pythonise(qty_dict[key].value), qty_dict[key].unit.to_string()]
        elif type(qty_dict[key]) is str:
            vals_dict[key] = [pythonise(qty_dict[key]), "string"]
        else:
            vals_dict[key] = [pythonise(qty_dict[key]), "unitless"]
    with open(savename, "w") as f:
        rtoml.dump(vals_dict, f)


def get_flux_density_uncertainty(pulsar_coords: SkyCoord) -> float:
    """Get the relative uncertainty on the flux density, primarily accounting for the
    uncertainty on Tsky and the coherency factor (see Lee et al. 2025).

    Parameters
    ----------
    pulsar_coords : `SkyCoord`
        The coordinates of the pulsar.

    Returns
    -------
    relative_uncertainty : `float`
        The relative uncertainty on the flux density.
    """
    pulsar_lat = pulsar_coords.galactic.l.deg
    if abs(pulsar_lat) < 10:
        return 0.4
    else:
        return 0.3


# TODO: Make docstring
def get_offpulse_stats(
    cube: StokesCube, noise_cube: StokesCube | None = None
) -> tuple[np.float_, np.float_]:
    if isinstance(noise_cube, StokesCube):
        # The mean and standard deviation of the dispersed profile
        offpulse_mean = np.mean(noise_cube.profile)
        offpulse_std = np.std(noise_cube.profile)
    else:
        # Get the profile as a SplineProfile object to use for analysis
        profile = cube.spline_profile

        # Find the onpulse using the spline method
        profile.bootstrap_onpulse_regions()

        # It is important that the baseline is not overestimated, so we use
        # a simple sliding-window method to find the baseline
        offpulse_mean = profile.get_simple_noise_stats()[0]

        # We use the standard deviation of the profile residuals to get an
        # estimate of the profile noise. This approach also works when
        # there is no offpulse region
        offpulse_std = np.std(profile.residuals)

        logger.info(f"Saving plot file: {cube.source}_profile_diagnostics.png")
        profile.plot_diagnostics(
            plot_underestimate=False,
            plot_overestimate=True,
            savename=f"{cube.source}_profile_diagnostics.png",
        )
    return offpulse_mean, offpulse_std
