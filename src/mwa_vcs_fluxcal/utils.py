########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import builtins
import logging
from typing import Any

import numpy as np
import psrchive
import rtoml
from astropy.coordinates import Angle, Latitude, Longitude, SkyCoord
from astropy.units import Quantity

__all__ = [
    "read_archive",
    "log_nan_zeros",
    "pythonise",
    "qty_dict_to_toml",
    "get_flux_density_uncertainty",
]

logger = logging.getLogger(__name__)


def read_archive(
    filename: str,
    bscrunch: int | None = None,
    subtract_baseline: bool = True,
    dedisperse: bool = True,
) -> psrchive.Archive:
    """Read a PSRCHIVE Archive, check that the data is in Stokes format,
    dedisperse, and subtract the baseline.

    Parameters
    ----------
    filename : `str`
        The path to the archive file to load.
    bscrunch : `int`, optional
        Bscrunch to this number of phase bins. Default: None.
    subtract_baseline : `bool`, optional
        Subtract the baseline. Default: True.
    dedisperse : `bool`, optional
        Apply channel delays to correct for dispersion. Default: True.

    Returns
    -------
    archive : `psrchive.Archive`
        The data stored in an Archive object.
    """
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

    if not archive.get_dedispersed() and dedisperse:
        try:
            archive.dedisperse()
            logger.debug("Successfully dedispersed.")
        except RuntimeError:
            logger.error("Could not dedisperse.")

    if subtract_baseline:
        try:
            archive.remove_baseline()
            logger.debug("Successfully removed baseline.")
        except RuntimeError:
            logger.error("Could not remove baseline.")

    logger.info(f"Profile has {archive.get_nbin()} bins")
    if type(bscrunch) is int:
        if bscrunch < archive.get_nbin():
            logger.info(f"Averaging to {bscrunch} bins")
            archive.bscrunch_to_nbin(bscrunch)

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


def pythonise(input: Any) -> Any:
    """Convert numpy types to builtin types using recursion.

    Parameters
    ----------
    input : `Any`
        A number, iterator, or dictionary.

    Returns
    -------
    output : `Any`
        A number, iterator, or dictionary containing only builtin types.
    """
    match type(input):
        case np.bool_:
            output = bool(input)
        case np.int_ | np.int32:
            output = int(input)
        case np.float_ | np.float32:
            output = float(input)
        case np.str_:
            output = str(input)
        case builtins.tuple:
            output = tuple(pythonise(item) for item in input)
        case builtins.list:
            output = [pythonise(item) for item in input]
        case builtins.dict:
            output = {key: pythonise(val) for (key, val) in input.items()}
        case np.ndarray:
            output = pythonise(input.tolist())
        case _:
            output = input
    return output


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
