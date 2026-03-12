########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging
from typing import Any

import numpy as np
import rtoml
from astropy.coordinates import Angle, Latitude, Longitude
from astropy.units import Quantity

__all__ = [
    "log_nan_zeros",
    "pythonise",
    "qty_dict_to_toml",
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


def pythonise(var: Any) -> Any:
    """Convert numpy types to builtin types.

    Parameters
    ----------
    var : Any
        A number or iterator.

    Returns
    -------
    Any
        The input variable cast into builtin types.
    """
    match var:
        case np.bool_():
            output = bool(var)
        case np.integer():
            output = int(var)
        case np.floating():
            output = float(var)
        case np.str_():
            output = str(var)
        case tuple():
            output = tuple(pythonise(item) for item in var)
        case list():
            output = [pythonise(item) for item in var]
        case dict():
            output = {key: pythonise(val) for (key, val) in var.items()}
        case np.ndarray():
            output = pythonise(var.tolist())
        case _:
            output = var
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
            vals_dict[key] = [
                pythonise(qty_dict[key].value),
                qty_dict[key].unit.to_string(),
            ]
        elif type(qty_dict[key]) is str:
            vals_dict[key] = [pythonise(qty_dict[key]), "string"]
        else:
            vals_dict[key] = [pythonise(qty_dict[key]), "unitless"]
    with open(savename, "w") as f:
        rtoml.dump(vals_dict, f)
