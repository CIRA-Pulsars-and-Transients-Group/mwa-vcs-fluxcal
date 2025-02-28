########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from healpy import read_map
from healpy.pixelfunc import get_interp_val
from mwalib import MetafitsContext
from scipy.interpolate import CubicSpline

import mwa_vcs_fluxcal
from mwa_vcs_fluxcal import RCVR_TEMP_FILE, SKY_TEMP_MAP_FILE

__all__ = ["splineSkyTempAtCoord", "getSkyTempAtCoords", "splineRecieverTemp", "getAmbientTemp"]


def splineSkyTempAtCoord(
    context: MetafitsContext,
    coord: SkyCoord,
    sky_index: float = -2.55,
    logger: logging.Logger | None = None,
) -> CubicSpline:
    """Estimate the sky temperature at a given coordinate, provided a metafits contenxt
    for frequency information. Returns a CubicSpline for interpolation across the observed
    frequency band.

    :param context: A mwalib.MetafitsContext object that contains the
                    array configuration and more importantly, frequency selections.
    :type context: MetafitsContext
    :param coord: An astropy SkyCoord object for the desired sky location
                  to retrive a sky temperature estimate from the Healpix map.
    :type coord: SkyCoord
    :param sky_index: The assumec spectral index of the sky temperature. Default: -2.55.
    :type sky_index: float

    :return: A cubic spline interpolation object based on the sampled data.
             Input into the spline object must be in MHz for correct temperatures in Kelvin.
    :rtype: CubicSpline
    """
    if logger is None:
        logger = mwa_vcs_fluxcal.get_logger()

    # read the sky temperature map
    sky_temp_map = read_map(SKY_TEMP_MAP_FILE)

    # convert ra, dec to Galactic coordinate
    logger.debug("Converting pointing coordinate to Galactic frame")
    gal = coord.transform_to("galactic")
    gl = gal.l.deg
    gb = gal.b.deg

    # retrieve sky temperature from sky map
    logger.info(f"Estimating Tsky at (l, b) = ({gl}, {gb}) deg")
    map_tsky = get_interp_val(sky_temp_map, gl, gb, nest=False, lonlat=True)

    # scale sky temperature to center frequency of the survey
    logger.info("Interpolating Tsky across observing bandwdith")
    map_freq_mhz = 408.0
    obs_freq_mhz = context.metafits_fine_chan_freqs_hz / 1e6
    freqs = np.linspace(min(obs_freq_mhz), max(obs_freq_mhz), 100)
    tsky_sample = map_tsky * (freqs / map_freq_mhz) ** sky_index

    # compute a cubic spline interpolation
    tsky_spline = CubicSpline(freqs, tsky_sample)

    return tsky_spline


def getSkyTempAtCoords(
    coords: SkyCoord,
    obs_freq_mhz: float,
    sky_index: float = -2.55,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Estimate the sky temperature at given coordinates, provided a metafits contenxt
    for frequency information. Returns the sky temperature at obs_freq_mhz for each
    coordinate.

    :param coords: An astropy SkyCoord object for the desired sky locations
                  to retrive a sky temperature estimate from the Healpix map.
    :type coord: SkyCoord
    :param sky_index: The assumec spectral index of the sky temperature. Default: -2.55.
    :type sky_index: float

    :return: The sky temperature at each coordinate interpolated to the given frequency.
    :rtype: ndarray
    """
    if logger is None:
        logger = mwa_vcs_fluxcal.get_logger()

    # read the sky temperature map
    sky_temp_map = read_map(SKY_TEMP_MAP_FILE)

    # convert ra, dec to Galactic coordinate
    logger.debug("Converting pointing coordinate to Galactic frame")
    gal = coords.transform_to("galactic")
    gl = gal.l.deg
    gb = gal.b.deg

    # retrieve sky temperature from sky map
    logger.debug("Estimating Tsky")
    map_tsky = get_interp_val(sky_temp_map, gl, gb, nest=False, lonlat=True)

    # scale sky temperature to center frequency of the survey
    logger.debug("Interpolating Tsky to provided frequency")
    map_freq_mhz = 408.0
    tsky_interp = map_tsky * (obs_freq_mhz / map_freq_mhz) ** sky_index

    return tsky_interp


def splineRecieverTemp() -> CubicSpline:
    """Get receiver temperature from the receiver temperature (Trcvr) file, returning
    a CubicSpline for interpolation across the full MWA band

    :return: A cubic spline interpolation object based on the sampled Trcvr data.
             Input into the spline object must be in MHz for correct temperatures in Kelvin.
    :rtype: CubicSpline
    """
    # The frequency data column is in MHz, and the rcvr temperatue column in Kelvin
    tab = Table.read(RCVR_TEMP_FILE, format="csv")
    trcvr_spline = CubicSpline(tab["freq"].value, tab["trec"].value)

    return trcvr_spline


def getAmbientTemp(metafits: str) -> u.Quantity:
    """Get the ambient temperature from a metafits file.

    Parameters
    ----------
    metafits : `str`
        The path to the metafits file.

    Returns
    -------
    mean_temp : `u.Quantity`
        The mean ambient tempertature. If the temperature cannot be found in the
        metafits then use the mean annual temperature of 22.4 deg C. The output
        Quantity is in units of Kelvin.
    """
    hdu_list = fits.open(metafits)
    temps_C = hdu_list[1].data["BFTemps"]
    print(temps_C)
    temps_C = temps_C[~np.isnan(temps_C)]
    print(temps_C)
    if len(temps_C) == 0:
        mean_temp = 22.4 * u.deg_C
    else:
        mean_temp = np.mean(temps_C) * u.deg_C
    return mean_temp.to(u.K)
