#!/usr/bin/env python

########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################
import logging
import pkg_resources
from astropy.coordinates import SkyCoord
from scipy.interpolate import CubicSpline
from healpy import read_map
from healpy.pixelfunc import get_interp_val
from mwalib import MetafitsContext

import mwa_vcs_fluxcal


logger = mwa_vcs_fluxcal.get_logger()

# Get package data files (mildly hacky?)
SKY_TEMP_MAP = read_map(pkg_resources.resource_filename("mwa_vcs_fluxcal", "data/haslam408_ds_Remazeilles2014.fits"), dtype=None, verbose=False)
RCVR_TEMP_FILE = pkg_resources.resource_filename("mwa_vcs_fluxcal", "data/MWA_Trcvr_tile_56.csv")

def splineSkyTempAtCoord(context: MetafitsContext, coord: SkyCoord, sky_index: float = -2.55) -> CubicSpline:
    """Estimate the sky temperature at a given coordinate, provided a metafits contenxt 
    for frequency information. Returns a CubicSpline for interpolation across the observed frequency band.
    
    :param context: A mwalib.MetafitsContext object that contains the
                    array configuration and more importantly, frequency selections.
    :type context: MetafitsContext
    :param coord: An astropy SkyCoord object for the desired sky location
                  to retrive a sky temperature estimate from the Healpix map.
    :type coord: SKyCoord
    :param sky_index: The assumec spectral index of the sky temperature. Default: -2.55.
    :type sky_index: float

    :return: A cubic spline interpolation object based on the sampled data.
             Input into the spline object must be in MHz for correct temperatures in Kelvin.
    :rtype: CubicSpline
    """ 
    # convert ra, dec to Galactic coordinate
    logger.info("Converting pointing coordinate to Galactic frame")
    gal = coord.transform_to("galactic")
    gl = gal.l.deg
    gb = gal.b.deg

    # retrieve sky temperature from sky map
    logger.info(f"Estimating Tsky at (l, b) = ({gl}, {gb}) deg")
    map_tsky = get_interp_val(
        SKY_TEMP_MAP, gl, gb, nest=False, lonlat=True
    )

    # scale sky temperature to center frequency of the survey
    logger.info("Interpolating Tsky across observing bandwdith")
    map_freq_mhz = 408.0
    # TODO: update this to inspect context to figure our the frequency span
    freqs = np.linspace(140, 170, 100)
    tsky_sample = map_tsky * (freqs / map_freq_mhz) ** sky_index

    # compute a cubic spline interpolation
    tsky_spline = CubicSpline(freqs, tsky_sample)

    return tsky_spline


def splineRecieverTemp(context: MetafitsContext):
    """Get receiver temperature from the receiver temperature (Trcvr) file, returning
    a CubicSpline for interpolation across the full MWA band

    :param context: A mwalib.MetafitsContext object that contains the
                    array configuration and more importantly, frequency selections.
    :type context: MetafitsContext

    :return: A cubic spline interpolation object based on the sampled Trcvr data.
             Input into the spline object must be in MHz for correct temperatures in Kelvin.
    :rtype: CubicSpline
    """
    # The frequency data column is in MHz, and the rcvr temperatue column in Kelvin
    tab = Table.read(RCVR_TEMP_FILE, format="csv")
    trcvr_spline = CubicSpline(tab["freq"].value, tab["trec"].value)

    return trcvr_spline

