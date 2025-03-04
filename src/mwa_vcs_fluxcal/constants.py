########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import pkg_resources
from astropy import units as u
from astropy.coordinates import EarthLocation

__all__ = [
    "MWA_CENTRE_LON",
    "MWA_CENTRE_LAT",
    "MWA_CENTRE_H",
    "MWA_CENTRE_CABLE_LEN",
    "MWA_LOCATION",
    "SKY_TEMP_MAP_FILE",
    "RCVR_TEMP_FILE",
    "SI_TO_JY",
    "fc",
    "eta",
    "npol",
]

# MWA telescope location
MWA_CENTRE_LON = 116.67081524 * u.deg
MWA_CENTRE_LAT = -26.70331940 * u.deg
MWA_CENTRE_H = 377.8269 * u.m
MWA_CENTRE_CABLE_LEN = 0.0 * u.m
MWA_LOCATION = EarthLocation.from_geodetic(
    lon=MWA_CENTRE_LON, lat=MWA_CENTRE_LAT, height=MWA_CENTRE_H
)

# Get package data files (mildly hacky?)
SKY_TEMP_MAP_FILE = pkg_resources.resource_filename(
    "mwa_vcs_fluxcal", "data/haslam408_ds_Remazeilles2014.fits"
)
RCVR_TEMP_FILE = pkg_resources.resource_filename("mwa_vcs_fluxcal", "data/MWA_Trcvr_tile_56.csv")

# Jansky conversion factor
SI_TO_JY = 10**26 * u.Jy / (u.W * u.m**-2 * u.Hz**-1)

# Assumptions
fc = 0.7
eta = 0.98
npol = 2
