########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import click
import mwalib
import numpy as np
import psrchive

import mwa_vcs_fluxcal

"""
The flux density can be estimated using the radiometer equation:

              f_c T_sys
S = (S/N) x ---------------
            G sqrt(n dv dt)

where

S/N   = signal-to-noise ratio in pulse profile
f_c   = coherency factor
T_sys = system temperature
G     = system gain
n     = number of instrumental polarisations summed (assume 2)
dv    = observing bandwidth
dt    = integration time

The system temperature consists of contributions from the antennas (T_ant),
the receiver (T_rec), and the environment (T_0 ~ 290 K):

T_sys = eta T_ant + (1 - eta) T_0 + T_rec

where eta is the radiation efficiency of the array. The antenna temperature is
an integral of the antenna pattern and the sky temperature (T_sky) over the
solid angle of the beam and the frequency band.

To calculate G, we must first calculate the effective collecting area A_e:

            4 pi lambda^2
A_e = eta * -------------
               Omega_A

where Omega_A is the beam solid angle -- the integral of the array factor
power pattern over the sky.

The gain is related to the effective area and Boltzmann's constant k_B as:

     A_e
G = -----
    2 k_B

For further details, see Meyers et al. (2017):
https://ui.adsabs.harvard.edu/abs/2017ApJ...851...20M/abstract
"""


@click.command()
@click.argument("archive", nargs=1, type=click.Path(exists=True))
@click.option(
    "-L",
    "log_level",
    type=click.Choice(["DEBUG", "INFO", "ERROR", "CRITICAL"], case_sensitive=False),
    default="INFO",
    help="The logger verbosity level.",
)
@click.option("-m", "metafits", type=click.Path(exists=True), help="An MWA metafits file.")
@click.option("-w", "windowsize", type=int, help="Window size to use to find the offpulse.")
@click.option("--plot_profile", is_flag=True, help="Plot the pulse profile.")
@click.option("--plot_trec", is_flag=True, help="Plot the receiver temperature.")
def main(
    archive: str,
    log_level: str,
    metafits: str,
    windowsize: int,
    plot_profile: bool,
    plot_trec: bool,
) -> None:
    log_level_dict = mwa_vcs_fluxcal.get_log_levels()
    logger = mwa_vcs_fluxcal.get_logger(log_level=log_level_dict[log_level])

    logger.info(f"Loading archive: {archive}")
    try:
        archive = psrchive.Archive_load(archive)
    except AttributeError:
        archive = psrchive.Archive.load(archive)

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

    # Get the Stokes I profile as a numpy array
    profile = mwa_vcs_fluxcal.get_profile_from_archive(archive)

    # Get a list of bin indices for the offpulse region
    offpulse_win = mwa_vcs_fluxcal.get_offpulse_region(
        profile, windowsize=windowsize, logger=logger
    )

    # Convert the bin indices to a mask
    mask = np.full(profile.shape[0], False)
    for bin_idx in offpulse_win:
        mask[bin_idx] = True

    # Compute the signal/noise ratio
    offpulse_sigma = np.std(profile[mask])
    snr = np.max(profile) / offpulse_sigma
    logger.info(f"S/N = {snr}")

    if plot_profile:
        mwa_vcs_fluxcal.plot_pulse_profile(
            profile,
            offpulse_win,
            offpulse_sigma,
            snr,
            logger=logger,
        )

    if metafits is None:
        logger.info("No metafits file provided. Exiting.")
        exit(0)

    # Prepare metadata
    logger.info(f"Loading metafits: {metafits}")
    context = mwalib.MetafitsContext(metafits)

    # Get bandwidth and integration time from archive
    fctr = archive.get_centre_frequency()
    logger.info(f"fctr = {fctr} MHz")
    df = archive.get_bandwidth()
    logger.info(f"df = {df} MHz")
    t0 = archive.get_first_Integration().get_start_time()
    t1 = archive.get_last_Integration().get_end_time()
    dt = (t1 - t0).in_seconds()
    logger.info(f"dt = {dt} s")

    # Get T_rec
    trcvr_spline = mwa_vcs_fluxcal.splineRecieverTemp(context)
    if plot_trec:
        mwa_vcs_fluxcal.plot_trcvr_vc_freq(trcvr_spline, fctr, df, logger=logger)

    # Calculate T_ant

    # Calculate T_sys

    # Calculate G

    # Radiometer equation


if __name__ == "__main__":
    main()
