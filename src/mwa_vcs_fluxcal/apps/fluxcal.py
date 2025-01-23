import logging

import click
import numpy as np
import psrchive

import mwa_vcs_fluxcal


@click.command()
@click.argument("archive", nargs=1, type=click.Path(exists=True))
@click.option("-w", "windowsize", type=int, help="Window size to use to find the offpulse.")
def main(archive: str, windowsize: int) -> None:
    logger = mwa_vcs_fluxcal.get_logger(log_level=logging.INFO)

    logger.info(f"Loading archive: {archive}")
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

    # Generate a plot of the pulse profile with useful statistics
    mwa_vcs_fluxcal.plot_pulse_profile(
        profile,
        offpulse_win,
        offpulse_sigma,
        snr,
        logger=logger,
    )


if __name__ == "__main__":
    main()
