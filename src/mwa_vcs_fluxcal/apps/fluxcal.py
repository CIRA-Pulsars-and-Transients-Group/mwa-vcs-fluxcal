########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

# This script is an implementation of the flux density calibration
# method described in Meyers et al. (2017), hereafter abbreviated M+17.
# https://ui.adsabs.harvard.edu/abs/2017ApJ...851...20M/abstract

# TODO: Get the additional flagged tiles from the calibration solution
# TODO: Work out the amount of flagged data from the archive

import astropy.units as u
import click
import mwalib
import numpy as np
from astropy.constants import c
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from scipy import integrate

import mwa_vcs_fluxcal
from mwa_vcs_fluxcal import npol


@click.command()
@click.argument("archive", nargs=1, type=click.Path(exists=True))
@click.option(
    "-L",
    "log_level",
    type=click.Choice(list(mwa_vcs_fluxcal.get_log_levels()), case_sensitive=False),
    default="INFO",
    help="The logger verbosity level.",
)
@click.option("-m", "metafits", type=click.Path(exists=True), help="An MWA metafits file.")
@click.option("-w", "windowsize", type=int, help="Window size to use to find the offpulse.")
@click.option(
    "--fine_res", type=float, default=1, help="The resolution of the integral, in arcmin."
)
@click.option(
    "--coarse_res", type=float, default=30, help="The resolution of the primary beam map, in arcmin"
)
@click.option(
    "--min_pbp", type=float, default=0.001, help="Only integrate above this primary beam power."
)
@click.option("--nfreq", type=int, default=1, help="The number of frequency steps to evaluate.")
@click.option("--ntime", type=int, default=1, help="The number of time steps to evaluate.")
@click.option("--plot_profile", is_flag=True, help="Plot the pulse profile.")
@click.option("--plot_trec", is_flag=True, help="Plot the receiver temperature.")
@click.option("--plot_pb", is_flag=True, help="Plot the primary beam.")
@click.option("--plot_images", is_flag=True, help="Plot visualisations of the integral quantities.")
@click.option("--plot_3d", is_flag=True, help="Plot the results in 3D (time,freq,data).")
def main(
    archive: str,
    log_level: str,
    metafits: str,
    windowsize: int,
    fine_res: float,
    coarse_res: float,
    min_pbp: float,
    nfreq: int,
    ntime: int,
    plot_profile: bool,
    plot_trec: bool,
    plot_pb: bool,
    plot_images: bool,
    plot_3d: bool,
) -> None:
    log_level_dict = mwa_vcs_fluxcal.get_log_levels()
    logger = mwa_vcs_fluxcal.get_logger(log_level=log_level_dict[log_level])

    # Load, dedisperse, and baseline-subtract archive
    archive = mwa_vcs_fluxcal.read_archive(archive, logger=logger)

    # Get the Stokes I profile as a numpy array
    profile = mwa_vcs_fluxcal.get_profile_from_archive(archive)

    # Get the offpulse region of the profile
    op_idx, op_mask = mwa_vcs_fluxcal.get_offpulse_region(
        profile, windowsize=windowsize, logger=logger
    )
    offpulse = profile[op_mask]

    # Correct the profile baseline
    profile -= np.mean(offpulse)

    # Convert the profile to S/N
    snr_profile = profile / np.std(offpulse)

    if plot_profile:
        mwa_vcs_fluxcal.plot_pulse_profile(
            snr_profile,
            op_idx,
            offpulse_std=1,
            ylabel="S/N",
            logger=logger,
        )

    if metafits is None:
        logger.info("No metafits file provided. Exiting.")
        exit(0)

    # Prepare metadata
    logger.info(f"Loading metafits: {metafits}")
    context = mwalib.MetafitsContext(metafits)
    T_amb = mwa_vcs_fluxcal.getAmbientTemp(metafits)

    # Get frequency and time metadata from archive
    fctr = archive.get_centre_frequency() * u.MHz
    bw = archive.get_bandwidth() * u.MHz
    f0 = fctr - bw / 2
    f1 = fctr + bw / 2
    t0 = archive.get_first_Integration().get_start_time()
    t1 = archive.get_last_Integration().get_end_time()
    dt = (t1 - t0).in_seconds() * u.s
    start_time = Time(t0.in_days(), format="mjd")
    logger.info(f"Centre frequency = {fctr.to_string()}")
    logger.info(f"Bandwidth = {bw.to_string()}")
    logger.info(f"Integration time = {dt.to_string()}")
    logger.info(f"Start MJD = {start_time.to_string()}")

    # Calculate which freqs/times to evalue the integral at
    if nfreq == 1:
        eval_freqs = np.array([fctr.value]) * u.MHz
    else:
        eval_freqs = np.linspace(f0.value, f1.value, nfreq) * u.MHz
    if ntime == 1:
        eval_offsets = np.array([dt.value / 2]) * u.s
    else:
        eval_offsets = np.linspace(0, dt.value, ntime) * u.s
    eval_times = start_time + eval_offsets
    logger.info(f"Evaluating at freqs: {eval_freqs}")
    logger.info(f"Evaluating at offsets: {eval_offsets}")
    logger.info(f"Evaluating at times: {eval_times}")

    # Calculate the beam width
    max_baseline, _, _ = mwa_vcs_fluxcal.find_max_baseline(context)
    max_baseline *= u.m
    width = ((c / fctr.to(1 / u.s)) / max_baseline) * u.rad
    logger.info(f"Maximum baseline: {max_baseline.to_string()}")
    logger.info(f"Beam width ~ lambda/D: {width.to(u.arcmin).to_string()}")

    # Define the grid resolutions
    fine_grid_res = Angle(fine_res, u.arcmin)
    coarse_grid_res = Angle(coarse_res, u.arcmin)
    logger.info(f"Fine grid resolution = {fine_grid_res.to_string()}")
    logger.info(f"Coarse grid resolution = {coarse_grid_res.to_string()}")
    if not np.isclose((coarse_grid_res.arcmin / fine_grid_res.arcmin) % 1, 0.0, rtol=1e-5):
        logger.critical("Coarse grid resolution not divisible by fine grid resolution.")
        exit(1)

    # Create a SkyCoord object defining the RA/Dec of the pulsar
    ra_hms, dec_dms = archive.get_coordinates().getHMSDMS().split(" ")
    pulsar_coords = SkyCoord(ra_hms, dec_dms, frame="icrs", unit=("hourangle", "deg"))

    if plot_trec:
        # Plot the receiver temperature vs frequency
        T_rec_spline = mwa_vcs_fluxcal.splineRecieverTemp()
        mwa_vcs_fluxcal.plot_trcvr_vc_freq(
            T_rec_spline, fctr.to(u.MHz).value, bw.to(u.MHz).value, logger=logger
        )

    # Compute the sky integrals required to get T_sys and gain
    inputs, results = mwa_vcs_fluxcal.compute_sky_integrals(
        context,
        start_time,
        eval_offsets,
        eval_freqs,
        pulsar_coords,
        fine_grid_res,
        coarse_grid_res,
        min_pbp,
        plot_pb=plot_pb,
        plot_images=plot_images,
        T_amb=T_amb,
        logger=logger,
    )

    if plot_3d and nfreq >= 4 and ntime >= 4:
        # Fit a 2D spline to show the freq/time scaling of T_sys, gain, and SEFD
        mwa_vcs_fluxcal.plot_3d_result(
            eval_offsets.to(u.s).value,
            eval_freqs.to(u.MHz).value,
            results["T_sys"].value,
            zlabel="$T_\mathrm{sys}$ [K]",
            savename="3d_tsys.png",
            logger=logger,
        )
        mwa_vcs_fluxcal.plot_3d_result(
            eval_offsets.to(u.s).value,
            eval_freqs.to(u.MHz).value,
            results["G"].value,
            zlabel="Gain [K/Jy]",
            savename="3d_gain.png",
            logger=logger,
        )
        mwa_vcs_fluxcal.plot_3d_result(
            eval_offsets.to(u.s).value,
            eval_freqs.to(u.MHz).value,
            results["SEFD"].value,
            zlabel="SEFD [Jy]",
            savename="3d_sefd.png",
            logger=logger,
        )

    # Radiometer equation (Eq 3 of M+17)
    dt_bin = dt / archive.get_nbin()
    radiometer_noise = results["SEFD_mean"] / np.sqrt(npol * bw.to(1 / u.s) * dt_bin)
    flux_density_profile = snr_profile * radiometer_noise
    S_peak = np.max(flux_density_profile)
    S_mean = integrate.trapezoid(flux_density_profile) / archive.get_nbin()
    logger.info(f"SEFD = {results['SEFD_mean'].to(u.Jy).to_string()}")
    logger.info(f"Peak flux density = {S_peak.to(u.mJy).to_string()}")
    logger.info(f"Mean flux density = {S_mean.to(u.mJy).to_string()}")

    # Add to the inputs dictionary
    inputs["beam_width"] = width.to(u.arcmin)

    # Add to the results dictionary
    results["noise_rms"] = radiometer_noise
    results["S_peak"] = S_peak
    results["S_mean"] = S_mean

    # Dump the dictionaries to toml files
    mwa_vcs_fluxcal.qty_dict_to_toml(inputs, "inputs.toml")
    mwa_vcs_fluxcal.qty_dict_to_toml(results, "results.toml")


if __name__ == "__main__":
    main()
