########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

# This script is an implementation of the flux density calibration
# method described in Meyers et al. (2017), hereafter abbreviated M+17.
# https://ui.adsabs.harvard.edu/abs/2017ApJ...851...20M/abstract

# TODO: Get the additional flagged tiles from the calibration solution

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
@click.option(
    "-L",
    "log_level",
    type=click.Choice(list(mwa_vcs_fluxcal.get_log_levels()), case_sensitive=False),
    default="INFO",
    show_default=True,
    help="The logger verbosity level.",
)
@click.option("-m", "metafits", type=click.Path(exists=True), help="An MWA metafits file.")
@click.option(
    "--target",
    type=str,
    help="The target's RA/Dec in hour/deg units in any format accepted by SkyCoord.",
)
@click.option(
    "-a",
    "archive",
    type=click.Path(exists=True),
    help="An archive file to use to compute the pulse profile.",
)
@click.option(
    "-n",
    "noise_archive",
    type=click.Path(exists=True),
    help="An archive file to use to compute the offpulse noise.",
)
@click.option("-b", "bscrunch", type=int, help="Bscrunch to this number of phase bins.")
@click.option("-w", "windowsize", type=int, help="Window size to use to find the offpulse.")
@click.option(
    "--fine_res",
    type=float,
    default=2,
    show_default=True,
    help="The resolution of the integral, in arcmin.",
)
@click.option(
    "--coarse_res",
    type=float,
    default=30,
    show_default=True,
    help="The resolution of the primary beam map, in arcmin.",
)
@click.option(
    "--min_pbp",
    type=float,
    default=0.001,
    show_default=True,
    help="Only integrate above this primary beam power.",
)
@click.option(
    "--nfreq",
    type=int,
    default=1,
    show_default=True,
    help="The number of frequency steps to evaluate.",
)
@click.option(
    "--ntime", type=int, default=1, show_default=True, help="The number of time steps to evaluate."
)
@click.option(
    "--max_pix_per_job",
    type=int,
    default=10**5,
    show_default=True,
    help="The maximum number of sky area pixels to compute per job.",
)
@click.option(
    "--bw_flagged",
    type=click.FloatRange(0.0, 1.0),
    default=0.0,
    show_default=True,
    help="The fraction of the bandwidth flagged.",
)
@click.option(
    "--time_flagged",
    type=click.FloatRange(0.0, 1.0),
    default=0.0,
    show_default=True,
    help="The fraction of the integration time flagged.",
)
@click.option("--plot_profile", is_flag=True, help="Plot the pulse profile.")
@click.option("--plot_trec", is_flag=True, help="Plot the receiver temperature.")
@click.option("--plot_pb", is_flag=True, help="Plot the primary beam.")
@click.option("--plot_images", is_flag=True, help="Plot visualisations of the integral quantities.")
@click.option("--plot_3d", is_flag=True, help="Plot the results in 3D (time,freq,data).")
def main(
    log_level: str,
    metafits: str,
    target: str,
    archive: str,
    noise_archive: str,
    bscrunch: int,
    windowsize: int,
    fine_res: float,
    coarse_res: float,
    min_pbp: float,
    nfreq: int,
    ntime: int,
    max_pix_per_job: int,
    bw_flagged: float,
    time_flagged: float,
    plot_profile: bool,
    plot_trec: bool,
    plot_pb: bool,
    plot_images: bool,
    plot_3d: bool,
) -> None:
    log_level_dict = mwa_vcs_fluxcal.get_log_levels()
    logger = mwa_vcs_fluxcal.get_logger(log_level=log_level_dict[log_level])

    # If the archive is not provided, then the SEFD can be still be calculated but
    # the mean flux density cannot
    if archive is not None:
        # Load, dedisperse, and baseline-subtract the detection archive
        archive = mwa_vcs_fluxcal.read_archive(
            archive, bscrunch=bscrunch, subtract_baseline=True, dedisperse=True, logger=logger
        )

        if noise_archive is not None:
            # Load, dedisperse, and baseline-subtract the noise archive
            noise_archive = mwa_vcs_fluxcal.read_archive(
                noise_archive,
                bscrunch=bscrunch,
                subtract_baseline=True,
                dedisperse=False,
                logger=logger,
            )

        snr_profile = mwa_vcs_fluxcal.get_snr_profile(
            archive,
            noise_archive=noise_archive,
            windowsize=windowsize,
            plot_profile=plot_profile,
            logger=logger,
        )
    else:
        if target is None:
            logger.info("No target coordinates provided.")
            exit(0)

    # Cannot go any further without metadata
    if metafits is None:
        logger.info("No metafits file provided. Exiting.")
        exit(0)

    # Prepare metadata
    logger.info(f"Loading metafits: {metafits}")
    context = mwalib.MetafitsContext(metafits)
    T_amb = mwa_vcs_fluxcal.getAmbientTemp(metafits)

    # Get frequency and time metadata
    chan_freqs_hz = context.metafits_fine_chan_freqs_hz
    fctr = (np.min(chan_freqs_hz) + np.max(chan_freqs_hz)) / 2 / 1e6 * u.MHz
    bw = context.obs_bandwidth_hz / 1e6 * u.MHz
    t0 = Time(context.sched_start_mjd, format="mjd")
    t1 = Time(context.sched_end_mjd, format="mjd")

    if archive is not None:
        fctr_ar = archive.get_centre_frequency() * u.MHz
        bw_ar = archive.get_bandwidth() * u.MHz
        t0_ar = Time(archive.get_first_Integration().get_start_time().in_days(), format="mjd")
        t1_ar = Time(archive.get_last_Integration().get_end_time().in_days(), format="mjd")

        if not np.isclose(fctr.value, fctr_ar.value, atol=0.01, rtol=0.0):
            logger.warning(
                f"Centre frequency in the archive ({fctr_ar.to_string()}) "
                + f"is different to the metafits ({fctr.to_string()}). "
                + "Using the archive value."
            )
            fctr = fctr_ar

        if not np.isclose(bw.value, bw_ar.value, atol=0.01, rtol=0.0):
            logger.warning(
                f"Bandwidth in the archive ({bw_ar.to_string()}) "
                + f"is different to the metafits ({bw.to_string()}). "
                + "Using the archive value."
            )
            bw = bw_ar

        if not np.isclose(t0.mjd, t0_ar.mjd, atol=1e-5, rtol=0.0):
            logger.warning(
                f"Start MJD in the archive ({t0_ar.mjd}) "
                + f"differs by {(t0_ar - t0).to(u.s).to_string(precision=3)} "
                + f"to the metafits ({t0.mjd}). Using the archive value."
            )
            t0 = t0_ar

        if not np.isclose(t1.mjd, t1_ar.mjd, atol=1e-5, rtol=0.0):
            logger.warning(
                f"End MJD in the archive ({t1_ar.mjd}) "
                + f"differs by {(t1_ar - t1).to(u.s).to_string(precision=3)} "
                + f"to the metafits ({t1.mjd}). Using the archive value."
            )
            t1 = t1_ar

    f0 = fctr - bw / 2
    f1 = fctr + bw / 2
    dt = t1 - t0

    logger.info(f"Centre frequency = {fctr.to_string()}")
    logger.info(f"Bandwidth = {bw.to_string()}")
    logger.info(f"Integration time = {dt.to(u.s).to_string()}")
    logger.info(f"Start MJD = {t0.to_string()}")
    logger.info(f"Bandwidth flagged = {bw_flagged * 100:.2f}%")
    logger.info(f"Integration time flagged = {time_flagged * 100:.2f}%")

    # Calculate which freqs/times to evalue the integral at
    if nfreq == 1:
        eval_freqs = np.array([fctr.to(u.MHz).value]) * u.MHz
    else:
        eval_freqs = np.linspace(f0.to(u.MHz).value, f1.to(u.MHz).value, nfreq) * u.MHz
    if ntime == 1:
        eval_offsets = np.array([dt.to(u.s).value / 2]) * u.s
    else:
        eval_offsets = np.linspace(0, dt.to(u.s).value, ntime) * u.s
    logger.info(f"Evaluating at {nfreq} frequencies: {eval_freqs}")
    logger.info(f"Evaluating at {ntime} offsets: {eval_offsets}")

    # Define the grid resolutions
    fine_grid_res = Angle(fine_res, u.arcmin)
    coarse_grid_res = Angle(coarse_res, u.arcmin)
    logger.info(f"Fine grid resolution = {fine_grid_res.to_string()}")
    logger.info(f"Coarse grid resolution = {coarse_grid_res.to_string()}")
    if not np.isclose((coarse_grid_res.arcmin / fine_grid_res.arcmin) % 1, 0.0, rtol=1e-5):
        logger.critical("Coarse grid resolution not divisible by fine grid resolution.")
        exit(1)

    # Create a SkyCoord object defining the RA/Dec of the pulsar
    if archive is not None:
        ra_hms, dec_dms = archive.get_coordinates().getHMSDMS().split(" ")
        pulsar_coords = SkyCoord(ra_hms, dec_dms, frame="icrs", unit=("hourangle", "deg"))
    else:
        pulsar_coords = SkyCoord(target, frame="icrs", unit=("hourangle", "deg"))

    logger.info(f"Target RA/Dec = {pulsar_coords.to_string(style='hmsdms')}")

    if plot_trec:
        # Plot the receiver temperature vs frequency
        T_rec_spline = mwa_vcs_fluxcal.splineRecieverTemp()
        mwa_vcs_fluxcal.plot_trcvr_vc_freq(
            T_rec_spline, fctr.to(u.MHz).value, bw.to(u.MHz).value, logger=logger
        )

    # Compute the sky integrals required to get T_sys and gain
    inputs, results = mwa_vcs_fluxcal.compute_sky_integrals(
        context,
        t0,
        eval_offsets,
        eval_freqs,
        pulsar_coords,
        fine_grid_res,
        coarse_grid_res,
        min_pbp,
        max_pix_per_job=max_pix_per_job,
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

    if archive is not None:
        # Radiometer equation (Eq 3 of M+17)
        dt_bin = dt * (1 - time_flagged) / archive.get_nbin()
        bw_valid = bw * (1 - bw_flagged)
        radiometer_noise = results["SEFD_mean"] / np.sqrt(npol * bw_valid.to(1 / u.s) * dt_bin)
        flux_density_profile = snr_profile * radiometer_noise
        S_peak = np.max(flux_density_profile)
        S_mean = integrate.trapezoid(flux_density_profile) / archive.get_nbin()
        logger.info(f"SEFD = {results['SEFD_mean'].to(u.Jy).to_string()}")
        logger.info(f"Peak flux density = {S_peak.to(u.mJy).to_string()}")
        logger.info(f"Mean flux density = {S_mean.to(u.mJy).to_string()}")

        # Add to the results dictionary
        results["noise_rms"] = radiometer_noise
        results["S_peak"] = S_peak
        results["S_mean"] = S_mean

    # Dump the dictionaries to toml files
    mwa_vcs_fluxcal.qty_dict_to_toml(inputs, "inputs.toml")
    mwa_vcs_fluxcal.qty_dict_to_toml(results, "results.toml")


if __name__ == "__main__":
    main()
