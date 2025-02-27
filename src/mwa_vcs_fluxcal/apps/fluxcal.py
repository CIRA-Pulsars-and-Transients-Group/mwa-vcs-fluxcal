########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

# This script is an implementation of the flux density calibration
# method described in Meyers et al. (2017), hereafter abbreviated M+17.
# https://ui.adsabs.harvard.edu/abs/2017ApJ...851...20M/abstract

import logging
from time import perf_counter as pc

import astropy.units as u
import click
import mwalib
import numpy as np
import toml
from astropy.constants import c, k_B
from astropy.coordinates import AltAz, Angle, SkyCoord
from astropy.time import Time
from scipy import integrate
from tqdm import tqdm

import mwa_vcs_fluxcal
from mwa_vcs_fluxcal import MWA_LOCATION, SI_TO_JY


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
@click.option("--plot_sefd", is_flag=True, help="Plot the SEFD results in 3D (time,freq,SEFD).")
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
    plot_sefd: bool,
) -> None:
    log_level_dict = mwa_vcs_fluxcal.get_log_levels()
    logger = mwa_vcs_fluxcal.get_logger(log_level=log_level_dict[log_level])

    # If level is below INFO, disable progress bar as it will be broken up by
    # verbose log statements. If it is above INFO, also disable it.
    disable_tqdm = True
    if logger.level is logging.INFO:
        disable_tqdm = False

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
    tile_positions = mwa_vcs_fluxcal.extractWorkingTilePositions(context)

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

    # Get the sky coordinates of the pulsar
    ra_hms, dec_dms = archive.get_coordinates().getHMSDMS().split(" ")
    pulsar_position = SkyCoord(ra_hms, dec_dms, frame="icrs", unit=("hourangle", "deg"))
    altaz_frame = AltAz(location=MWA_LOCATION, obstime=eval_times)
    pulsar_position_altaz = pulsar_position.transform_to(altaz_frame)

    # This dictionary will store the results to be written out
    results = dict(
        integral_resolution=fine_grid_res.to(u.arcmin),
        tab_width=width.to(u.arcmin),
        t=eval_offsets.to(u.s),
        f=eval_freqs.to(u.MHz),
        pulsar_az=pulsar_position_altaz.az.to(u.deg),
        pulsar_za=pulsar_position_altaz.alt.to(u.deg),
        T_rec=u.Quantity(np.empty((nfreq), dtype=np.float64), u.K),
        T_ant=u.Quantity(np.empty((ntime, nfreq), dtype=np.float64), u.K),
        T_sys=u.Quantity(np.empty((ntime, nfreq), dtype=np.float64), u.K),
        Omega_A=u.Quantity(np.empty((ntime, nfreq), dtype=np.float64), u.sr),
        A_eff=u.Quantity(np.empty((ntime, nfreq), dtype=np.float64), u.m**2),
        G=u.Quantity(np.empty((ntime, nfreq), dtype=np.float64), u.K * u.Jy**-1),
        SEFD=u.Quantity(np.empty((ntime, nfreq), dtype=np.float64), u.Jy),
        SEFD_mean=u.Quantity(np.float64(0.0), u.Jy),
        SNR_peak=u.Quantity(np.max(snr_profile), u.dimensionless_unscaled),
        noise_rms=u.Quantity(np.float64(0.0), u.Jy),
        S_peak=u.Quantity(np.float64(0.0), u.Jy),
        S_mean=u.Quantity(np.float64(0.0), u.Jy),
    )

    # Assumptions
    fc = 0.7
    eta = 0.98
    npol = 2
    t0 = 290 * u.K

    # Create a coarse meshgrid so that we can estimate the sky area with
    # significant power in the primary beam
    az_box_coarse = np.arange(0, 2 * np.pi, coarse_grid_res.radian)
    za_box_coarse = np.arange(0, np.pi / 2, coarse_grid_res.radian)
    az_grid_coarse, za_grid_coarse = np.meshgrid(az_box_coarse, za_box_coarse)
    alt_grid_coarse = np.pi / 2 - za_grid_coarse

    # We will need to keep track of which pixels correspond to which array
    # indices in the original array
    az_grid_idx_coarse, za_grid_idx_coarse = np.meshgrid(
        np.arange(az_grid_coarse.shape[0]), np.arange(az_grid_coarse.shape[1]), indexing="ij"
    )

    # Flatten the masked coarse meshgrid. Since the coarse pixels will be
    # up-sampled later, we'll use the shorthand 'blocks' to mean the coarse
    # pixels from the primary beam map.
    az_blocks = az_grid_coarse.flatten()
    za_blocks = za_grid_coarse.flatten()
    az_idx_blocks = az_grid_idx_coarse.flatten()
    za_idx_blocks = za_grid_idx_coarse.flatten()

    # How many fine pixels will we compute per job?
    max_pixels_per_job = 10**5

    # How many fine pixels per coarse pixel?
    upscale_ratio = (coarse_grid_res.arcmin / fine_grid_res.arcmin) ** 2

    # How many coarse pixels per job?
    max_blocks_per_job = max_pixels_per_job // upscale_ratio

    # Receiver temperature
    T_rec_spline = mwa_vcs_fluxcal.splineRecieverTemp()
    if plot_trec:
        mwa_vcs_fluxcal.plot_trcvr_vc_freq(
            T_rec_spline, fctr.to(u.MHz).value, bw.to(u.MHz).value, logger=logger
        )
    T_rec = T_rec_spline(eval_freqs.to(u.MHz).value) * u.K
    results["T_rec"] = T_rec

    # For each evaluation frequency we will calculate which parts of the sky are
    # within the primary beam and only integrate the pixels in those regions
    for ii in range(nfreq):
        logger.info(f"Computing frequency {ii}: {eval_freqs[ii].to_string(precision=2)}")

        # Define a "look" vector pointing towards the pulsar
        look_psi = mwa_vcs_fluxcal.calcGeometricDelays(
            tile_positions,
            eval_freqs[ii].to(u.Hz).value,
            pulsar_position_altaz.alt.rad,
            pulsar_position_altaz.az.rad,
        ).T

        # Compute the primary beam power
        grid_pbp = mwa_vcs_fluxcal.getPrimaryBeamPower(
            context, eval_freqs[ii].to(u.Hz).value, alt_grid_coarse, az_grid_coarse, logger=logger
        )["I"].reshape(az_grid_coarse.shape)

        # Create a mask selecting the coarse pixels covering the primary beam
        if min_pbp > 0.0:
            pb_mask = mwa_vcs_fluxcal.tesellate_primary_beam(
                az_grid_coarse,
                za_grid_coarse,
                grid_pbp,
                coarse_grid_res.radian,
                plevel=min_pbp,
                plot=plot_pb,
                pulsar_coords=pulsar_position_altaz,
                savename=f"primary_beam_masked_{eval_freqs[ii].to(u.MHz).value:.0f}MHz.png",
                logger=logger,
            )
        else:
            pb_mask = np.full(shape=grid_pbp.shape, fill_value=True, dtype=bool)

        pb_mask = ~pb_mask.flatten()

        # Calculate how many blocks there are before/after masking and how many
        # jobs we will require
        num_blocks = az_blocks.size
        num_blocks_cutout = az_blocks[~pb_mask].size
        cutout_frac = num_blocks_cutout / num_blocks
        num_jobs = np.ceil(num_blocks / max_blocks_per_job).astype(int)
        logger.info(f"Looping over {num_blocks * upscale_ratio:.0f} pixels in {num_jobs} jobs")
        logger.info(f"T_ant will be computed for {cutout_frac * 100:.2f}% of pixels")

        # Split up the blocks array into groups of one or more blocks (i.e. jobs)
        az_jobs = np.array_split(az_blocks, num_jobs)
        za_jobs = np.array_split(za_blocks, num_jobs)
        pb_mask_jobs = np.array_split(pb_mask, num_jobs)
        az_idx_jobs = np.array_split(az_idx_blocks, num_jobs)
        za_idx_jobs = np.array_split(za_idx_blocks, num_jobs)

        # Arrays to store integrals for each time step
        integral_B_T = np.zeros(ntime, dtype=np.float64)
        integral_B = np.zeros(ntime, dtype=np.float64)
        integral_afp = np.zeros(ntime, dtype=np.float64)

        if plot_images:
            # Arrays to store the coarse images for plotting
            pbp_coarse = np.zeros(shape=grid_pbp.shape, dtype=np.float64)
            afp_coarse = np.zeros(shape=(ntime, *grid_pbp.shape), dtype=np.float64)
            tabp_coarse = np.zeros(shape=(ntime, *grid_pbp.shape), dtype=np.float64)
            tsky_coarse = np.zeros(shape=(ntime, *grid_pbp.shape), dtype=np.float64)
            int_B_T_coarse = np.zeros(shape=(ntime, *grid_pbp.shape), dtype=np.float64)
            int_B_coarse = np.zeros(shape=(ntime, *grid_pbp.shape), dtype=np.float64)
            int_afp_coarse = np.zeros(shape=(ntime, *grid_pbp.shape), dtype=np.float64)

        # Make sure there are no INFO-level logs in this loop
        for jj in tqdm(range(num_jobs), unit="job", disable=disable_tqdm):
            logger.debug(f"Computing job {jj}")

            # Alt/Az coordinates of coarse pixels (blocks) in this job
            az_job = az_jobs[jj]
            za_job = za_jobs[jj]

            # Grid indices of the coarse pixels (blocks) in this job
            az_idx = az_idx_jobs[jj]
            za_idx = za_idx_jobs[jj]

            # Mask indicating which coarse pixels are in the beam
            pb_mask_job = pb_mask_jobs[jj]

            # We will need these for reshaping later
            num_blocks_job = az_job.size
            num_blocks_inbeam_job = az_job[~pb_mask_job].size

            # Compute the fine pixel coordinates
            az_fine, za_fine = mwa_vcs_fluxcal.upsample_blocks(
                az_job, za_job, coarse_grid_res.radian, fine_grid_res.radian
            )
            alt_fine = np.pi / 2 - za_fine

            # Define a grid of "target" vectors pointing towards each pixel
            bench_t0 = pc()
            target_psi = mwa_vcs_fluxcal.calcGeometricDelays(
                tile_positions,
                eval_freqs[ii].to(u.Hz).value,
                alt_fine,
                az_fine,
            )
            logger.debug(f"Computing target_psi took {pc() - bench_t0} s")

            # Calculate the array factor power (Eq 11 of M+17)
            # afp will have shape (ntime,npixels)
            bench_t0 = pc()
            afp = mwa_vcs_fluxcal.calcArrayFactorPower(look_psi, target_psi, logger=logger)
            logger.debug(f"Computing afp took {pc() - bench_t0} s")

            # Reshape the flattened arrays into (nblocks,pixels_per_block) so
            # that we can apply the coarse pixel mask
            az_fine_blocks = az_fine.reshape(num_blocks_job, -1)
            za_fine_blocks = za_fine.reshape(num_blocks_job, -1)
            alt_fine_blocks = alt_fine.reshape(num_blocks_job, -1)
            afp_blocks = afp.reshape(ntime, num_blocks_job, -1)

            # Apply the mask (to get the in-beam pixels) and flatten again
            az_fine_inbeam = az_fine_blocks[~pb_mask_job].flatten()
            za_fine_inbeam = za_fine_blocks[~pb_mask_job].flatten()
            alt_fine_inbeam = alt_fine_blocks[~pb_mask_job].flatten()
            afp_inbeam = afp_blocks[:, ~pb_mask_job, :].reshape(ntime, -1)

            # The differential has shape (npixels,)
            pixel_area_inbeam = fine_grid_res.radian**2 * np.sin(za_fine_inbeam)
            pixel_area = fine_grid_res.radian**2 * np.sin(za_fine)

            # Compute the partial array factor power integral (Eq 14 of M+17)
            # Note that afp has shape (ntime,npixels) and broadcasting will be
            # done along the pixel axis assuming that npixels > ntime
            integrand_afp = afp * pixel_area
            integral_afp += np.sum(integrand_afp, axis=1)

            if plot_images:
                # Store the afp image for later
                for ll, (nn, mm) in enumerate(zip(az_idx, za_idx, strict=True)):
                    for kk in range(ntime):
                        afp_coarse[kk, nn, mm] = np.mean(afp_blocks[kk, ll])

                # Map the flattened integrand array back to the coarse grid
                int_afp_blocks = integrand_afp.reshape(ntime, num_blocks_job, -1)
                for ll, (nn, mm) in enumerate(zip(az_idx, za_idx, strict=True)):
                    for kk in range(ntime):
                        int_afp_coarse[kk, nn, mm] = np.sum(int_afp_blocks[kk, ll])

            ##################################################################
            # Skip the rest of the loop cycle if there are no in-beam pixels #
            ##################################################################
            if az_fine_inbeam.size == 0:
                logger.debug("Skipping T_ant integral")
                continue

            # Calculate the primary beam power for each pixel in the job
            bench_t0 = pc()
            pbp = mwa_vcs_fluxcal.getPrimaryBeamPower(
                context,
                eval_freqs[ii].to(u.Hz).value,
                alt_fine_inbeam,
                az_fine_inbeam,
                logger=logger,
            )["I"]
            logger.debug(f"Computing PB took {pc() - bench_t0} s")

            bench_t0 = pc()
            tsky = np.empty_like(afp_inbeam)
            tabp = np.empty_like(afp_inbeam)
            for kk in range(ntime):
                # Get the sky temperature for each pixel in the grid
                pix_coords = SkyCoord(
                    az=Angle(az_fine_inbeam, u.rad),
                    alt=Angle(alt_fine_inbeam, u.rad),
                    frame="altaz",
                    location=MWA_LOCATION,
                    obstime=eval_times[kk],
                )
                tsky[kk, :] = mwa_vcs_fluxcal.getSkyTempAtCoords(
                    pix_coords, eval_freqs[ii].to(u.MHz).value, logger=logger
                )

                # Calculate the tied-array beam power (Eq 12 of M+17)
                # afp has shape (ntime,npixels) and pbp has shape (npixels,)
                tabp[kk, :] = afp_inbeam[kk, :] * pbp
            logger.debug(f"Computing T_sky and TAB took {pc() - bench_t0} s")

            # Compute the beam integrals (Eq 13 of M+17)
            # Note that tabp/tsky have shape (ntime,npixels) and broadcasting
            # will be done along the pixel axis assuming that npixels > ntime
            integrand_B_T = tabp * tsky * pixel_area_inbeam
            integrand_B = tabp * pixel_area_inbeam
            # The integrals have shape (ntime,)
            integral_B_T += np.sum(integrand_B_T, axis=1)
            integral_B += np.sum(integrand_B, axis=1)

            if plot_images:
                # Now we want to map the results on the fine grid back onto the
                # coarse grid so that we can plot it. To do this we will take the
                # mean value of the fine pixels in each coarse pixel (block).
                pbp_blocks = pbp.reshape(num_blocks_inbeam_job, -1)
                tabp_blocks = tabp.reshape(ntime, num_blocks_inbeam_job, -1)
                tsky_blocks = tsky.reshape(ntime, num_blocks_inbeam_job, -1)
                int_B_T_blocks = integrand_B_T.reshape(ntime, num_blocks_inbeam_job, -1)
                int_B_blocks = integrand_B.reshape(ntime, num_blocks_inbeam_job, -1)
                for ll, (nn, mm) in enumerate(
                    zip(az_idx[~pb_mask_job], za_idx[~pb_mask_job], strict=True)
                ):
                    pbp_coarse[nn, mm] = np.mean(pbp_blocks[ll])
                    for kk in range(ntime):
                        tabp_coarse[kk, nn, mm] = np.mean(tabp_blocks[kk, ll])
                        tsky_coarse[kk, nn, mm] = np.mean(tsky_blocks[kk, ll])
                        int_B_T_coarse[kk, nn, mm] = np.sum(int_B_T_blocks[kk, ll])
                        int_B_coarse[kk, nn, mm] = np.sum(int_B_blocks[kk, ll])

        if plot_images:
            for tt in range(ntime):
                mwa_vcs_fluxcal.plot_sky_images(
                    az_grid_coarse,
                    za_grid_coarse,
                    [afp_coarse[tt], pbp_coarse, tabp_coarse[tt]],
                    [
                        "$|f(\\theta,\phi)|^2$",
                        "$|D(\\theta,\phi)|^2$",
                        "$B_\mathrm{array}(\\theta,\phi)$",
                    ],
                    pulsar_position_altaz[tt],
                    savename=f"input_images_{eval_freqs[ii].to(u.MHz).value:.0f}MHz_t{tt}.png",
                    logger=logger,
                )
                mwa_vcs_fluxcal.plot_sky_images(
                    az_grid_coarse,
                    za_grid_coarse,
                    [
                        mwa_vcs_fluxcal.log_nan_zeros(afp_coarse[tt]),
                        mwa_vcs_fluxcal.log_nan_zeros(pbp_coarse),
                        mwa_vcs_fluxcal.log_nan_zeros(tabp_coarse[tt]),
                    ],
                    [
                        "$\log_{10}[|f(\\theta,\phi)|^2]$",
                        "$\log_{10}[|D(\\theta,\phi)|^2]$",
                        "$\log_{10}[B_\mathrm{array}(\\theta,\phi)]$",
                    ],
                    pulsar_position_altaz[tt],
                    savename=f"log_input_images_{eval_freqs[ii].to(u.MHz).value:.0f}MHz_t{tt}.png",
                    logger=logger,
                )
                mwa_vcs_fluxcal.plot_sky_images(
                    az_grid_coarse,
                    za_grid_coarse,
                    [int_afp_coarse[tt], int_B_T_coarse[tt], int_B_coarse[tt]],
                    [
                        "$|f(\\theta,\phi)|^2\,\mathrm{d}\Omega$",
                        "$B_\mathrm{array}(\\theta,\phi) "
                        + "T_\mathrm{sky}(\\theta,\phi)\,\mathrm{d}\Omega$",
                        "$B_\mathrm{array}(\\theta,\phi)\,\mathrm{d}\Omega$",
                    ],
                    pulsar_position_altaz[tt],
                    savename=f"integral_images_{eval_freqs[ii].to(u.MHz).value:.0f}MHz_t{tt}.png",
                    logger=logger,
                )
                mwa_vcs_fluxcal.plot_sky_images(
                    az_grid_coarse,
                    za_grid_coarse,
                    [
                        mwa_vcs_fluxcal.log_nan_zeros(int_afp_coarse[tt]),
                        mwa_vcs_fluxcal.log_nan_zeros(int_B_T_coarse[tt]),
                        mwa_vcs_fluxcal.log_nan_zeros(int_B_coarse[tt]),
                    ],
                    [
                        "$\log_{10}[|f(\\theta,\phi)|^2\,\mathrm{d}\Omega]$",
                        "$\log_{10}[B_\mathrm{array}(\\theta,\phi) "
                        + "T_\mathrm{sky}(\\theta,\phi)\,\mathrm{d}\Omega]$",
                        "$\log_{10}[B_\mathrm{array}(\\theta,\phi)\,\mathrm{d}\Omega]$",
                    ],
                    pulsar_position_altaz[tt],
                    savename=f"log_integral_images_{eval_freqs[ii].to(u.MHz).value:.0f}MHz_t{tt}.png",
                    logger=logger,
                )
                mwa_vcs_fluxcal.plot_sky_images(
                    az_grid_coarse,
                    za_grid_coarse,
                    [mwa_vcs_fluxcal.log_nan_zeros(tsky_coarse[tt])],
                    ["$\mathrm{log}_{10}\,T_\mathrm{sky}$ [K]"],
                    pulsar_position_altaz[tt],
                    savename=f"tsky_image_{eval_freqs[ii].to(u.MHz).value:.0f}MHz_t{tt}.png",
                    logger=logger,
                )

        # Antenna temperature (Eq 13 of M+17)
        # The integrals have units of sr that cancel out
        T_ant = integral_B_T / integral_B * u.K

        # System temperature (Eq 1 of M+17)
        T_sys = eta * T_ant + (1 - eta) * t0 + T_rec[ii]

        # Beam solid angle (Eq 14 of M+17)
        Omega_A = integral_afp * u.sr

        # Effective area (Eq 15 of M+17)
        A_eff = eta * 4 * np.pi * u.sr * (c / eval_freqs[ii].to(u.s**-1)) ** 2 / Omega_A

        # Gain (Eq 16 of M+17)
        G = A_eff / (2 * k_B) * SI_TO_JY

        # SEFD
        sefd = fc * T_sys / G

        # Save the results
        results["T_ant"][:, ii] = T_ant
        results["T_sys"][:, ii] = T_sys
        results["Omega_A"][:, ii] = Omega_A
        results["A_eff"][:, ii] = A_eff
        results["G"][:, ii] = G
        results["SEFD"][:, ii] = sefd

    if plot_sefd and nfreq >= 4 and ntime >= 4:
        mwa_vcs_fluxcal.plot_3d_result(
            eval_offsets.to(u.s).value,
            eval_freqs.to(u.MHz).value,
            results["SEFD"].value,
            zlabel="SEFD [Jy]",
            logger=logger,
        )

    # Radiometer equation (Eq 3 of M+17)
    sefd_mean = np.mean(results["SEFD"])
    dt_bin = dt / archive.get_nbin()
    radiometer_noise = sefd_mean / np.sqrt(npol * bw.to(1 / u.s) * dt_bin)
    flux_density_profile = snr_profile * radiometer_noise
    S_peak = np.max(flux_density_profile)
    S_mean = integrate.trapezoid(flux_density_profile) / archive.get_nbin()
    logger.info(f"SEFD = {sefd_mean.to(u.Jy).to_string()}")
    logger.info(f"Peak flux density = {S_peak.to(u.mJy).to_string()}")
    logger.info(f"Mean flux density = {S_mean.to(u.mJy).to_string()}")
    results["SEFD_mean"] = sefd_mean
    results["noise_rms"] = radiometer_noise
    results["S_peak"] = S_peak
    results["S_mean"] = S_mean

    # Write results
    results_vals = dict()
    for key in results:
        results_vals[key] = [results[key].value, results[key].unit.to_string()]
    with open("results.toml", "w") as f:
        toml.dump(results_vals, f, encoder=toml.TomlNumpyEncoder())


if __name__ == "__main__":
    main()
