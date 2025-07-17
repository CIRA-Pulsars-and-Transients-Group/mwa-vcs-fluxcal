########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging
from time import perf_counter as pc

import astropy.units as u
import enlighten
import numpy as np
from astropy.constants import c, k_B
from astropy.coordinates import AltAz, Angle, SkyCoord
from astropy.time import Time
from mwalib import MetafitsContext

import mwa_vcs_fluxcal
from mwa_vcs_fluxcal import MWA_LOCATION, SI_TO_JY, eta, fc

__all__ = ["compute_sky_integrals"]

logger = logging.getLogger(__name__)


# TODO: Make docstring
def compute_sky_integrals(
    context: MetafitsContext,
    start_time: Time,
    eval_offsets: u.Quantity,
    eval_freqs: u.Quantity,
    pulsar_coords: SkyCoord,
    fine_grid_res: Angle,
    coarse_grid_res: Angle,
    min_pbp: float,
    max_pix_per_job: int = 10**5,
    plot_pb: bool = False,
    plot_tab: bool = False,
    plot_tsky: bool = False,
    plot_integrals: bool = False,
    T_amb: u.Quantity = 295.55 * u.K,
    file_prefix: str = "fluxcal",
    pbar_manager: enlighten.Manager | None = None,
) -> dict[str, u.Quantity]:
    # Making these plots uses some extra memory
    if plot_tab or plot_tsky or plot_integrals:
        plot_images = True
    else:
        plot_images = False

    ntime = len(eval_offsets)
    nfreq = len(eval_freqs)

    # Transform the pulsar coordinates to Alt/Az at each epoch
    eval_times = start_time + eval_offsets
    altaz_frame = AltAz(location=MWA_LOCATION, obstime=eval_times)
    pulsar_coords_altaz = pulsar_coords.transform_to(altaz_frame)

    # Compute the tile positions from the metadata
    tile_positions = mwa_vcs_fluxcal.extractWorkingTilePositions(context)

    # Dictionary to store the inputs and outputs of the integral calculation
    results = dict(
        Times=eval_offsets.to(u.s),
        Freqs=eval_freqs.to(u.MHz),
        Pulsar_Az=pulsar_coords_altaz.az.to(u.deg),
        Pulsar_Alt=pulsar_coords_altaz.alt.to(u.deg),
        Angular_resolution=fine_grid_res.to(u.arcmin),
        T_amb=T_amb.to(u.K),
        T_rec=u.Quantity(np.empty((nfreq), dtype=np.float64), u.K),
        T_ant=u.Quantity(np.empty((ntime, nfreq), dtype=np.float64), u.K),
        T_sys=u.Quantity(np.empty((ntime, nfreq), dtype=np.float64), u.K),
        Omega_A=u.Quantity(np.empty((ntime, nfreq), dtype=np.float64), u.sr),
        A_eff=u.Quantity(np.empty((ntime, nfreq), dtype=np.float64), u.m**2),
        G=u.Quantity(np.empty((ntime, nfreq), dtype=np.float64), u.K * u.Jy**-1),
        SEFD=u.Quantity(np.empty((ntime, nfreq), dtype=np.float64), u.Jy),
        SEFD_mean=u.Quantity(np.float64(0.0), u.Jy),
    )

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

    # How many fine pixels per coarse pixel?
    upscale_ratio = (coarse_grid_res.arcmin / fine_grid_res.arcmin) ** 2

    # How many coarse pixels per job?
    max_blocks_per_job = max_pix_per_job // upscale_ratio

    # Receiver temperature
    T_rec_spline = mwa_vcs_fluxcal.splineRecieverTemp()
    T_rec = T_rec_spline(eval_freqs.to(u.MHz).value) * u.K
    results["T_rec"] = T_rec

    # For each evaluation frequency we will calculate which parts of the sky are
    # within the primary beam and only integrate the pixels in those regions
    if pbar_manager:
        fpbar = pbar_manager.counter(total=nfreq, desc="  Computing:", unit="freqs")
    for ii in range(nfreq):
        freq_val = eval_freqs[ii].to(u.MHz).value
        logger.info(f"Frequency {ii}: {freq_val:.2f} MHz")

        # Define a "look" vector pointing towards the pulsar
        look_psi = mwa_vcs_fluxcal.calcGeometricDelays(
            tile_positions,
            eval_freqs[ii].to(u.Hz).value,
            pulsar_coords_altaz.alt.rad,
            pulsar_coords_altaz.az.rad,
        ).T

        # Compute the primary beam power
        grid_pbp = mwa_vcs_fluxcal.getPrimaryBeamPower(
            context, eval_freqs[ii].to(u.Hz).value, alt_grid_coarse, az_grid_coarse
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
                pulsar_coords=pulsar_coords_altaz,
                savename=f"{file_prefix}_primary_beam_masked_{eval_freqs[ii].to(u.MHz).value:.0f}MHz.png",
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

        if pbar_manager:
            jpbar = pbar_manager.counter(
                total=num_jobs, desc=f"{freq_val:-7.2f} MHz:", unit="jobs", leave=False
            )
        for jj in range(num_jobs):
            logger.debug(f"Job {jj}")

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
            afp = mwa_vcs_fluxcal.calcArrayFactorPower(look_psi, target_psi)
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
                context, eval_freqs[ii].to(u.Hz).value, alt_fine_inbeam, az_fine_inbeam
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
                    pix_coords, eval_freqs[ii].to(u.MHz).value
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

            if pbar_manager:
                jpbar.update()

        jpbar.close()

        if plot_images:
            for tt in range(ntime):
                if plot_tab:
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
                        pulsar_coords_altaz[tt],
                        savename=f"{file_prefix}_log_beam_images_{eval_freqs[ii].to(u.MHz).value:.0f}MHz_t{tt}.png",
                    )
                if plot_integrals:
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
                        pulsar_coords_altaz[tt],
                        savename=f"{file_prefix}_log_integral_images_{eval_freqs[ii].to(u.MHz).value:.0f}MHz_t{tt}.png",
                    )
                if plot_tsky:
                    mwa_vcs_fluxcal.plot_sky_images(
                        az_grid_coarse,
                        za_grid_coarse,
                        [mwa_vcs_fluxcal.log_nan_zeros(tsky_coarse[tt])],
                        ["$\mathrm{log}_{10}\,T_\mathrm{sky}$ [K]"],
                        pulsar_coords_altaz[tt],
                        savename=f"{file_prefix}_log_tsky_image_{eval_freqs[ii].to(u.MHz).value:.0f}MHz_t{tt}.png",
                    )

        # Antenna temperature (Eq 13 of M+17)
        # The integrals have units of sr that cancel out
        T_ant = integral_B_T / integral_B * u.K

        # System temperature (Eq 1 of M+17)
        T_sys = eta * T_ant + (1 - eta) * T_amb + T_rec[ii]

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

        if pbar_manager:
            fpbar.update()

    fpbar.close()

    results["SEFD_mean"] = np.mean(results["SEFD"])

    return results
