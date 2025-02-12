########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import astropy.units as u
import click
import mwalib
import numpy as np
import psrchive
from astropy.constants import c, k_B
from astropy.coordinates import AltAz, Angle, SkyCoord
from astropy.time import Time

import mwa_vcs_fluxcal
from mwa_vcs_fluxcal import MWA_LOCATION, SI_TO_JY

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
@click.option("--nfreq", type=int, default=1, help="The number of frequency steps to evaluate.")
@click.option("--ntime", type=int, default=1, help="The number of time steps to evaluate.")
@click.option("--plot_profile", is_flag=True, help="Plot the pulse profile.")
@click.option("--plot_trec", is_flag=True, help="Plot the receiver temperature.")
@click.option("--plot_pb", is_flag=True, help="Plot the primary beam.")
def main(
    archive: str,
    log_level: str,
    metafits: str,
    windowsize: int,
    fine_res: float,
    coarse_res: float,
    nfreq: int,
    ntime: int,
    plot_profile: bool,
    plot_trec: bool,
    plot_pb: bool,
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
    tile_positions = mwa_vcs_fluxcal.extractWorkingTilePositions(context)

    # Get frequency and time metadata from archive
    fctr = archive.get_centre_frequency() * u.MHz
    df = archive.get_bandwidth() * u.MHz
    f0 = fctr - df / 2
    f1 = fctr + df / 2
    t0 = archive.get_first_Integration().get_start_time()
    t1 = archive.get_last_Integration().get_end_time()
    dt = (t1 - t0).in_seconds() * u.s
    start_time = Time(t0.in_days(), format="mjd")
    logger.info(f"Centre frequency = {fctr.to_string()}")
    logger.info(f"Bandwidth = {df.to_string()}")
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

    # This dictionary will store the results for all time/freq steps
    results = dict(
        T_ant=u.Quantity(np.empty((ntime, nfreq), dtype=np.float64), u.K),
        T_sys=u.Quantity(np.empty((ntime, nfreq), dtype=np.float64), u.K),
        Omega_A=u.Quantity(np.empty((ntime, nfreq), dtype=np.float64), u.radian**2),
        A_eff=u.Quantity(np.empty((ntime, nfreq), dtype=np.float64), u.m**2),
        G=u.Quantity(np.empty((ntime, nfreq), dtype=np.float64), u.K * u.Jy**-1),
        SEFD=u.Quantity(np.empty((ntime, nfreq), dtype=np.float64), u.Jy),
    )

    # Calculate the beam width
    max_baseline, _, _ = mwa_vcs_fluxcal.find_max_baseline(context)
    max_baseline *= u.m
    width = ((c / fctr.to(1 / u.s)) / max_baseline) * u.rad
    logger.info(f"Maximum baseline: {max_baseline.to_string()}")
    logger.info(f"Beam width ~ lambda/D: {width.to(u.arcminute).to_string()}")

    # Define the grid resolutions
    fine_grid_res = Angle(fine_res, u.arcmin)
    coarse_grid_res = Angle(coarse_res, u.arcmin)
    logger.info(f"Fine grid resolution = {fine_grid_res.to_string()}")
    logger.info(f"Coarse grid resolution = {coarse_grid_res.to_string()}")

    # Get the sky coordinates of the pulsar
    ra_hms, dec_dms = archive.get_coordinates().getHMSDMS().split(" ")
    pulsar_position = SkyCoord(ra_hms, dec_dms, frame="icrs", unit=("hourangle", "deg"))
    altaz_frame = AltAz(location=MWA_LOCATION, obstime=eval_times)
    pulsar_position_altaz = pulsar_position.transform_to(altaz_frame)

    # Create a coarse meshgrid so that we can estimate the sky area with
    # significant power in the primary beam
    az_box_coarse = np.arange(0, 2 * np.pi, coarse_grid_res.radian)
    za_box_coarse = np.arange(0, np.pi / 2, coarse_grid_res.radian)
    az_grid_coarse, za_grid_coarse = np.meshgrid(az_box_coarse, za_box_coarse)
    alt_grid_coarse = np.pi / 2 - za_grid_coarse

    # How many fine pixels will we compute per job?
    max_pixels_per_job = 10**5

    # How many fine pixels per coarse pixel?
    upscale_ratio = (coarse_grid_res.arcmin / fine_grid_res.arcmin) ** 2

    # How many coarse pixels per job?
    max_blocks_per_job = max_pixels_per_job // upscale_ratio

    # Receiver temperature
    T_rec_spline = mwa_vcs_fluxcal.splineRecieverTemp()
    if plot_trec:
        mwa_vcs_fluxcal.plot_trcvr_vc_freq(T_rec_spline, fctr, df, logger=logger)
    T_rec = T_rec_spline(eval_freqs.to(u.MHz).value) * u.K

    # Hardcode these for now
    eta = 0.9
    t0 = 290 * u.K

    # For each evaluation frequency we will calculate which parts of the sky are
    # within the primary beam and only integrate the pixels in those regions
    for ii in range(nfreq):
        logger.info(f"Computing frequency {ii}: {eval_freqs[ii].to_string()}")

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

        # Select the coarse pixels covering the primary beam
        az_grid_coarse_inbeam, za_grid_coarse_inbeam, _ = mwa_vcs_fluxcal.tesellate_primary_beam(
            az_grid_coarse,
            za_grid_coarse,
            grid_pbp,
            coarse_grid_res.radian,
            plot=plot_pb,
            pulsar_coords=pulsar_position_altaz,
            savename=f"primary_beam_masked_{eval_freqs[ii].to(u.MHz).value:.0f}MHz.png",
            logger=logger,
        )

        # Flatten the masked coarse meshgrid. Since the coarse pixels will be
        # up-sampled later, we'll use the shorthand 'blocks' to mean the coarse
        # pixels from the primary beam map.
        az_blocks = az_grid_coarse_inbeam.flatten()
        za_blocks = za_grid_coarse_inbeam.flatten()

        # Calculate how many blocks there are before/after masking
        num_blocks_tot = az_blocks.size
        nan_blocks = np.isnan(az_blocks)
        az_blocks = az_blocks[~nan_blocks]
        za_blocks = za_blocks[~nan_blocks]
        num_blocks_cut = az_blocks.size
        logger.info(f"Integrating {num_blocks_cut / num_blocks_tot * 100:.2f}% of the sky")

        # How many jobs will we require?
        num_jobs = np.ceil(num_blocks_cut / max_blocks_per_job).astype(int)
        logger.info(f"Will compute {num_blocks_cut * upscale_ratio:.0f} pixels in {num_jobs} jobs")

        # Split up the blocks array into groups of one or more blocks (i.e. jobs)
        az_jobs = np.array_split(az_blocks, num_jobs)
        za_jobs = np.array_split(za_blocks, num_jobs)

        # Loop through subboxes and integrate
        integral_top = np.zeros(ntime, dtype=np.float64)
        integral_bot = np.zeros(ntime, dtype=np.float64)
        Omega_A = np.zeros(ntime, dtype=np.float64)
        for jj in range(num_jobs):
            logger.info(f"Computing job {jj}")

            az_job = az_jobs[jj]
            za_job = za_jobs[jj]

            az_fine, za_fine = mwa_vcs_fluxcal.upsample_blocks(
                az_job, za_job, coarse_grid_res.radian, fine_grid_res.radian
            )
            alt_fine = np.pi / 2 - za_fine

            # Calculate the primary beam power
            pbp = mwa_vcs_fluxcal.getPrimaryBeamPower(
                context,
                eval_freqs[ii].to(u.Hz).value,
                alt_fine,
                az_fine,
                logger=logger,
            )["I"]

            # Define a grid of "target" vectors pointing towards each pixel
            target_psi = mwa_vcs_fluxcal.calcGeometricDelays(
                tile_positions,
                eval_freqs[ii].to(u.Hz).value,
                alt_fine,
                az_fine,
            )

            # Calculate the array factor power
            afp = mwa_vcs_fluxcal.calcArrayFactorPower(look_psi, target_psi, logger=logger)

            tsky = np.empty_like(afp)
            tabp = np.empty_like(afp)
            for kk in range(ntime):
                # Get the sky temperature for each pixel in the grid
                pix_coords = SkyCoord(
                    az=Angle(az_fine, u.rad),
                    alt=Angle(alt_fine, u.rad),
                    frame="altaz",
                    location=MWA_LOCATION,
                    obstime=eval_times[kk],
                )
                tsky[kk, ...] = mwa_vcs_fluxcal.getSkyTempGrid(
                    pix_coords, eval_freqs[ii].to(u.MHz).value, logger=logger
                )

                # Calculate the tied-array beam power
                # afp has dimensions (ntime,npixels) and pbp has dimensions (npixels,)
                tabp[kk, ...] = afp[kk, ...] * pbp

            # Compute the integral
            # integrands have dimensions (ntime,npixels)
            # integrals have dimensions (ntime,)
            pixel_area = fine_grid_res.radian * fine_grid_res.radian * np.sin(za_fine)
            integrand_top = tabp * tsky * pixel_area
            integrand_bot = tabp * pixel_area
            integrand_Omega_A = afp * pixel_area
            integral_top += np.sum(integrand_top, axis=1)
            integral_bot += np.sum(integrand_bot, axis=1)
            Omega_A += np.sum(integrand_Omega_A, axis=1)

        # Antenna temperature
        T_ant = integral_top / integral_bot * u.K

        # System temperature
        T_sys = eta * T_ant + (1 - eta) * t0 + T_rec[ii]

        # Beam solid angle
        Omega_A = Omega_A * u.radian**2

        # Effective area
        A_eff = eta * (4 * np.pi * u.radian**2 * c**2 / (eval_freqs[ii].to(u.s**-1) ** 2 * Omega_A))

        # Gain
        G = A_eff / (2 * k_B) * SI_TO_JY

        # SEFD
        sefd = T_sys / G

        # Save the results
        results["T_ant"][:, ii] = T_ant
        results["T_sys"][:, ii] = T_sys
        results["Omega_A"][:, ii] = Omega_A
        results["A_eff"][:, ii] = A_eff
        results["G"][:, ii] = G
        results["SEFD"][:, ii] = sefd

    for key in results:
        logger.info(f"{key} = \n{results[key].to_string()}")

    # TODO: fit polynomials in time/freq to get mean T_sys and G
    # Radiometer equation
    # fc = 0.7
    # npol = 2
    # Smean = snr * fc * tsys / (gain * np.sqrt(npol * df * dt))
    # Smean = Smean.to(u.Jy)
    # logger.info(f"S_mean = {Smean.to_string()}")


if __name__ == "__main__":
    main()
