########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import astropy.units as u
import click
import mwalib
import numpy as np
import psrchive
from astropy.coordinates import AltAz, Angle, SkyCoord
from astropy.time import Time

import mwa_vcs_fluxcal
from mwa_vcs_fluxcal import C0, KB, MWA_LOCATION, SI_TO_JY

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
@click.option("--plot_pb", is_flag=True, help="Plot the primary beam.")
def main(
    archive: str,
    log_level: str,
    metafits: str,
    windowsize: int,
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
    t0 = archive.get_first_Integration().get_start_time()
    t1 = archive.get_last_Integration().get_end_time()
    dt = (t1 - t0).in_seconds() * u.s
    mjdctr = (t0.in_days() + t1.in_days()) / 2
    logger.info(f"fctr={fctr.to_string()}, df={df.to_string()}")
    logger.info(f"mjdctr={mjdctr}, dt={dt.to_string()}")

    # Hardcode these for now
    eval_freq = fctr
    eval_time = mjdctr
    az_range = (Angle(0, u.rad), Angle(2 * np.pi, u.rad))
    za_range = (Angle(0, u.rad), Angle(np.pi / 2, u.rad))
    grid_res = Angle(10, u.arcmin)
    az_subbox_size = 500
    za_subbox_size = 500
    logger.info(f"Grid resolution = {grid_res.to_string()}")

    if plot_pb:
        # Make a low-res map of the primary beam
        grid_stepsize = np.deg2rad(1)
        box_az = np.arange(az_range[0], az_range[1], grid_stepsize)
        box_za = np.arange(za_range[0], za_range[1], grid_stepsize)
        grid_az, grid_za = np.meshgrid(box_az, box_za)
        grid_alt = np.pi / 2 - grid_za
        grid_pbp = mwa_vcs_fluxcal.getPrimaryBeamPower(
            context, eval_freq.to(u.Hz).value, grid_alt.flatten(), grid_az.flatten(), logger=logger
        )["I"].reshape(grid_az.shape)
        mwa_vcs_fluxcal.plot_primary_beam(grid_az, grid_za, grid_pbp, logger=logger)

    # Define a box covering the full range in Az/ZA
    az_box = np.arange(az_range[0].radian, az_range[1].radian, grid_res.radian)
    za_box = np.arange(za_range[0].radian, za_range[1].radian, grid_res.radian)
    logger.info(f"Grid size (az,za) = ({az_box.size},{za_box.size})")

    # Calculate the solid angle pixel size as a column vector
    pixel_size = grid_res * grid_res * np.sin(za_box.reshape(-1, 1))

    # Divide the box into subboxes with maximum size (az_subbox_size, za_subbox_size)
    az_subbox_num = np.ceil(az_box.size / az_subbox_size).astype(int)
    za_subbox_num = np.ceil(za_box.size / za_subbox_size).astype(int)
    az_subboxes = np.array_split(az_box, az_subbox_num)
    za_subboxes = np.array_split(za_box, za_subbox_num)
    logger.info(f"Splitting into {az_subbox_num * za_subbox_num} subgrids")

    # Get target position
    ra_hms, dec_dms = archive.get_coordinates().getHMSDMS().split(" ")
    target_position = SkyCoord(ra_hms, dec_dms, frame="icrs", unit=("hourangle", "deg"))
    time = Time(eval_time, format="mjd")
    altaz_frame = AltAz(location=MWA_LOCATION, obstime=time)
    target_position_altaz = target_position.transform_to(altaz_frame)
    target_psi = mwa_vcs_fluxcal.calcGeometricDelays(
        tile_positions,
        eval_freq.to(u.Hz).value,
        target_position_altaz.alt.rad,
        target_position_altaz.az.rad,
    )

    # Loop through subboxes and integrate
    int_top = 0
    int_bot = 0
    Omega_A = 0
    for ii in range(az_subbox_num):
        az_subbox = az_subboxes[ii]
        for jj in range(za_subbox_num):
            za_subbox = za_subboxes[jj]

            logger.info(f"Computing subbox index az={ii} za={jj}")

            # Map boxes to a 2D grid
            az_subgrid, za_subgrid = np.meshgrid(az_subbox, za_subbox)
            alt_subgrid = np.pi / 2 - za_subgrid
            subgrid_coords = SkyCoord(
                az=Angle(az_subgrid, u.rad),
                alt=Angle(alt_subgrid, u.rad),
                frame="altaz",
                location=MWA_LOCATION,
                obstime=time,
            )

            # Get the interpolated sky temperature
            tsky = mwa_vcs_fluxcal.getSkyTempGrid(
                subgrid_coords, eval_freq.to(u.MHz).value, logger=logger
            )

            # Calculate the primary beam power
            pbp = mwa_vcs_fluxcal.getPrimaryBeamPower(
                context,
                eval_freq.to(u.Hz).value,
                alt_subgrid.flatten(),
                az_subgrid.flatten(),
                logger=logger,
            )["I"].reshape(az_subgrid.shape)

            # Loop through pixels
            afp = np.zeros(shape=az_subgrid.shape, dtype=np.float64)
            tabp = np.zeros(shape=az_subgrid.shape, dtype=np.float64)
            for mm in range(az_subbox.size):
                for nn in range(za_subbox.size):
                    look_psi = mwa_vcs_fluxcal.calcGeometricDelays(
                        tile_positions,
                        eval_freq.to(u.Hz).value,
                        alt_subgrid[nn, mm],
                        az_subgrid[nn, mm],
                    )

                    # Calculate the array factor power
                    afp[nn, mm] = mwa_vcs_fluxcal.calcArrayFactorPower(
                        look_psi, target_psi, logger=logger
                    )

            # Calculate the tied-array beam power
            tabp = afp * pbp

            # Get sin(theta)*d(theta)*d(phi) in rad^2
            pixel_size_subbox = pixel_size.value[
                jj * za_subbox_num : jj * za_subbox_num + za_subbox.size, :
            ]
            pixel_size_grid = np.repeat(pixel_size_subbox, az_subbox.size, axis=1)

            # Compute the integral
            int_top += np.sum(tabp * tsky * pixel_size_grid)
            int_bot += np.sum(tabp * pixel_size_grid)
            Omega_A += np.sum(afp * pixel_size_grid)

    # Antenna temperature
    tant = int_top / int_bot * u.K
    logger.info(f"T_ant = {tant.to_string()}")

    # Beam solid angle
    Omega_A = Omega_A * u.radian**2
    logger.info(f"Omega_A = {Omega_A.to_string()}")

    # Receiver temperature
    trec_spline = mwa_vcs_fluxcal.splineRecieverTemp()
    if plot_trec:
        mwa_vcs_fluxcal.plot_trcvr_vc_freq(trec_spline, fctr, df, logger=logger)
    trec = trec_spline(eval_freq.to(u.MHz).value) * u.K
    logger.info(f"T_rec = {trec.to_string()}")

    # System temperature
    eta = 0.9
    t0 = 290 * u.K
    tsys = eta * tant + (1 - eta) * t0 + trec
    logger.info(f"T_sys = {tsys.to_string()}")

    # Effective area
    Aeff = eta * (4 * np.pi * u.radian**2 * C0**2 / (eval_freq.to(u.s**-1) ** 2 * Omega_A))
    logger.info(f"A_eff = {Aeff.to_string()}")

    # Gain
    gain = Aeff / (2 * KB) * SI_TO_JY
    gain = gain.to(u.K * u.Jy**-1)
    logger.info(f"G = {gain.to_string()}")

    # Radiometer equation
    fc = 0.7
    npol = 2
    Smean = snr * fc * tsys / (gain * np.sqrt(npol * df * dt))
    Smean = Smean.to(u.Jy)
    logger.info(f"S_mean = {Smean.to_string()}")


if __name__ == "__main__":
    main()
