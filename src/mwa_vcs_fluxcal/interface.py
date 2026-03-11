########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging

import astropy.units as u
import enlighten
import mwalib
import numpy as np
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time

from .integral import compute_sky_integrals
from .plotting import plot_3d_result, plot_trcvr_vs_freq
from .temperatures import getAmbientTemp, splineRecieverTemp

__all__ = ["simulate_sefd"]

logger = logging.getLogger(__name__)


def simulate_sefd(
    metafits: str,
    target_coords: SkyCoord,
    start_time_offset: float,
    end_time_offset: float,
    fine_grid_res: float = 2.0,
    coarse_grid_res: float = 30.0,
    min_pbp: float = 0.001,
    nfreq: int = 1,
    ntime: int = 1,
    max_pix_per_job: int = int(10**5),
    fc: float = 1.43,
    eta: float = 0.98,
    plot_trec: bool = False,
    plot_pb: bool = False,
    plot_tab: bool = False,
    plot_tsky: bool = False,
    plot_integrals: bool = False,
    plot_3d: bool = False,
    file_prefix: str = "sim",
) -> dict[str, u.Quantity]:
    # Load metadata
    logger.info(f"Loading metafits file: {metafits}")
    context = mwalib.MetafitsContext(metafits)
    T_amb = getAmbientTemp(metafits)
    chan_freqs_hz = context.metafits_fine_chan_freqs_hz
    fctr = (np.min(chan_freqs_hz) + np.max(chan_freqs_hz)) / 2 / 1e6 * u.MHz
    bw = context.obs_bandwidth_hz / 1e6 * u.MHz
    obs_start = Time(context.sched_start_mjd, format="mjd")
    obs_end = Time(context.sched_end_mjd, format="mjd")
    obs_length = (obs_end - obs_start).to(u.s).round(3)
    if start_time_offset is None:
        start_time_offset = 0 * u.s
    else:
        start_time_offset *= u.s
    if end_time_offset is None:
        end_time_offset = obs_length
    else:
        end_time_offset *= u.s
    start_time_frac = start_time_offset / obs_length
    end_time_frac = end_time_offset / obs_length

    # Input checks
    if end_time_offset < start_time_offset:
        raise ValueError("Start time offset must proceed end time offset.")
    if start_time_frac < 0.0 or start_time_frac > 1.0:
        raise ValueError(
            f"Fractional start time must be in the range [0,1] (provided {start_time_frac})."
        )
    if end_time_frac < 0.0 or end_time_frac > 1.0:
        raise ValueError(
            f"Fractional end time must be in the range [0,1] (provided {end_time_frac})."
        )
    if min_pbp < 0.0 or min_pbp >= 1:
        raise ValueError(f"Primary beam power must in the range [0,1) (provided {min_pbp}).")
    if fc < 1.0:
        raise ValueError(f"fc must be >= 1 (provided {fc}).")
    if eta > 1.0:
        raise ValueError(f"eta must be <= 1 (provided {eta}).")

    # Check grid resolution is valid
    fine_grid_res = Angle(fine_grid_res, u.arcmin)
    coarse_grid_res = Angle(coarse_grid_res, u.arcmin)
    if not np.isclose((coarse_grid_res.arcmin / fine_grid_res.arcmin) % 1, 0.0, rtol=1e-5):
        logger.critical("Coarse grid resolution not divisible by fine grid resolution.")
        exit(1)

    # Get frequency information
    fbot = (fctr - bw / 2).to(u.MHz).round(3)
    ftop = (fctr + bw / 2).to(u.MHz).round(3)
    logger.info(f"Frequency range (MHz): {fbot.value} to {ftop.value}")

    # Get time information
    start_time = obs_start + start_time_offset
    end_time = obs_start + end_time_offset
    int_time = end_time_offset - start_time_offset
    logger.info(f"Time offset range (s): {start_time_offset.value} to {end_time_offset.value}")
    logger.info(f"Time range (MJD): {start_time.mjd} to {end_time.mjd}")
    logger.info(f"Time range (GPS): {start_time.gps:.0f} to {end_time.gps:.0f}")

    # Simulation frequencies
    if nfreq == 1:
        eval_freqs = np.array([fctr.to(u.MHz).value]) * u.MHz
    else:
        eval_freqs = np.linspace(fbot.to(u.MHz).value, ftop.to(u.MHz).value, nfreq) * u.MHz
    logger.info(f"Evaluating at {nfreq} frequencies: {eval_freqs}")

    # Simulation times relative to time t0
    if ntime == 1:
        eval_offsets = np.array([int_time.to(u.s).value / 2]) * u.s
    else:
        eval_offsets = np.linspace(0, int_time.to(u.s).value, ntime) * u.s
    logger.info(f"Evaluating at {ntime} offsets: {eval_offsets}")

    if plot_trec:
        # Plot the receiver temperature vs frequency
        T_rec_spline = splineRecieverTemp()
        plot_trcvr_vs_freq(
            T_rec_spline,
            fctr.to(u.MHz).value,
            bw.to(u.MHz).value,
            savename=f"{file_prefix}_trcvr_vs_freq.png",
        )

    # Compute the sky integrals required to get T_sys and gain
    # The progress bar will only be shown if stdout is attached to a TTY
    with enlighten.get_manager() as manager:
        results = compute_sky_integrals(
            context,
            start_time,
            eval_offsets,
            eval_freqs,
            target_coords,
            fine_grid_res,
            coarse_grid_res,
            min_pbp,
            max_pix_per_job=max_pix_per_job,
            plot_pb=plot_pb,
            plot_tab=plot_tab,
            plot_tsky=plot_tsky,
            plot_integrals=plot_integrals,
            fc=fc,
            eta=eta,
            T_amb=T_amb,
            file_prefix=file_prefix,
            pbar_manager=manager,
        )

    if plot_3d and nfreq >= 4 and ntime >= 4:
        # Fit a 2D spline to show the freq/time scaling of T_sys, gain, and SEFD
        plot_3d_result(
            eval_offsets.to(u.s).value,
            eval_freqs.to(u.MHz).value,
            results["T_sys"].value,
            zlabel="$T_\mathrm{sys}$ [K]",
            savename=f"{file_prefix}_3d_tsys.png",
        )
        plot_3d_result(
            eval_offsets.to(u.s).value,
            eval_freqs.to(u.MHz).value,
            results["G"].value,
            zlabel="Gain [K/Jy]",
            savename=f"{file_prefix}_3d_gain.png",
        )
        plot_3d_result(
            eval_offsets.to(u.s).value,
            eval_freqs.to(u.MHz).value,
            results["SEFD"].value,
            zlabel="SEFD [Jy]",
            savename=f"{file_prefix}_3d_sefd.png",
        )

    return results
