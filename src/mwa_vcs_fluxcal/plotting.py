########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from scipy.interpolate import CubicSpline
from skimage import measure

import mwa_vcs_fluxcal

__all__ = [
    "plot_pulse_profile",
    "plot_trcvr_vc_freq",
    "plot_primary_beam",
    "tesellate_primary_beam",
]

plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "cm"
plt.rcParams["font.size"] = 12


def plot_pulse_profile(
    profile: np.ndarray,
    offpulse_win: np.ndarray,
    offpulse_sigma: float,
    snr: float,
    savename: str = "pulse_profile.png",
    logger: logging.Logger | None = None,
) -> None:
    """Generate a plot of a pulse profile, indicating the noise level and the
    offpulse region.

    Parameters
    ----------
    profile : `np.ndarray`
        The pulse profile amplitudes.
    offpulse_win : `np.ndarray`
        The bin indices of the offpulse region.
    offpulse_sigma : `float`
        The standard deviation of the offpulse noise.
    snr : `float`
        The signal/noise ratio.
    savename : `str`, optional
        The filename to save the plot as. Default: "pulse_profile.png".
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.
    """
    if logger is None:
        logger = mwa_vcs_fluxcal.get_logger()

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300, tight_layout=True)

    lw = 0.8

    num_bin = profile.shape[0]

    bins = np.arange(num_bin) / num_bin
    ax.plot(bins, profile, color="k", linewidth=lw)
    xlims = [bins[0], bins[-1]]
    ylims = ax.get_ylim()

    # Shade the offpulse window
    if offpulse_win is not None:
        offpulse_win = offpulse_win.astype(float) / (num_bin - 1)
        if offpulse_win[0] < offpulse_win[-1]:
            ax.fill_betweenx(
                ylims,
                offpulse_win[0],
                offpulse_win[-1],
                color="tab:blue",
                alpha=0.4,
                zorder=0,
                label="Offpulse region",
            )
        else:
            ax.fill_betweenx(
                ylims,
                offpulse_win[0],
                xlims[-1],
                color="tab:blue",
                alpha=0.4,
                zorder=0,
                label="Offpulse region",
            )
            ax.fill_betweenx(
                ylims, xlims[0], offpulse_win[-1], color="tab:blue", alpha=0.4, zorder=0
            )

    # Plot the noise baseline and shade the standard deviation
    ax.axhline(0, linestyle="--", linewidth=lw, color="k")
    ax.fill_between(
        xlims,
        -offpulse_sigma,
        offpulse_sigma,
        color="k",
        alpha=0.2,
        zorder=0,
        label=f"$\sigma={offpulse_sigma:.6f}$",
    )

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", right=True, top=True, direction="in")
    ax.set_xlabel("Pulse Phase")
    ax.set_ylabel("Flux Density [arb. units]")
    ax.set_title(f"S/N = {snr:.2f}")

    ax.legend()

    logger.info(f"Saving plot file: {savename}")
    fig.savefig(savename)

    plt.close()


def plot_trcvr_vc_freq(
    trcvr_spline: CubicSpline,
    fctr: float,
    df: float,
    savename: str = "trcvr_vs_freq.png",
    logger: logging.Logger | None = None,
) -> None:
    """Plot the receiver temperature as a function of frequency.

    Parameters
    ----------
    trcvr_spline : `CubicSpline`
        A spline fit to the temperatures in Kelvin as a function of frequency in MHz.
    fctr : `float`
        The centre frequency in MHz.
    df : `float`
        The bandwidth in MHz.
    savename : `str`, optional
        The filename to save the plot as. Default: "trcvr_vs_freq.png".
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.
    """
    if logger is None:
        logger = mwa_vcs_fluxcal.get_logger()

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300, tight_layout=True)

    freqs = np.linspace(fctr - df / 2, fctr + df / 2, 1000)

    ax.plot(freqs, trcvr_spline(freqs), linestyle="-", color="k", label="Cubic Spline")
    ylims = ax.get_ylim()

    ax.set_xlim([fctr - df / 2, fctr + df / 2])
    ax.set_ylim(ylims)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", right=True, top=True, direction="in")
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("$T_\mathrm{rec}$ [K]")

    ax.legend()

    logger.info(f"Saving plot file: {savename}")
    fig.savefig(savename)

    plt.close()


def plot_primary_beam(
    grid_az: np.ndarray[float],
    grid_za: np.ndarray[float],
    grid_pbp: np.ndarray[float],
    savename: str = "primary_beam.png",
    logger: logging.Logger | None = None,
) -> None:
    """Plot the primary beam power.

    Parameters
    ----------
    grid_az : `np.ndarray[float]`
        A 2D grid of azimuth angles in radians.
    grid_za : `np.ndarray[float]`
        A 2D grid of zenith angles in radians.
    grid_pbp : `np.ndarray[float]`
        A 2D grid of powers.
    savename : `str`, optional
        The filename to save the plot as. Default: "primary_beam.png".
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.
    """
    if logger is None:
        logger = mwa_vcs_fluxcal.get_logger()

    cmap = plt.get_cmap("cmr.arctic_r")
    cmap.set_under(color="w")
    contour_levels = [0.01, 0.1, 0.5, 0.9]

    fig = plt.figure(figsize=(6, 5), dpi=300, tight_layout=True)
    ax = fig.add_subplot(projection="polar")
    im = ax.pcolormesh(
        grid_az,
        grid_za,
        grid_pbp,
        vmax=1.0,
        vmin=0.01,
        rasterized=True,
        shading="auto",
        cmap=cmap,
    )
    ax.contour(grid_az, grid_za, grid_pbp, contour_levels, colors="k", linewidths=1, zorder=1e2)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(157.5)
    ax.grid(ls=":", color="0.5")
    ax.set_ylim(np.radians([0, 90]))
    ax.set_yticks(np.radians([20, 40, 60, 80]))
    ax.set_yticklabels(
        ["${}^\\circ$".format(int(x)) for x in np.round(np.degrees(ax.get_yticks()), 0)]
    )
    ax.set_xlabel("Azimuth angle [deg]", labelpad=5)
    ax.set_ylabel("Zenith angle [deg]", labelpad=30)
    cbar = plt.colorbar(
        im,
        pad=0.13,
        extend="min",
        ticks=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    cbar.ax.set_ylabel("Zenith-normalised beam power", labelpad=10)
    cbar.ax.tick_params(labelsize=10)
    for contour_level in contour_levels:
        cbar.ax.axhline(contour_level, color="k", lw=1)

    logger.info(f"Saving plot file: {savename}")
    fig.savefig(savename)

    plt.close()


def tesellate_primary_beam(
    grid_az: np.ndarray,
    grid_za: np.ndarray,
    grid_pbp: np.ndarray,
    res: float,
    savename: str = "primary_beam_tesselation.png",
    logger: logging.Logger | None = None,
) -> None:
    """Tesellate the primary beam by finding the pixels inside contours.

    Parameters
    ----------
    grid_az : `np.ndarray[float]`
        A 2D grid of azimuth angles in radians.
    grid_za : `np.ndarray[float]`
        A 2D grid of zenith angles in radians.
    grid_pbp : `np.ndarray[float]`
        A 2D grid of powers.
    res : `float`
        The resolution of the grid in radians.
    savename : `str`, optional
        The filename to save the plot as. Default: "primary_beam_tesselation.png".
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.
    """
    cmap = plt.get_cmap("cmr.arctic_r")
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(5, 8),
        dpi=400,
        tight_layout=True,
        subplot_kw={"projection": "polar"},
    )

    axes[0].pcolormesh(
        grid_az,
        grid_za,
        grid_pbp,
        vmax=1.0,
        vmin=0.01,
        rasterized=True,
        shading="auto",
        cmap=cmap,
    )

    # pbp/xv/yv have dimension order (za, az) which we will label (y, x)
    yv, xv = np.meshgrid(np.arange(grid_pbp.shape[1]), np.arange(grid_pbp.shape[0]))

    # Create an (N,2) array of pixel coordinates
    points = np.vstack((xv.ravel(), yv.ravel())).T

    contours = measure.find_contours(grid_pbp, 0.001)

    mask = np.full(shape=grid_pbp.shape, fill_value=False)
    for contour in contours:
        contour_fill = contour
        if np.isclose((contour[0, 1] - contour[-1, 1]) * res, 2 * np.pi, atol=np.deg2rad(1)):
            az_l, az_r = 0, grid_pbp.shape[1]
            za_b, za_ul, za_ur = 0, contour[0, 0], contour[-1, 0]
            loop = [[za_ul, az_l], [za_b, az_l], [za_b, az_r], [za_ur, az_r]]
            contour_fill = np.append(contour, loop, axis=0)

        path = Path(contour_fill)
        submask = path.contains_points(points, radius=-1).reshape(grid_pbp.shape)
        mask = np.logical_or(mask, submask)
        for ax in axes:
            ax.plot(
                contour[:, 1] * res,
                contour[:, 0] * res,
                color="tab:red",
                linewidth=0.7,
                alpha=0.7,
            )

    axes[1].pcolormesh(grid_az, grid_za, mask, rasterized=True, shading="auto", cmap=cmap)

    for ax in axes:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(157.5)
        ax.grid(visible=False)
        # ax.grid(ls=":", lw=0.5, color="0.5")
        ax.set_ylim(np.radians([0, 90]))
        ax.set_yticks(np.radians([20, 40, 60, 80]))
        ax.set_yticklabels([])
        ax.set_xlabel("Azimuth Angle [deg]", labelpad=5)
        ax.set_ylabel("Zenith Angle [deg]", labelpad=30)

    logger.info(f"Saving plot file: {savename}")
    fig.savefig(savename)

    plt.close()
