########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.interpolate import CubicSpline, RegularGridInterpolator

import mwa_vcs_fluxcal

__all__ = [
    "plot_pulse_profile",
    "plot_trcvr_vc_freq",
    "plot_primary_beam",
    "plot_tied_array_beam",
    "plot_sky_images",
    "plot_3d_result",
]

plt.rcParams["mathtext.fontset"] = "dejavuserif"
# plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "cm"
plt.rcParams["font.size"] = 12


def plot_pulse_profile(
    profile: np.ndarray,
    noise_profile: np.ndarray | None,
    offpulse_win: np.ndarray | None,
    offpulse_std: float | None,
    ylabel: str = "Flux Density",
    title: str = None,
    savename: str = "pulse_profile.png",
    logger: logging.Logger | None = None,
) -> None:
    """Generate a plot of a pulse profile, indicating the noise level and the
    offpulse region.

    Parameters
    ----------
    profile : `np.ndarray`
        The pulse profile amplitudes.
    noise_profile : `np.ndarray`, optional
        The profile from which the noise was computed. Default: None.
    offpulse_win : `np.ndarray`, optional
        The bin indices of the offpulse region. Default: None.
    offpulse_std : `float`, optional
        The standard deviation of the offpulse noise. Default: None.
    ylabel : `str`, optional
        The y-axis label. Default: "Flux Density".
    title : `str`, optional
        A title for the plot. Default: None.
    savename : `str`, optional
        The filename to save the plot as. Default: "pulse_profile.png".
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.
    """
    if logger is None:
        logger = mwa_vcs_fluxcal.get_logger()

    num_bin = profile.shape[0]
    bins = np.arange(num_bin) / (num_bin - 1)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300, tight_layout=True)

    lw = 0.6

    ax.plot(
        np.concatenate([bins, bins + 1]),
        np.concatenate([profile, profile]),
        color="k",
        linewidth=lw,
        zorder=1,
    )

    if noise_profile is not None:
        ax.plot(
            np.concatenate([bins, bins + 1]),
            np.concatenate([noise_profile, noise_profile]),
            color="tab:red",
            linewidth=lw,
            zorder=2,
        )

    xlims = [0, 2]
    ylims = ax.get_ylim()

    if offpulse_win is not None:
        # Shade the offpulse window
        shade_args = dict(
            color="tab:blue",
            alpha=0.4,
            zorder=0,
        )
        offpulse_win = offpulse_win.astype(float) / (num_bin - 1)
        if offpulse_win[0] < offpulse_win[-1]:
            ax.fill_betweenx(ylims, offpulse_win[0], offpulse_win[-1], **shade_args)
            ax.fill_betweenx(ylims, offpulse_win[0] + 1, offpulse_win[-1] + 1, **shade_args)
        else:
            ax.fill_betweenx(ylims, 0, offpulse_win[-1], **shade_args)
            ax.fill_betweenx(ylims, offpulse_win[0], offpulse_win[-1] + 1, **shade_args)
            ax.fill_betweenx(ylims, offpulse_win[0] + 1, 2, **shade_args)

    if offpulse_std is not None:
        # Plot the noise baseline and shade the standard deviation
        ax.axhline(0, linestyle="--", linewidth=lw, color="k")
        ax.fill_between(
            xlims,
            -offpulse_std,
            offpulse_std,
            color="k",
            alpha=0.2,
            zorder=0,
        )

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", right=True, top=True, direction="in")
    ax.set_xlabel("Pulse Phase")
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

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


def plot_tied_array_beam(
    grid_az: np.ndarray[float],
    grid_za: np.ndarray[float],
    grid_tabp: np.ndarray[float],
    savename: str = "tied_array_beam.png",
    logger: logging.Logger | None = None,
) -> None:
    """Plot the tied array beam power.

    Parameters
    ----------
    grid_az : `np.ndarray[float]`
        A 2D grid of azimuth angles in radians.
    grid_za : `np.ndarray[float]`
        A 2D grid of zenith angles in radians.
    grid_tabp : `np.ndarray[float]`
        A 2D grid of powers.
    savename : `str`, optional
        The filename to save the plot as. Default: "tied_array_beam.png".
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.
    """
    if logger is None:
        logger = mwa_vcs_fluxcal.get_logger()

    cmap = plt.get_cmap("cmr.arctic_r")
    cmap.set_under(color="w")

    fig = plt.figure(figsize=(6, 5), dpi=300, tight_layout=True)
    ax = fig.add_subplot(projection="polar")
    im = ax.pcolormesh(
        grid_az,
        grid_za,
        grid_tabp,
        rasterized=True,
        shading="auto",
        cmap=cmap,
    )
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
    cbar = plt.colorbar(im, pad=0.13, extend="min")
    cbar.ax.set_ylabel("Zenith-normalised beam power", labelpad=10)
    cbar.ax.tick_params(labelsize=10)

    logger.info(f"Saving plot file: {savename}")
    fig.savefig(savename)

    plt.close()


def plot_sky_images(
    grid_az: np.ndarray[float],
    grid_za: np.ndarray[float],
    grid_list: list,
    label_list: list,
    pulsar_coords: SkyCoord = None,
    savename: str = "sky_images.png",
    logger: logging.Logger | None = None,
) -> None:
    """Plot multiple arrays on polar Az/ZA skymaps.

    Parameters
    ----------
    grid_az : `np.ndarray[float]`
        A 2D grid of azimuth angles in radians.
    grid_za : `np.ndarray[float]`
        A 2D grid of zenith angles in radians.
    grid_list : `list`
        A list of 2D data grids to plot.
    label_list : `list`
        A list of data labels to use for the colour bars.
    pulsar_coords : `SkyCoord`, optional
        The coordinates of the target pulsar to plot in the beam. Default: None.
    savename : `str`, optional
        The filename to save the plot as. Default: "sky_images.png".
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.
    """
    if logger is None:
        logger = mwa_vcs_fluxcal.get_logger()

    cmap = plt.get_cmap("cmr.arctic_r")
    cmap.set_under(color="w")

    num_images = len(grid_list)

    fig, axes = plt.subplots(
        ncols=num_images,
        figsize=(5 * num_images, 6),
        dpi=300,
        tight_layout=True,
        subplot_kw={"projection": "polar"},
    )
    if type(axes) is not np.ndarray:
        axes = np.array([axes])

    im_list = []
    for ii, grid_data in enumerate(grid_list):
        im = axes[ii].pcolormesh(
            grid_az,
            grid_za,
            grid_data,
            rasterized=True,
            shading="auto",
            cmap=cmap,
        )
        im_list.append(im)

    for ax, im, lab in zip(axes, im_list, label_list, strict=True):
        if pulsar_coords is not None:
            ax.plot(
                pulsar_coords.az.radian,
                np.pi / 2 - pulsar_coords.alt.radian,
                linestyle="none",
                marker="o",
                color="tab:red",
                ms=3,
                mfc="none",
            )

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
        cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.13)
        cbar.ax.set_xlabel(lab, labelpad=10)
        cbar.ax.tick_params(labelsize=10)

    logger.info(f"Saving plot file: {savename}")
    fig.savefig(savename)

    plt.close()


def plot_3d_result(
    t: np.ndarray[float],
    f: np.ndarray[float],
    d: np.ndarray[float],
    zlabel: str,
    num_points: int = 20,
    savename: str = "results.png",
    logger: logging.Logger | None = None,
) -> RegularGridInterpolator:
    """Make a 3D plot of the (t,f) parameter space for data d.

    Parameters
    ----------
    t : `np.ndarray[float]`
        The points defining the regular grid in dimension t.
    f : `np.ndarray[float]`
        The points defining the regular grid in dimension f.
    d : `np.ndarray[float]`
        The data on the regular grid in 2 dimensions.
    zlabel : `str`
        The z-axis label associated with the data.
    num_points : `int`, optional
        The number of points to interpolate the data to in each dimension. Default: 20.
    savename : `str`, optional
        The filename to save the plot as. Default: "results.png".
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.

    Returns
    -------
    interp : `RegularGridInterpolator`
        A function defining the interpolated 2D surface.
    """
    if logger is None:
        logger = mwa_vcs_fluxcal.get_logger()

    tg, fg = np.meshgrid(t, f, indexing="ij")
    interp = RegularGridInterpolator((t, f), d, method="cubic")
    tt = np.linspace(np.min(t), np.max(t), num_points)
    ff = np.linspace(np.min(f), np.max(f), num_points)
    ttg, ffg = np.meshgrid(tt, ff, indexing="ij")
    ddg = interp((ttg, ffg))

    fig, ax = plt.subplots(dpi=300, subplot_kw={"projection": "3d"})
    ax.scatter(tg.ravel(), fg.ravel(), d.ravel(), s=20, c="k")
    ax.plot_wireframe(ttg, ffg, ddg, alpha=0.4)
    ax.set(
        xlabel="Time [s]",
        ylabel="Frequency [MHz]",
        zlabel=zlabel,
    )
    logger.info(f"Saving plot file: {savename}")
    fig.savefig(savename)
    plt.close()

    return interp
