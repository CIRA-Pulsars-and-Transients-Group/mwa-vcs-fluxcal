########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.interpolate import CubicSpline, RegularGridInterpolator

__all__ = [
    "plot_trcvr_vs_freq",
    "plot_sky_images",
    "plot_3d_result",
]

logger = logging.getLogger(__name__)

plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 12
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.family"] = "serif"

USE_LATEX = False
USE_DARKMODE = False

if USE_LATEX:
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.serif"] = "cm"

if USE_DARKMODE:
    plt.style.use("dark_background")
    LINE_COLOUR = "w"
    BG_COLOUR = "k"
    CMAP = "cmr.arctic"
else:
    LINE_COLOUR = "k"
    BG_COLOUR = "w"
    CMAP = "cmr.arctic_r"


def plot_trcvr_vs_freq(
    trcvr_spline: CubicSpline,
    fctr: float,
    df: float,
    savename: str = "trcvr_vs_freq.webp",
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
        The filename to save the plot as. Default: "trcvr_vs_freq.webp".
    """
    fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)

    freqs = np.linspace(fctr - df / 2, fctr + df / 2, 1000)

    ax.plot(freqs, trcvr_spline(freqs), linestyle="-", color=LINE_COLOUR, label="Cubic Spline")
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


def plot_sky_images(
    grid_az: np.ndarray[float],
    grid_za: np.ndarray[float],
    grid_list: list[np.ndarray[float]],
    label_list: list[str],
    title_list: list[str] | None = None,
    vrange_list: list[tuple[float | None, float | None] | None] | None = None,
    contour_grid_list: list[np.ndarray[float]] | None = None,
    contour_levels: int | list[float] | None = 1,
    extend_list: list[str] | None = None,
    target_coords: SkyCoord = None,
    other_target_coords: SkyCoord = None,
    savename: str = "sky_images.webp",
) -> None:
    """Plot multiple arrays on polar Az/ZA skymaps.

    Parameters
    ----------
    grid_az : `np.ndarray[float]`
        A 2D grid of azimuth angles in radians.
    grid_za : `np.ndarray[float]`
        A 2D grid of zenith angles in radians.
    grid_list : `list[np.ndarray[float]]`
        A list of 2D data grids to plot.
    label_list : `list[str]`
        A list of data labels to use for the colour bars.
    title_list : `list[str]`, optional
        A list of plot titles.
    vrange_list : `list[tuple[float | None, float | None] | None]`, optional
        A list of `(vmin, vmax)` pairs, specifying the dynamic range of the
        image. Either limit can be left as None, or a whole tuple can be left
        as None. If None, then the limits will be automatically chosen.
    contour_grid_list : `list[array_like[float]]`, optional
        A list of 2D data grids to use to calculate contours. For example,
        to plot the primary beam contours. Default: None.
    contour_levels : `int | list[float]`, optional
        The contour levels to draw if contour_grid_list is provided. This
        will be passed directly to `plt.contour`. Default: 1.
    extend_list : `str`, optional
        A list of strings specifying whether to make pointed end(s) for
        out-of-range values in the colorbar for each image. The options
        are "min", "max", or "neither". Default: "neither" for all.
    target_coords : `SkyCoord`, optional
        The coordinates of the primary target to plot. Default: None.
    other_target_coords : `SkyCoord`, optional
        The coordinates of a secondary target to plot. Default: None.
    savename : `str`, optional
        The filename to save the plot as. Default: "sky_images.webp".
    """
    cmap = plt.get_cmap(CMAP)
    cmap.set_under(color=BG_COLOUR)

    num_images = len(grid_list)

    if title_list is None:
        title_list = [None] * num_images

    if vrange_list is None:
        vrange_list = [None] * num_images

    if contour_grid_list is None:
        contour_grid_list = [None] * num_images

    if extend_list is None:
        extend_list = ["neither"] * num_images

    fig, axes = plt.subplots(
        ncols=num_images,
        figsize=(5 * num_images, 6),
        layout="tight",
        subplot_kw={"projection": "polar"},
    )
    if type(axes) is not np.ndarray:
        axes = np.array([axes])

    for ax, grid_data, lab, title, vrange, cgrid_data, extend in zip(
        axes,
        grid_list,
        label_list,
        title_list,
        vrange_list,
        contour_grid_list,
        extend_list,
        strict=True,
    ):
        if vrange is not None:
            vmin, vmax = vrange
        else:
            vmin = None
            vmax = None

        if title is not None:
            ax.set_title(title, pad=10)

        im = ax.pcolormesh(
            grid_az,
            grid_za,
            grid_data,
            rasterized=True,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        if cgrid_data is not None:
            ax.contour(
                grid_az,
                grid_za,
                cgrid_data,
                contour_levels,
                colors=LINE_COLOUR,
                linewidths=0.7,
                negative_linestyles="solid",
                zorder=1e2,
            )

        if target_coords is not None:
            ax.plot(
                target_coords.az.radian,
                np.pi / 2 - target_coords.alt.radian,
                linestyle="none",
                marker="o",
                color="r",
                ms=3,
                mfc="none",
            )

        if other_target_coords is not None:
            ax.plot(
                other_target_coords.az.radian,
                np.pi / 2 - other_target_coords.alt.radian,
                linestyle="none",
                marker="o",
                color="lime",
                ms=3,
                mfc="none",
            )

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(157.5)
        ax.grid(ls=":", color=LINE_COLOUR)
        ax.set_ylim(np.radians([0, 90]))
        ax.set_yticks(np.radians([20, 40, 60, 80]))
        ax.set_yticklabels(
            ["${}^\\circ$".format(int(x)) for x in np.round(np.degrees(ax.get_yticks()), 0)]
        )
        # ax.set_xlabel("Azimuth angle [deg]", labelpad=5)
        # ax.set_ylabel("Zenith angle [deg]", labelpad=30)
        cbar = plt.colorbar(im, ax=ax, extend=extend, orientation="horizontal", pad=0.1)
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
    savename: str = "results.webp",
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
        The filename to save the plot as. Default: "results.webp".

    Returns
    -------
    interp : `RegularGridInterpolator`
        A function defining the interpolated 2D surface.
    """
    tg, fg = np.meshgrid(t, f, indexing="ij")
    interp = RegularGridInterpolator((t, f), d, method="cubic")
    tt = np.linspace(np.min(t), np.max(t), num_points)
    ff = np.linspace(np.min(f), np.max(f), num_points)
    ttg, ffg = np.meshgrid(tt, ff, indexing="ij")
    ddg = interp((ttg, ffg))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(tg.ravel(), fg.ravel(), d.ravel(), s=20, c=LINE_COLOUR)
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
