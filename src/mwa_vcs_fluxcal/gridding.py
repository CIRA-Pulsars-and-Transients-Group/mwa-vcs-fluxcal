########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from matplotlib.path import Path
from skimage import measure

__all__ = ["tesellate_primary_beam", "upsample_blocks"]


plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "cm"
plt.rcParams["font.size"] = 12


def tesellate_primary_beam(
    az: np.ndarray[float],
    za: np.ndarray[float],
    pbp: np.ndarray[float],
    res: float,
    plevel: float = 0.001,
    plot: bool = True,
    pulsar_coords: SkyCoord = None,
    savename: str = "primary_beam_masked.png",
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tesellate the primary beam by finding pixels inside iso-power contours.

    Parameters
    ----------
    az : `np.ndarray[float]`
        A 2D grid of azimuth angles in radians.
    za : `np.ndarray[float]`
        A 2D grid of zenith angles in radians.
    pbp : `np.ndarray[float]`
        A 2D grid of powers.
    res : `float`
        The resolution of the grid in radians.
    plevel : `float`
        The zenith-normalised power level to use to find the contours.
    plot : `bool`
        Make a plot showing the masked primary beam.
    pulsar_coords : `SkyCoord`, optional
        The coordinates of the target pulsar to plot in the beam. Default: None.
    savename : `str`, optional
        The filename to save the plot as. Default: "primary_beam_masked.png".
    logger : `logging.Logger`, optional
        A logger to use. Default: `None`.

    Returns
    -------
    az_inbeam : `np.ndarray[float | np.nan]`
        A duplicate of `az` with pixels outside of the beam set to NaN.
    za_inbeam : `np.ndarray[float | np.nan]`
        A duplicate of `za` with pixels outside of the beam set to NaN.
    pbp_inbeam : `np.ndarray[float | np.nan]`
        A duplicate of `pbp` with pixels outside of the beam set to NaN.
    """
    # Grid has dimensions (za, az)
    za_idx, az_idx = np.meshgrid(np.arange(pbp.shape[1]), np.arange(pbp.shape[0]))

    # Create an (N,2) array of pixel array indices
    points = np.vstack((az_idx.ravel(), za_idx.ravel())).T

    # Find iso-power contours of primary beam
    contours = measure.find_contours(pbp, plevel)

    # This mask defines which pixels are inside (True) and outside (False) the
    # iso-contours of the primary beam
    mask = np.full(shape=pbp.shape, fill_value=False)

    # The pixels inside each contour must be found separately
    for contour in contours:
        # If the contour encloses the polar origin, add path points at ZA=0 so
        # that point finding algorithm recognises the sky area near the origin
        contour_fill = contour
        if np.isclose((contour[0, 1] - contour[-1, 1]) * res, 2 * np.pi, rtol=0.1):
            az_l, az_r = 0, pbp.shape[1]
            za_b, za_ul, za_ur = 0, contour[0, 0], contour[-1, 0]
            loop = [[za_ul, az_l], [za_b, az_l], [za_b, az_r], [za_ur, az_r]]
            contour_fill = np.append(contour, loop, axis=0)

        # Find pixels inside the contour using a matplotlib Path
        path = Path(contour_fill)
        submask = path.contains_points(points, radius=-1).reshape(pbp.shape)

        # Update the overall pixel mask
        mask = np.logical_or(mask, submask)

    # Mask the input arrays
    az_inbeam = np.where(mask, az, np.nan)
    za_inbeam = np.where(mask, za, np.nan)
    pbp_inbeam = np.where(mask, pbp, np.nan)

    # Make a plot showing the masked primary beam power
    if plot:
        cmap = plt.get_cmap("cmr.arctic")
        fig, ax = plt.subplots(
            figsize=(6, 5), dpi=300, tight_layout=True, subplot_kw={"projection": "polar"}
        )
        im = ax.pcolormesh(az, za, np.log10(pbp_inbeam), rasterized=True, shading="auto", cmap=cmap)
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
        cbar = plt.colorbar(im, pad=0.13, ticks=[-4, -3, -2, -1, 0])
        cbar.ax.set_ylim([np.log10(plevel), 0])
        cbar.ax.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
        cbar.ax.set_ylabel("Zenith-normalised beam power (\%)", labelpad=10)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(157.5)
        ax.grid(ls=":", color="0.5", lw=0.5)
        ax.set_ylim(np.radians([0, 90]))
        ax.set_yticks(np.radians([20, 40, 60, 80]))
        ax.set_yticklabels(
            ["${}^\\circ$".format(int(x)) for x in np.round(np.degrees(ax.get_yticks()), 0)]
        )
        ax.set_xlabel("Azimuth Angle [deg]", labelpad=5)
        ax.set_ylabel("Zenith Angle [deg]", labelpad=30)
        logger.info(f"Saving plot file: {savename}")
        fig.savefig(savename)
        plt.close()

    return az_inbeam, za_inbeam, pbp_inbeam


def upsample_blocks(
    az_coarse: np.ndarray, za_coarse: np.ndarray, coarse_res: float, fine_res: float
) -> tuple[np.ndarray, np.ndarray]:
    """Up-sample gridded data.

    Parameters
    ----------
    az_coarse : `np.ndarray`
        The lower azimuth edge of the coarse-gridded pixels.
    za_coarse : `np.ndarray`
        The lower zenith edge of the coarse-gridded pixels.
    coarse_res : `float`
        The width/height of the coarse pixels (assume square pixels).
    fine_res : `float`
        The width/height of the fine pixels (assume square pixels).

    Returns
    -------
    az_fine : `np.ndarray`
        The lower azimuth edge of the fine-gridded pixels.
    za_fine : `np.ndarray`
        The lower zenith edge of the fine-gridded pixels.
    """
    ratio = int(coarse_res / fine_res)
    num_blocks = az_coarse.size
    num_pixels = num_blocks * ratio**2
    az_fine = np.empty(num_pixels, dtype=np.float64)
    za_fine = np.empty(num_pixels, dtype=np.float64)
    for ii in range(num_blocks):
        az_fine_grid, za_fine_grid = np.meshgrid(
            az_coarse[ii] + np.arange(ratio) * fine_res,
            za_coarse[ii] + np.arange(ratio) * fine_res,
        )
        az_fine[ii * ratio**2 : (ii + 1) * ratio**2] = az_fine_grid.flatten()
        za_fine[ii * ratio**2 : (ii + 1) * ratio**2] = za_fine_grid.flatten()
    return az_fine, za_fine
