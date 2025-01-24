########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import logging

import matplotlib.pyplot as plt
import numpy as np

import mwa_vcs_fluxcal

__all__ = ["plot_pulse_profile", "plot_trcvr_vc_freq"]

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
    trcvr_spline,
    fctr: float,
    df: float,
    savename: str = "trcvr_vs_freq.png",
    logger: logging.Logger | None = None,
):
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
