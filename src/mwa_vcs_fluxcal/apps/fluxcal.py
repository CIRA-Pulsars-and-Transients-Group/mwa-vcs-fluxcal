########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

import click
from astropy.coordinates import SkyCoord

from mwa_vcs_fluxcal import __version__
from mwa_vcs_fluxcal.interface import simulate_sefd
from mwa_vcs_fluxcal.logger import log_levels, setup_logger
from mwa_vcs_fluxcal.utils import qty_dict_to_toml


@click.command()
@click.help_option("-h", "--help")
@click.version_option(__version__, "-V", "--version")
@click.option(
    "-L",
    "log_level",
    type=click.Choice(log_levels.keys(), case_sensitive=False),
    default="info",
    show_default=True,
    help="The logger verbosity level.",
)
@click.option(
    "-m",
    "--metafits",
    "metafits",
    type=click.Path(exists=True),
    help="A metafits file for the beamformed observation.",
    required=True,
)
@click.option(
    "-t",
    "--target",
    type=str,
    help="The target's RA/Dec in hour/deg units in any format accepted by SkyCoord.",
    required=True,
)
@click.option(
    "-s",
    "--start_offset",
    type=float,
    help="The difference (in seconds) between the scheduled observation start "
    + "time (the obs ID) and the start time of the data being calibrated. "
    + "Will override the start time from the metafits.",
)
@click.option(
    "-i",
    "--int_time",
    type=float,
    help="The integration time (in seconds) of the data being calibrated. "
    + "Will override the integration time from the metafits.",
)
@click.option(
    "--fine_res",
    type=float,
    default=2.0,
    show_default=True,
    help="The resolution of the integral, in arcmin.",
)
@click.option(
    "--coarse_res",
    type=float,
    default=30.0,
    show_default=True,
    help="The resolution of the primary beam map, in arcmin. Must be an integer "
    + "multiple of --fine_res.",
)
@click.option(
    "--min_pbp",
    type=click.FloatRange(0.0, 1.0),
    default=0.001,
    show_default=True,
    help="Only integrate above this primary beam power.",
)
@click.option(
    "--nfreq",
    type=int,
    default=1,
    show_default=True,
    help="The number of frequency steps to simulate.",
)
@click.option(
    "--ntime",
    type=int,
    default=1,
    show_default=True,
    help="The number of time steps to simulate.",
)
@click.option(
    "--freq_list",
    type=str,
    show_default=True,
    help="A comma-separated list of frequencies to simulate (in MHz).",
)
@click.option(
    "--time_list",
    type=str,
    show_default=True,
    help="A comma-separated list of times to simulate (in sec relative to the start time).",
)
@click.option(
    "--max_pix_per_job",
    type=int,
    default=10**5,
    show_default=True,
    help="The maximum number of sky area pixels to compute per job.",
)
@click.option(
    "--fc",
    type=click.FloatRange(1.0, 10.0),
    default=1.43,
    show_default=True,
    help="The beamforming coherency factor.",
)
@click.option(
    "--eta",
    type=click.FloatRange(0.0, 1.0),
    default=0.98,
    show_default=True,
    help="The radiation efficiency of the array.",
)
@click.option("-f", "--file_prefix", type=str, help="The prefix of the output file names.")
@click.option("--plot_trec", is_flag=True, help="Plot the receiver temperature.")
@click.option("--plot_pb", is_flag=True, help="Plot the primary beam in Alt/Az.")
@click.option("--plot_tab", is_flag=True, help="Plot the tied-array beam in Alt/Az.")
@click.option("--plot_tsky", is_flag=True, help="Plot sky temperature in Alt/Az.")
@click.option("--plot_integrals", is_flag=True, help="Plot the integral quantities in Alt/Az.")
@click.option("--plot_3d", is_flag=True, help="Plot the results in 3D (time,freq,data).")
@click.option(
    "--extra_tile_flags", type=str, help="A comma-separated list of tile names or IDs to flag."
)
@click.option(
    "--cal_metafits",
    type=click.Path(exists=True),
    help="A metafits file for a calibration observation to use for getting extra flagged tiles.",
)
def main(
    log_level: str,
    metafits: str,
    target: str,
    start_offset: float | None,
    int_time: float | None,
    fine_res: float,
    coarse_res: float,
    min_pbp: float,
    nfreq: int,
    ntime: int,
    freq_list: str | None,
    time_list: str | None,
    max_pix_per_job: int,
    fc: float,
    eta: float,
    file_prefix: str,
    plot_trec: bool,
    plot_pb: bool,
    plot_tab: bool,
    plot_tsky: bool,
    plot_integrals: bool,
    plot_3d: bool,
    extra_tile_flags: list[str] | None,
    cal_metafits: str,
) -> None:
    setup_logger("mwa_vcs_fluxcal", log_level)

    target_coords = SkyCoord(target, frame="icrs", unit=("hourangle", "deg"))

    if file_prefix is None:
        file_prefix = target_coords.to_string(style="hmsdms", precision=2).replace(" ", "_")

    end_offset = None
    if start_offset is not None and int_time is not None:
        end_offset = start_offset + int_time

    if isinstance(freq_list, str):
        freqs = [float(freq) for freq in freq_list.split(",")]
    else:
        freqs = nfreq

    if isinstance(time_list, str):
        times = [float(freq) for freq in time_list.split(",")]
    else:
        times = ntime

    if isinstance(extra_tile_flags, str):
        extra_tile_flags = extra_tile_flags.split(",")

    results = simulate_sefd(
        metafits=metafits,
        target_coords=target_coords,
        start_time_offset=start_offset,
        end_time_offset=end_offset,
        fine_grid_res=fine_res,
        coarse_grid_res=coarse_res,
        min_pbp=min_pbp,
        freqs=freqs,
        times=times,
        max_pix_per_job=max_pix_per_job,
        fc=fc,
        eta=eta,
        plot_trec=plot_trec,
        plot_pb=plot_pb,
        plot_tab=plot_tab,
        plot_tsky=plot_tsky,
        plot_integrals=plot_integrals,
        plot_3d=plot_3d,
        extra_tile_flags=extra_tile_flags,
        cal_metafits=cal_metafits,
        file_prefix=file_prefix,
    )

    qty_dict_to_toml(results, f"{file_prefix}_fluxcal_results.toml")


if __name__ == "__main__":
    main()
