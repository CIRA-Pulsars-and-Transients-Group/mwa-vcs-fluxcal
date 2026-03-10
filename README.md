# mwa-vcs-fluxcal
A Python implementation of the MWA tied-array sensitivity simulation method
developed and described by
[Meyers et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017ApJ...851...20M).

# Installation
The package can be installed as follows:
```bash
pip install git+https://github.com/CIRA-Pulsars-and-Transients-Group/mwa-vcs-fluxcal.git
```
This will install the `mwa_vcs_fluxcal` library and the `fluxcal` command line
program.

# Usage

## Command line interface
The `fluxcal` command can be used to simulate the system temperature, tied-array
gain, and system equivalent flux density (SEFD) for an MWA observation towards
a particular astronomical target.

For example, say you have a beamformed observation of PSR J0034-0534 in the
VCS observation 1255444104, starting 600 seconds after the scheduled start time
and ending 1800 seconds later. The simulation can be run with the following
command line options:
```bash
fluxcal -m 1255444104.metafits -t '00:34:21.83 -05:34:36.72' -s 600 -i 1800
```
This will prduce a file `00h34m21.83s_-05d34m36.72s_fluxcal_results.toml` that
contains the simulation results.

There are lots of other options detailed in the help menu:
```bash
fluxcal -h
```

```txt
Usage: fluxcal [OPTIONS]

Options:
  -h, --help                      Show this message and exit.
  -V, --version                   Show the version and exit.
  -L [debug|info|warning|error|critical]
                                  The logger verbosity level.  [default: INFO]
  -m, --metafits PATH             An MWA metafits file.
  -t, --target TEXT               The target's RA/Dec in hour/deg units in any
                                  format accepted by SkyCoord.
  -s, --start_offset FLOAT        The difference (in seconds) between the
                                  scheduled observation start time (the obs
                                  ID) and the start time of the data being
                                  calibrated. Will override the start time
                                  from the metafits, if provided.
  -i, --int_time FLOAT            The integration time (in seconds) of the
                                  data being calibrated. Will override the
                                  integration time from the metafits, if
                                  provided.
  --fine_res FLOAT                The resolution of the integral, in arcmin.
                                  [default: 2]
  --coarse_res FLOAT              The resolution of the primary beam map, in
                                  arcmin.  [default: 30]
  --min_pbp FLOAT RANGE           Only integrate above this primary beam
                                  power.  [default: 0.001; 0.0<=x<=1.0]
  --nfreq INTEGER                 The number of frequency steps to evaluate.
                                  [default: 1]
  --ntime INTEGER                 The number of time steps to evaluate.
                                  [default: 1]
  --max_pix_per_job INTEGER       The maximum number of sky area pixels to
                                  compute per job.  [default: 100000]
  --fc FLOAT RANGE                The beamforming coherency factor.  [default:
                                  1.43; 1.0<=x<=10.0]
  --eta FLOAT RANGE               The radiation efficiency of the array.
                                  [default: 0.98; 0.0<=x<=1.0]
  -f, --file_prefix TEXT          The prefix of the output file names.
  --plot_trec                     Plot the receiver temperature.
  --plot_pb                       Plot the primary beam in Alt/Az.
  --plot_tab                      Plot the tied-array beam in Alt/Az.
  --plot_tsky                     Plot sky temperature in Alt/Az.
  --plot_integrals                Plot the integral quantities in Alt/Az.
  --plot_3d                       Plot the results in 3D (time,freq,data).
```

## Python interface
The simulations can be run within Python via the
`mwa_vcs_fluxcal.interface.simulate_sefd` function. The options are very similar
to what is available in the command line interface. The function will return a
dictionary containing the simulation results.

## Notes on pixel sizes
Computing the system temperature requires integrating over the sky with a
resolution sufficient to resolve the tied-array beam. Since the tied-array
beam power depends on the primary beam power, the integral only needs to be
performed where the primary beam power is significant. With this in mind, we
divide the sky into 'coarse pixels' which we use to determine the primary beam
cutout. For the coarse pixels with a primary beam power above a certain level,
the integrals are performed, and other pixels are ignored. The integration
itself uses a finer resolution, i.e. 'fine pixels'. The user can specify the
angular resolution using the options `--coarse_res` and `--fine_res`, where the
coarse resolution must be cleanly divisible by the fine resolution (in arcmin).
As a general rule, the fine resolution should be smaller than around 1/5 of the
width of the tied-array beam. Also note that the sky plots will be shown with
coarse pixels, where the value for each coarse pixel is the mean of the fine
pixels computed within it. The `--max_pix_per_job` option specifies how many
fine pixels will be computed in memory at a given time. It can be tweaked
depending on the available CPU and memory resources.

## Notes on frequency and time steps
The system temperature, gain, and thus the SEFD are all functions of frequency
and time (as the target moves across the beam). Therefore, depending on the
bandwidth and integration time, these quantities should be computed in multiple
steps. The `--nfreq` and `--ntime` options can be used to select the number of
uniformly spaced simulations across the bandwidth and integration time. If four
or more steps are provided for both dimensions, then the `--plot_3d` option can
be used to generate 3D visualisations using Matplotlib.

# Credit
If you use `mwa-vcs-fluxcal` in your work, please give credit by citing
[Meyers et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017ApJ...851...20M)
and [Lee et al. (2025)](https://ui.adsabs.harvard.edu/abs/2025PASA...42..117L).