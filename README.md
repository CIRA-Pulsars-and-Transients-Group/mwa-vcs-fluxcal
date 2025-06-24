# mwa-vcs-fluxcal
A suite of tools to estimate flux densities for MWA-VCS data.

# Installation
This package requires Python 3.10+ and PSRCHIVE with Python bindings installed.
All of the required dependencies can be installed from scratch is using `conda`.
For example:

```bash
conda create -n mwa-vcs-fluxcal python=3.10
conda activate mwa-vcs-fluxcal
conda install conda-forge::dspsr
pip install git+https://github.com/CIRA-Pulsars-and-Transients-Group/mwa-vcs-fluxcal.git
```

# Usage
This package provides a library of tools to perform flux density calibration of
MWA-VCS data. Each function is documented via docstrings in the code. However,
for most purposes it should be sufficient to use the included command line
utility, `fluxcal`. Essentialy, `fluxcal` follows the steps outlined in
[Meyers et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017ApJ...851...20M):
it computes the SEFD, measures the pulse S/N, then plugs the numbers into the
radiometer equation. To do this, the user can provide the MWA metafits file
and/or a PSRCHIVE archive. If a metafits file is not provided but an archive
is, then `fluxcal` can still be used to plot the pulse profile and check whether
the off-pulse window has been selected correctly. If the archive file is not
provided but the metafits file is, then the user can specify the target
coordinates so that the SEFD can be computed. As such, there are three ways one
might want to invoke `fluxcal`:

1) Measuring the flux density of a pulsar detection. E.g.

```bash
fluxcal -m 1255444104.metafits -a J0034-0534.ar
```

2) Measuring the SEFD given the target coordinates (`-t`), the start time
(`-s`), and the integration time (`-i`). E.g.

```bash
fluxcal -m 1255444104.metafits -t '00:34:21.83 -05:34:36.72' -s 600 -i 1800
```

3) Checking the off-pulse window selection. E.g.

```bash
fluxcal -a J0034-0534.ar --plot_profile
```

Besides these basic inputs, there are several options detailed in the help menu:

```bash
fluxcal --help
```

```txt
Usage: fluxcal [OPTIONS]

Options:
  -L [DEBUG|INFO|ERROR|CRITICAL]  The logger verbosity level.  [default: INFO]
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
  -a, --archive PATH              An archive file to use to compute the pulse
                                  profile and get the start/end times of the
                                  data. Will override the metafits or user-
                                  provided start/end times, if provided.
  -n, --noise_archive PATH        An archive file to use to compute the
                                  offpulse noise.
  -b, --bscrunch INTEGER          Bscrunch to this number of phase bins.
  -w, --window_size INTEGER       Window size to use to find the offpulse.
  --fine_res FLOAT                The resolution of the integral, in arcmin.
                                  [default: 2]
  --coarse_res FLOAT              The resolution of the primary beam map, in
                                  arcmin.  [default: 30]
  --min_pbp FLOAT                 Only integrate above this primary beam
                                  power.  [default: 0.001]
  --nfreq INTEGER                 The number of frequency steps to evaluate.
                                  [default: 1]
  --ntime INTEGER                 The number of time steps to evaluate.
                                  [default: 1]
  --max_pix_per_job INTEGER       The maximum number of sky area pixels to
                                  compute per job.  [default: 100000]
  --bw_flagged FLOAT RANGE        The fraction of the bandwidth flagged.
                                  [default: 0.0; 0.0<=x<=1.0]
  --time_flagged FLOAT RANGE      The fraction of the integration time
                                  flagged.  [default: 0.0; 0.0<=x<=1.0]
  --plot_profile                  Plot the pulse profile.
  --plot_trec                     Plot the receiver temperature.
  --plot_pb                       Plot the primary beam.
  --plot_images                   Plot visualisations of the integral
                                  quantities.
  --plot_3d                       Plot the results in 3D (time,freq,data).
  --help                          Show this message and exit.
```

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
width of the tied-array beam. Also note that the figures made with `--plot_pb`
and `--plot_images` will be shown with coarse pixels, where the value for each
coarse pixel is the mean of the fine pixels computed within it. The
`--max_pix_per_job` option specifies how many fine pixels will be computed
in memory at a given time. It can be tweaked depending on the available CPU and
memory resources.

## Notes on frequency and time steps
The system temperature, gain, and thus the SEFD are all functions of frequency
and time (as the target moves across the beam). Therefore, depending on the
bandwidth and integration time, these quantities should be computed in multiple
steps. The `--nfreq` and `--ntime` options can be used to select the number of
uniformly spaced evaluations across the bandwidth and integration time. If four
or more steps are provided for both dimensions, then the `--plot_3d` option can
be used to generate 3d visualisations using Matplotlib.

## Notes on data flagging
The effective bandwidth and integration time will depend on the number of
flagged frequency channels and time steps. Unfortunately, getting this
information from folded data is not always reliable, especially if the data was
downsampled after flagging at a higher resolution. Therefore, the user may
provide the fraction of the bandwidth and integration time flagged in the data
using the `--bw_flagged` and `--time_flagged` options. For example, most SMART
observations flag 4 out of 32 fine channels per coarse channel during
calibration. Therefore the user would specify `--bw_flagged 0.125`. This number
is used to modify the bandwidth used in the radiometer equation.