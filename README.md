# mwa-vcs-fluxcal
A suite of tools to estimate flux densities for pulsar profiles from MWA VCS data.

# Installation
This package requires Python 3.10+ and PSRCHIVE with Python bindings installed.
All of the required dependencies can be installed from scratch is using `conda`.
For example:

    conda create -n mwa-vcs-fluxcal python=3.10
    conda activate mwa-vcs-fluxcal
    conda install conda-forge::dspsr
    pip install git+https://github.com/CIRA-Pulsars-and-Transients-Group/mwa-vcs-fluxcal.git