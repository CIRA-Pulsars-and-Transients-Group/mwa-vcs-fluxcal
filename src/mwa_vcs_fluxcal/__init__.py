########################################################
# Licensed under the Academic Free License version 3.0 #
########################################################

from importlib.metadata import version

from matplotlib.pyplot import rcParams

__version__ = version(__name__)

rcParams["figure.dpi"] = 300
rcParams["font.size"] = 12
rcParams["mathtext.fontset"] = "dejavuserif"
rcParams["font.family"] = "serif"

# Uncomment only if LaTeX is installed
# rcParams["text.usetex"] = True
# rcParams["font.serif"] = "cm"

del version, rcParams
