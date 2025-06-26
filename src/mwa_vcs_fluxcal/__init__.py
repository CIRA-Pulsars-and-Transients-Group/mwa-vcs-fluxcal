from importlib.metadata import version

from matplotlib.pyplot import rcParams

from .constants import *
from .gridding import *
from .integral import *
from .logger import *
from .plotting import *
from .profile import *
from .tab import *
from .temperatures import *
from .utils import *

__version__ = version(__name__)

rcParams["font.size"] = 12
rcParams["mathtext.fontset"] = "dejavuserif"
rcParams["font.family"] = "serif"
# rcParams["text.usetex"] = True
# rcParams["font.serif"] = "cm"
