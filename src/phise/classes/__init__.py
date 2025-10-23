"""Aggregate and re-export main classes from `phise.classes`.

This module re-exports top-level classes used in the simulation
(Companion, Target, Telescope, KernelNuller, Interferometer, Context, Camera)
to allow simplified imports:

>>> from phise.classes import Camera, Telescope
"""
from . import companion
from .companion import Companion
from . import target
from .target import Target
from . import telescope
from .telescope import Telescope
from . import kernel_nuller
from .kernel_nuller import KernelNuller
from . import interferometer
from .interferometer import Interferometer
from . import context
from .context import Context
from . import camera
from .camera import Camera