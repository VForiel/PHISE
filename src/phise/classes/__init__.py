"""Module d'agrégation des classes principales du package `phise.classes`.

Ce module ré-exporte les classes de plus haut niveau utilisées dans la
simulation (Companion, Target, Telescope, KernelNuller, Interferometer,
Context, Camera) afin de permettre des importations simplifiées :

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