import numpy as np
import astropy.units as u

from .instrument import Instrument
from .body import Body
from ..modules import telescopes

class Scene:
    def __init__(
            self,
            instrument:Instrument,
            δ:u.Quantity,
            h:u.Quantity,
            Δh: u.Quantity,
            f:u.Quantity,
            Δt: u.Quantity,
            companions: list[Body] = None
        ):
        """
        Parameters
        ----------
        - instrument: Instrument object
        - δ: Declination of the star
        - h: Central hourangle of the observation
        - Δh: Hourangle range of the observation
        - f: Star photon flux
        - Δt:Exposition time
        - companions: List of Body objects
        """

        self.instrument = instrument
        self.δ = δ
        self.h = h
        self.Δh = Δh
        self.f = f
        self.Δt = Δt
        self.companions = companions if companions else []

        self.p = telescopes.project_position(r=self.instrument.r, h=self.h, l=self.instrument.l, δ=self.δ)

    # Observation -------------------------------------------------------------

    def observe(self):
        """
        Simulate the observation of the scene
        """
        self.instrument.observe(companions=self.companions, δ=self.δ, h=self.h, f=self.f, Δt=self.Δt)

    def get_trasmission_map(self, N:int) -> np.ndarray[float]:
        """
        Generate all the kernel-nuller transmission maps for a given resolution

        Parameters
        ----------
        - N: Resolution of the map

        Returns
        -------
        - Null outputs map (3 x resolution x resolution)
        - Dark outputs map (6 x resolution x resolution)
        - Kernel outputs map (3 x resolution x resolution)
        """
        return self.instrument.get_transmission_maps(N=N, h=self.h, δ=self.δ)
    
    def plot_transmission_maps(self, N:int):
        return self.instrument.plot_transmission_maps(N=N, h=self.h, δ=self.δ, companions=self.companions)
    
    def iplot_transmission_maps(self, N:int):
        return self.instrument.iplot_transmission_maps(N=N, δ=self.δ, h=self.h, Δh=self.Δh, companions=self.companions)