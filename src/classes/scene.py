import numpy as np
import astropy.units as u

from .instrument import Instrument
from .source import Source
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
            sources: list[Source] = None
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
        - sources: List of Source objects
        """

        self.instrument = instrument
        self.δ = δ
        self.h = h
        self.Δh = Δh
        self.f = f
        self.Δt = Δt
        self.sources = sources if sources else []

        self.p = telescopes.project_position(r=self.instrument.r, h=self.h, l=self.instrument.l, δ=self.δ)

    # Observation -------------------------------------------------------------

    def observe(self):
        """
        Simulate the observation of the scene

        Returns
        -------
        - np.ndarray: Dark outputs (6 float values)
        - np.ndarray: Kernel outputs (3 float values)
        - float: Bright output
        """
        return self.instrument.observe(sources=self.sources, δ=self.δ, h=self.h, f=self.f, Δt=self.Δt)

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
        return self.instrument.plot_transmission_maps(N=N, h=self.h, δ=self.δ, sources=self.sources)
    
    def iplot_transmission_maps(self, N:int):
        return self.instrument.iplot_transmission_maps(N=N, δ=self.δ, h=self.h, Δh=self.Δh, sources=self.sources)