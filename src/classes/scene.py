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
            input_ce_rms: u.Quantity,
            sources: list[Source] = None,
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
        self.input_ce_rms = input_ce_rms
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
        return self.instrument.observe(sources=self.sources, δ=self.δ, h=self.h, f=self.f, Δt=self.Δt, input_ce_rms=self.input_ce_rms)
    
    def instant_serie_observation(self, N:int):
        """
        Simulate the observation of the scene for a given number of times

        Parameters
        ----------
        - N: Number of observations

        Returns
        -------
        - dict[str, np.ndarray]
            - 'darks': Dark outputs for each observation (Nx6)
            - 'kernels': Kernel outputs for each observation (Nx3)
            - 'brights': Bright outputs for each observation (N)
        """
        
        darks = np.empty((N, 6))
        kernels = np.empty((N, 3))
        brights = np.empty(N)

        for i in range(N):
            darks[i], kernels[i], brights[i] = self.observe()

        return {"darks": darks, "kernels": kernels, "brights": brights}
    
    def time_serie_observation(self, nights:int):
        """
        Simulate the observation of the scene over all the observation time and for several nights

        Parameters
        ----------
        - nights: Number of nights

        Returns
        -------
        - dict[str, np.ndarray]
            - 'darks': Dark outputs for each observation (Nx6)
            - 'kernels': Kernel outputs for each observation (Nx3)
            - 'brights': Bright outputs for each observation (N)
            with N = nights * Δh/Δt -> Number of observations
        - np.ndarray: Hourangles of the observations
        """

        H = self.h

        time_resolution = int(self.Δh.to(u.hourangle).value / self.Δt.to(u.hour).value)
        hs = np.linspace(self.h-self.Δh/2, self.h+self.Δh/2, time_resolution)

        darks = np.empty((nights, time_resolution, 6))
        kernels = np.empty((nights, time_resolution, 3))
        brights = np.empty((nights, time_resolution))

        for n in range(nights):
            for i, h in enumerate(hs):
                self.h = h
                darks[n,i], kernels[n,i], brights[n,i] = self.observe()

        self.h = H

        return {"darks": darks, "kernels": kernels, "brights": brights}, hs

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