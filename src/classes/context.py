import numpy as np
import astropy.units as u
from copy import deepcopy as copy

from .interferometer import Interferometer
from .source import Source
from ..modules import telescopes

class Context:
    def __init__(
            self,
            instrument:Interferometer,
            δ:u.Quantity,
            h:u.Quantity,
            Δh: u.Quantity,
            f:u.Quantity,
            Δt: u.Quantity,
            input_ce_rms: u.Quantity,
            sources: list[Source] = None,
            name:str = "Unnamed",
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
        - name: Name of the scene
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
        self.name = name

    def copy(self,
            instrument:Interferometer = None,
            δ:u.Quantity = None,    
            h:u.Quantity = None,
            Δh: u.Quantity = None,
            f:u.Quantity = None,
            Δt: u.Quantity = None,
            input_ce_rms: u.Quantity = None,
            sources: list[Source] = None,
            **kwargs,
        ) -> "Context":
        """
        Return a copy of the Scene object with some parameters changed.

        Parameters
        ----------
        - instrument: Instrument object
        - δ: Declination of the star
        - h: Central hourangle of the observation
        - Δh: Hourangle range of the observation
        - f: Star photon flux
        - Δt:Exposition time
        - sources: List of Source objects
        - **kwargs: Additional parameters to edit the instrument, the kernel-nuller or the sources
        """

        if isinstance(sources, Source):
            sources = [sources] # Allow to pass a single source

        return Context(
            instrument = instrument.copy(**kwargs) if instrument is not None else self.instrument.copy(**kwargs),
            δ = copy(δ) if δ is not None else copy(self.δ),
            h = copy(h) if h is not None else copy(self.h),
            Δh = copy(Δh) if Δh is not None else copy(self.Δh),
            f = copy(f) if f is not None else copy(self.f),
            Δt = copy(Δt) if Δt is not None else copy(self.Δt),
            input_ce_rms = copy(input_ce_rms) if input_ce_rms is not None else copy(self.input_ce_rms),
            sources = copy(sources) if sources is not None else [s.copy(**kwargs) for s in self.sources],
        )
    
    @property
    def kn(self):
        return self.instrument.kn
    
    @kn.setter
    def kn(self, kn):
        self.instrument.kn = kn

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
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        sources = "\n | - ".join([str(s) for s in self.sources])
        return f'Scene: "{self.name}" \n' + \
            f' | δ = {self.δ}, h = {self.h}, Δh = {self.Δh}, f = {self.f}, Δt = {self.Δt} input_ce_rms = {self.input_ce_rms}' + "\n" \
            ' | ' + '\n | '.join(str(self.instrument).split('\n')) + \
            '\n | Sources : \n' + f' | - {sources}'