import numpy as np
from astropy import units as u

from . import kernel_nuller
from .kernel_nuller import KernelNuller
from .body import Body
from . import signals
from. import telescopes

class Instrument:
    def __init__(
            self,
            λ:u.Quantity,
            r:np.ndarray[u.Quantity],
            l:u.Quantity,
            kn:KernelNuller
        ):
        """Instrument object.

        Parameters
        ----------
        - λ: Wavelength
        - r: Relative telescope positions (Nx2 array)
        - l: Latitude of reference telescope
        - kn: Kernel-Nuller object
        """

        self.λ = λ
        self.r = r
        self.l = l
        self.kn = kn

    def observe(
            self,
            b:Body,
            φ:np.ndarray[u.Quantity],
            δ:u.Quantity,
            h:u.Quantity,
            f:float,
            dt:u.Quantity,
        ) -> np.ndarray[float]:
        """
        Simulate a 4 telescope Kernel-Nuller observation

        Parameters
        ----------
        - b: Body object
        - φ: Array of 14 injected OPD
        - δ: Declination
        - h: Hour angle
        - f: Flux of the star (photons/s)
        - dt: Integration time

        Returns
        -------
        - Array of 3 null outputs electric fields
        """

        # Project telescope positions
        p = telescopes.project_position(r=self.r, h=h, l=self.l, δ=δ)

        # Get input fields
        ψ = signals.get_input_fields(f=f*b.c, θ=b.θ, α=b.α, λ=self.λ, p=p)

        return self.kn.observe(ψ, φ, self.λ, f, dt)