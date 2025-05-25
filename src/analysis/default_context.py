from astropy import units as u
import numpy as np
from astropy import constants as const

from .. import Context, Interferometer, Target, Companion, KernelNuller, Camera
from .. import telescope

def get() -> Context:
    """
    Get the default context for the analysis.
    This context uses:
        - The VLTI with 4 UTs.
        - The first generation of active kernel nuller.
        - Vega as target star and an hypothetical companion at 2 mas with a contrast of 1e-6.
    """

    λ = 1.65 * u.um # Central wavelength

    ctx = Context(
        interferometer = Interferometer(
            l = -24.6275 * u.deg, # Latitude
            λ = λ, # Central wavelength
            Δλ = 0.1 * u.m, # Bandwidth
            fov = 10 * u.mas, # Field of view
            telescopes = telescope.get_VLTI_UTs(),
            name = "VLTI", # Interferometer name
            kn = KernelNuller(
                φ = np.zeros(14) * u.um, # Injected phase shifts
                σ = np.abs(np.random.normal(0, 1, 14)) * u.um, # Manufacturing OPD errors
                name = "First Generation Kernel-Nuller", # Kernel nuller name
            ),
            camera = Camera(
                e = 5 * u.min, # Exposure time
                name = "Default Camera", # Camera name
            ),
        ),
        target=Target(
            f = (1050 * u.Jy * 2 * np.pi * const.c / λ**2).to(u.W / u.m**2 / u.nm), # Target flux
            δ = -64.71 * u.deg, # Target declination
            name = "Vega", # Target name
            companions = [
                Companion(
                    c = 1e-6, # Companion contrast
                    θ = 2 * u.mas, # Companion angular separation
                    α = 45 * u.deg, # Companion position angle
                    name = "Hypothetical Companion", # Companion name
                ),
            ],
        ),
        h = 0 * u.hourangle, # Central hour angle
        Δh = 8 * u.hourangle, # Hour angle range
        Γ = 100 * u.nm, # Input cophasing error (RMS)
        name="Default Context", # Context name
    )

    return ctx