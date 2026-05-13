
import numpy as np
import astropy.units as u
from astropy import constants as const

#==============================================================================
# VLTI Context
#==============================================================================

def get() -> 'Context':
    """Get a default VLTI context for analysis.

    Uses:
        - VLTI with 4 UTs
        - First generation active kernel nuller
        - Vega as target star and a hypothetical 2 mas, 1e-6 contrast companion
    """

    
    import phise

    λ = 1.55 * u.um # Central wavelength

    ctx = phise.Context(
        h = 0 * u.hourangle, # Central hour angle
        Δh = 8 * u.hourangle, # Hour angle range
        Γ = 100 * u.nm, # Input cophasing error (RMS)
        monochromatic=True,
        name="Default Context", # Context name
        interferometer = phise.Interferometer(
            l = -24.6275 * u.deg, # Latitude
            λ = λ, # Central wavelength
            Δλ = 0.1 * u.um, # Bandwidth
            fov = 10 * u.mas, # Field of view
            η = 0.02, # Optical efficiency
            telescopes = phise.telescope.get_VLTI_UTs(),
            name = "VLTI", # Interferometer name
            chip = phise.SuperKN(
                φ = np.zeros(14) * u.nm, # Injected phase shifts
                σ = np.abs(np.random.normal(0, 10, 14)) * u.nm, # Manufacturing OPD errors
                λ0 = λ,
                name = "First Generation Kernel-Nuller", # Kernel nuller name
            ),
            camera = phise.Camera(
                e = 5 * u.min, # Exposure time
                name = "Default Camera", # Camera name
            ),
        ),
        target = phise.Target(
            f = 1e-12 * u.W / u.m**2 / u.nm, # Target flux (Sun-like star at 10 pc)
            δ = -64.71 * u.deg, # Target declination
            name = "Sun-like @ 10pc", # Target name
            companions = [
                phise.Companion(
                    c = 1e-2, # Companion contrast
                    ρ = 4 * u.mas, # Companion angular separation
                    θ = 0 * u.deg, # Companion position angle
                    name = "Hypothetical Companion", # Companion name
                ),
            ],
        ),
    )

    return ctx