import numpy as np
import astropy.units as u
from astropy import constants as const

#==============================================================================
# Photoniccs Lab Context
#==============================================================================

def get() -> 'Context':
    """Get a context representing the photonics lab for analysis.

    Uses:
        - 4 telescopes of LIFE
        - First generation active kernel nuller
        - Vega as target star and a hypothetical 2 mas, 1e-6 contrast companion
    """

    import phise

    # LIFE range: 4 to 18 μm
    λ = 1.55 * u.um # Central wavelength

    ctx = phise.Context(
        h = 0 * u.hourangle, # Central hour angle
        Δh = 24 * u.hourangle, # Hour angle range
        Γ = 1 * u.nm, # Input cophasing error (RMS)
        name="Default Context", # Context name
        monochromatic=True,
        interferometer = phise.Interferometer(
            l = -90 * u.deg, # Latitude
            λ = λ, # Central wavelength
            Δλ = 0.1 * u.um, # Bandwidth
            fov = 10 * u.mas, # Field of view
            η = 0.02, # Optical efficiency
            telescopes = [
                # 4 horizontal holes of 650 μm diameter each and separated by 1.3 mm center-to-center, representing the 4 inputs of the PHOB testbed
                phise.Telescope(
                    a = np.pi * (650*u.um/2)**2,
                    r = r,
                    name = f"Input {i+1}",
                )
            for i, r in enumerate(1.3*u.mm/2 * np.array([(-3, 0), (-1, 0), (1, 0), (3, 0)]))],
            name = "PHOB", # Interferometer name
            chip = phise.N4x4_T8(
                φ = np.zeros(4) * λ, # Injected phase shifts
                σ = np.abs(np.random.normal(0, 1, 4)) * λ, # Manufacturing OPD errors
                λ0 = λ,
                name = "N4x4-T8", # Kernel nuller name
            ),
            camera = phise.Camera(
                e = 1/600 * u.s, # Exposure time
                resolution=11,
                qe = 0.73, # Quantum efficiency
                gain = 2.0, # Gain (e-/ADU)
                ron = 37, # Read-out noise
                dc = 755_000, # Dark current (in e-/px/s)
                fwc = 1_400_000, # Full well capacity
                max_adu = 16383, # Maximum ADU
                name = "Default Camera", # Camera name
            ),
        ),
        target=phise.Target(
            f = 1e-06 * u.W / u.m**2 / u.nm, # Laser flux (roughly estimated)
            δ = -90 * u.deg, # Target declination
            name = "Tunable Laser", # Target name
            companions = [],
        ),
    )

    return ctx
