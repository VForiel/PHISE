from astropy import units as u
import numpy as np

from .. import Context, Interferometer, Target, Companion, KernelNuller, Camera
from .. import telescope

#==============================================================================
#  Default context for the analysis.
#  This context uses:
#      - The VLTI with 4 UTs.
#      - The first generation of active kernel nuller.
#      - Vega as target star and an hypothetical companion at 2 mas with a contrast of 1e-6.
#==============================================================================

ctx = Context(
    interferometer = Interferometer(
        l = -24.6275 * u.deg, # Latitude
        λ = 1.65 * u.um, # Central wavelength
        Δλ = 0.1 * u.m, # Bandwidth
        fov = 10 * u.mas, # Field of view
        telescopes = telescope.get_VLTI_UTs(),
        kn = KernelNuller(
            φ = np.zeros(14) * u.um, # Injected phase shifts
            σ = np.abs(np.random.normal(0, 1, 14)) * u.um, # Manufacturing OPD errors
        ),
        camera = Camera(
            e = 1 * u.s, # Exposure time
        ),
    ),
    target=Target(
        f = (1050 * u.Jy).to(u.W / u.m**2 / u.nm), # Target flux
        δ = -64.71 * u.deg, # Target declination
        companions = [
            Companion(
                c = 1e-6, # Companion contrast
                θ = 2 * u.mas, # Companion angular separation
                α = 45 * u.deg, # Companion position angle
            ),
        ],
    ),
    h = 0 * u.hourangle, # Central hour angle
    Δh = 8 * u.hourangle, # Hour angle range
    Γ = 100 * u.nm, # Input cophasing error (RMS)
)