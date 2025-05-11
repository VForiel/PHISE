from astropy import units as u
import numpy as np

from .. import Context, Interferometer, Target, Companion, KernelNuller, Camera
from .. import telescope

ctx = Context(
    interferometer = Interferometer(
        l = -24.6275 * u.deg,
        λ = 1.65 * u.um,
        Δλ = 0.1 * u.m,
        fov = 10 * u.mas,
        telescopes = telescope.get_VLTI_UTs(),
        kn = KernelNuller(
            φ = np.zeros(14) * u.um,
            σ = np.abs(np.random.normal(0, 1, 14)) * u.um,
        ),
    ),
    target=Target(
        m = 0 * u.mag,
        δ = -64.71 * u.deg,
        companions = [
            Companion(
                c = 1e-6,
                θ = 2 * u.mas,
                α = 45 * u.deg,
            ),
        ],
    ),
    h = 0 * u.hourangle,
    Δh = 8 * u.hourangle,
    e = 1 * u.s, # Unused
    Γ = 100 * u.nm, # Unused
)