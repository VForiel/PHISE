# External libs
from copy import deepcopy as copy
from astropy import units as u
import numpy as np
import fitter

# Internal libs
from src import Context
from . import contexts

def fit(ctx:Context=None):

    if ctx is None:
        ctx = contexts.get_VLTI()
        ctx.interferometer.kn.σ = np.zeros(14) * u.nm
    else:
        ctx = copy(ctx)

    # Instant serie -> Observation range = Exposure time
    ctx.Δh = ctx.interferometer.camera.e.to(u.hour).value * u.hourangle
    # No companions
    ctx.target.companions = []

    # Generate data
    N = 1000
    data = np.empty((N, 3))
    for i in range(N):
        _, k, b = ctx.observe()
        data[i] = k / b

    data = data[:, 0]  # Use only the first kernel

    f = fitter.Fitter(data, distributions=fitter.get_distributions())
    f.fit()
    f.summary()