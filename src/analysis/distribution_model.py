# External libs
from copy import deepcopy as copy
from astropy import units as u
import numpy as np
import fitter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

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
    N = 10000
    data = np.empty((N, 3))
    for i in range(N):
        _, k, b = ctx.observe()
        data[i] = k / b

    data = data[:, 0]  # Use only the first kernel

    # f = fitter.Fitter(data, distributions=fitter.get_distributions())
    # f.fit()
    # f.summary()

    def cauchy(x, μ, σ):
        return 1 / (np.pi * σ * (1 + ((x - μ) / σ) ** 2))
    
    def laplace(x, μ, σ):
        return (1 / (2 * σ)) * np.exp(-np.abs(x - μ) / σ)
    
    def johnsonsu(x, μ, σ, γ, δ):
        return (1 / (σ * np.sqrt(2 * np.pi))) \
            * 1 / np.sqrt(1 + ((x - μ) / σ) ** 2) \
            * np.exp(-0.5 * (γ + δ*np.sinh((x - μ) / σ))**2)

    hist, bin_edges = np.histogram(data, bins=50, density=True)
    x = (bin_edges[:-1] + bin_edges[1:]) / 2

    cauchy_pop, _ = curve_fit(cauchy, x, hist, p0=[np.mean(data), np.std(data)])
    laplace_pop, _ = curve_fit(laplace, x, hist, p0=[np.mean(data), np.std(data)])
    johnsonsu_pop, _ = curve_fit(johnsonsu, x, hist, p0=[np.mean(data), np.std(data), 0, 1])

    plt.figure(figsize=(5, 5))
    plt.hist(data, bins=50, density=True, label='Data', log=True)
    plt.plot(x, cauchy(x, *cauchy_pop), 'r-', label='Cauchy Fit')
    plt.plot(x, laplace(x, *laplace_pop), 'g-', label='Laplace Fit')
    plt.plot(x, johnsonsu(x, *johnsonsu_pop), 'b-', label='Johnson SU Fit')
    # plt.title('Distribution Fitting')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.ylim(bottom=1e-1, top=1e2)
    plt.grid()
    plt.show()