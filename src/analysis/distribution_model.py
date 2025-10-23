"""Ajustement de lois statistiques aux sorties du nuller.

Contient des routines pour générer des échantillons, ajuster des
distributions (Cauchy, Laplace, Johnson SU, ...) et tracer les
résultats.
"""
from copy import deepcopy as copy
from astropy import units as u
import numpy as np
import fitter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from phise import Context

def fit(ctx: Context=None):
    """"fit.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    if ctx is None:
        ctx = Context.get_VLTI()
        ctx.interferometer.kn.σ = np.zeros(14) * u.nm
    else:
        ctx = copy(ctx)
    ctx.Δh = ctx.interferometer.camera.e.to(u.hour).value * u.hourangle
    ctx.target.companions = []
    N = 100000
    data = np.empty((N, 3))
    print('⌛ Generating data...')
    for i in range(N):
        print(f'{(i + 1) / N * 100:.2f}%', end='\r')
        (_, k, b) = ctx.observe()
        data[i] = k / b
    print('✅ Data generation complete.')
    data = data[:, 0]

    def cauchy(x, μ, σ):
        """"cauchy.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        return 1 / (np.pi * σ * (1 + ((x - μ) / σ) ** 2))

    def laplace(x, μ, σ):
        """"laplace.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        return 1 / (2 * σ) * np.exp(-np.abs(x - μ) / σ)

    def johnsonsu(x, μ, σ, γ, δ):
        """"johnsonsu.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        return 1 / (σ * np.sqrt(2 * np.pi)) * 1 / np.sqrt(1 + ((x - μ) / σ) ** 2) * np.exp(-0.5 * (γ + δ * np.sinh((x - μ) / σ)) ** 2)
    (hist, bin_edges) = np.histogram(data, bins=500, density=True)
    x = (bin_edges[:-1] + bin_edges[1:]) / 2
    print('⌛ Fitting distributions...')
    (cauchy_pop, _) = curve_fit(cauchy, x, hist, p0=[np.mean(data), np.std(data)])
    print('✅ Cauchy fit complete.')
    (laplace_pop, _) = curve_fit(laplace, x, hist, p0=[np.mean(data), np.std(data)])
    print('✅ Laplace fit complete.')
    (johnsonsu_pop, _) = curve_fit(johnsonsu, x, hist, p0=[np.mean(data), np.std(data), 0, 1])
    print('✅ Johnson SU fit complete.')
    plt.figure(figsize=(5, 5))
    plt.hist(data, bins=50, density=True, label='Data', log=True)
    plt.plot(x, cauchy(x, *cauchy_pop), 'r-', label='Cauchy Fit')
    plt.plot(x, laplace(x, *laplace_pop), 'g-', label='Laplace Fit')
    plt.plot(x, johnsonsu(x, *johnsonsu_pop), 'b-', label='Johnson SU Fit')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.ylim(bottom=0.1, top=100.0)
    plt.grid()
    plt.show()