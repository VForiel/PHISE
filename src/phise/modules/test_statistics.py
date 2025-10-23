"""Module generated docstring."""
import numpy as np
import matplotlib.pyplot as plt
try:
    plt.rcParams['image.origin'] = 'lower'
except Exception:
    pass
from copy import deepcopy as copy
import astropy.units as u
from scipy import stats

def get_vectors(ctx=None, nmc: int=1000, size: int=1000):
    """"get_vectors.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    if ctx is None:
        # import localement pour éviter une circular import lors de l'import
        # du package phise (les modules et classes s'importent mutuellement)
        from phise.classes.context import Context
        ctx = Context.get_VLTI()
        ctx.interferometer.kn.σ = np.zeros(14) * u.m
    if ctx.target.companions == []:
        raise ValueError('No companions in the context. Please add companions to the context before generating vectors.')
    ctx_h1 = copy(ctx)
    ctx_h0 = copy(ctx)
    ctx_h0.target.companions = []
    T0 = np.zeros((3, nmc, size))
    T1 = np.zeros((3, nmc, size))
    fov = ctx.interferometer.fov.to(u.mas).value
    for i in range(nmc):
        print(f'⌛ Generating vectors... {round(i / nmc * 100, 2)}%', end='\r')
        for j in range(size):
            for c in ctx_h1.target.companions:
                c.α = np.random.uniform(0, 2 * np.pi) * u.rad
                c.θ = np.random.uniform(fov / 10, fov) * u.mas
            (_, k_h0, b_h0) = ctx_h0.observe()
            (_, k_h1, b_h1) = ctx_h1.observe()
            k_h0 /= b_h0
            k_h1 /= b_h1
            T0[:, i, j] = k_h0
            T1[:, i, j] = k_h1
    print(f'✅ Vectors generation complete')
    return (np.concatenate(T0), np.concatenate(T1))

def mean(u, v):
    """"mean.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    return np.abs(np.mean(u))

def median(u, v):
    """"median.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    return np.abs(np.median(u))

def argmax(u, v, bins=100):
    """"argmax.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    maxs = np.zeros(u.shape[0])
    (hist, bin_edges) = np.histogram(u, bins=bins)
    bin_edges = (bin_edges[1:] + bin_edges[:-1]) / 2
    return np.abs(bin_edges[np.argmax(hist)])

def argmax50(u, v):
    """"argmax50.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    return argmax(u, v, 50)

def argmax100(u, v):
    """"argmax100.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    return argmax(u, v, 100)

def argmax500(u, v):
    """"argmax500.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    return argmax(u, v, 500)

def kolmogorov_smirnov(u, v):
    """"kolmogorov_smirnov.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    return np.abs(stats.ks_2samp(u, v).statistic)

def cramer_von_mises(u, v):
    """"cramer_von_mises.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    return np.abs(stats.cramervonmises_2samp(u, v).statistic)

def mannwhitneyu(u, v):
    """"mannwhitneyu.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    return np.abs(stats.mannwhitneyu(u, v).statistic)

def wilcoxon_mann_whitney(u, v):
    """"wilcoxon_mann_whitney.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    return np.abs(stats.wilcoxon(u, v).statistic)

def anderson_darling(u, v):
    """"anderson_darling.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    return np.abs(stats.anderson_ksamp([u, v]).statistic)

def brunner_munzel(u, v):
    """"brunner_munzel.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    return np.abs(stats.brunnermunzel(u, v).statistic)

def wasserstein_distance(u, v):
    """"wasserstein_distance.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    return np.abs(stats.wasserstein_distance(u, v))

def flattening(u, v):
    """"flattening.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    med = np.median(u)
    return np.sum(np.abs(u - med))

def shift_and_flattening(u, v):
    """"shift_and_flattening.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    med = np.median(u)
    distances = np.sort(np.abs(u - med))
    x = np.linspace(0, 1, len(u))
    auc = np.trapz(distances + np.abs(med), x)
    return auc

def median_of_abs(u, v):
    """"median_of_abs.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    return np.median(np.abs(u))

def full_sum(u, v):
    """"full_sum.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    return np.sum(np.abs(u))
ALL_TESTS = {'Mean': mean, 'Median': median, 'Kolmogorov-Smirnov': kolmogorov_smirnov, 'Cramer von Mises': cramer_von_mises, 'Flattening': flattening, 'Median of Abs': median_of_abs}