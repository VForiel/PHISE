

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
from copy import deepcopy as copy
import astropy.units as u
from scipy import stats

from src.classes import Context

def get_vectors(ctx:Context=None, nmc:int=1000, size:int=1000):

    if ctx is None:
        ctx = Context.get_VLTI()
        ctx.interferometer.kn.σ = np.zeros(14) * u.m

    if ctx.target.companions == []:
        raise ValueError("No companions in the context. Please add companions to the context before generating vectors.")
    
    ctx_h1 = copy(ctx)
    ctx_h0 = copy(ctx)
    ctx_h0.target.companions = []

    T0 = np.zeros((3, nmc, size))
    T1 = np.zeros((3, nmc, size))

    fov = ctx.interferometer.fov.to(u.mas).value

    for i in range(nmc):
        print(f"⌛ Generating vectors... {round(i/nmc * 100,2)}%", end="\r")

        for j in range(size):

            for c in ctx_h1.target.companions:
                c.α = np.random.uniform(0, 2 * np.pi) * u.rad
                c.θ = np.random.uniform(fov/10, fov) * u.mas

            _, k_h0, b_h0 = ctx_h0.observe()
            _, k_h1, b_h1 = ctx_h1.observe()

            k_h0 /= b_h0
            k_h1 /= b_h1

            T0[:, i, j] = k_h0
            T1[:, i, j] = k_h1
    
    print(f"✅ Vectors generation complete")
    return np.concatenate(T0), np.concatenate(T1)

#==============================================================================
# Test statistics
#==============================================================================

# Mean ------------------------------------------------------------------------

def mean(u, v):
    return np.abs(np.mean(u))

# Median ----------------------------------------------------------------------

def median(u, v):
    return np.abs(np.median(u))

# Argmax ----------------------------------------------------------------------

def argmax(u, v, bins=100):
    maxs = np.zeros(u.shape[0])
    hist, bin_edges = np.histogram(u, bins=bins)
    bin_edges = (bin_edges[1:] + bin_edges[:-1]) / 2
    return np.abs(bin_edges[np.argmax(hist)])

def argmax50(u, v):
    return argmax(u, v, 50)

def argmax100(u, v):
    return argmax(u, v, 100)

def argmax500(u, v):
    return argmax(u, v, 500)

# Kolmogorov-Smirnov ----------------------------------------------------------

def kolmogorov_smirnov(u, v):
    return np.abs(stats.ks_2samp(u, v).statistic)

# Cramer-von Mises ------------------------------------------------------------

def cramer_von_mises(u, v):
    return np.abs(stats.cramervonmises_2samp(u, v).statistic)

# Mann-Whitney U --------------------------------------------------------------

def mannwhitneyu(u, v):
    return np.abs(stats.mannwhitneyu(u, v).statistic)

# Wilcoxon --------------------------------------------------------------------

def wilcoxon_mann_whitney(u, v):
    return np.abs(stats.wilcoxon(u, v).statistic)

# Anderson Darling ------------------------------------------------------------

def anderson_darling(u, v):
    return np.abs(stats.anderson_ksamp([u, v]).statistic)

# Brunner-Munzel --------------------------------------------------------------

def brunner_munzel(u, v):
    return np.abs(stats.brunnermunzel(u, v).statistic)

# Wasserstein distance --------------------------------------------------------

def wasserstein_distance(u, v):
    return np.abs(stats.wasserstein_distance(u, v))

# Distance to median ----------------------------------------------------------

# def flattening(u, v):
#     med = np.median(u)
#     distances = np.sort(np.abs(u - med))
#     x = np.linspace(0, 1, len(u))
#     auc = np.trapz(distances, x)
#     return auc

def flattening(u, v):
    med = np.median(u)
    return np.sum(np.abs(u - med))

def shift_and_flattening(u, v):
    med = np.median(u)
    distances = np.sort(np.abs(u - med))
    x = np.linspace(0, 1, len(u))
    auc = np.trapz(distances  + np.abs(med), x)
    return auc

def median_of_abs(u, v):
    return np.median(np.abs(u))

def full_sum(u, v):
    return np.sum(np.abs(u))
    

#==============================================================================
# All tests
#==============================================================================

ALL_TESTS = {
    'Mean': mean,
    'Median': median,
    # 'Central bin 50': argmax50,
    # 'Central bin 100': argmax100,
    # 'Central bin 500': argmax500,
    'Kolmogorov-Smirnov': kolmogorov_smirnov,
    # 'Cramer von Mises': cramer_von_mises,
    # 'Mann-Whitney U': mannwhitneyu,
    # 'Wilcoxon': wilcoxon_mann_whitney,
    # 'Anderson Darling': anderson_darling,
    # 'Brunner-Munzel': brunner_munzel,
    # 'Wasserstein distance': wasserstein_distance,
    'Flattening': flattening,
    'Median of Abs': median_of_abs,
}
