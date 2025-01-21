import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from copy import deepcopy as copy

from src.classes.scene import Scene

def get_vectors(nmc:int, size:int, scene_h1:Scene):

    T0 = np.zeros((3, nmc, size))
    T1 = np.zeros((3, nmc, size))

    scene_h0 = copy(scene_h1)
    scene_h0.sources = [scene_h1.sources[0]] # Keep only the star

    TREF = scene_h0.instant_serie_observation(size)['kernels'].T

    for i in range(nmc):
        print(f"Generating... {round(i/nmc * 100,2)}%", end="\r")
        dists_h0 = scene_h0.instant_serie_observation(size)
        dists_h1 = scene_h1.instant_serie_observation(size)
        for k in range(3):
            T0[k,i] = dists_h0['kernels'][:,k]
            T1[k,i] = dists_h1['kernels'][:,k]
    print(f"Generation complete ✅")
    return T0, T1, TREF

# Mean ------------------------------------------------------------------------

def mean(data, ref):
    return np.abs(np.mean(data, axis=1))

# Median ----------------------------------------------------------------------

def median(data, ref):
    return np.abs(np.median(data, axis=1))

# Argmax ----------------------------------------------------------------------

def argmax(data, ref, bins=100):
    maxs = np.zeros(data.shape[0])
    for i, dist in enumerate(data):
        hist = np.histogram(dist, bins=bins)
        maxs[i] = hist[1][np.argmax(hist[0])]
    return np.abs(maxs)

def argmax50(data, ref):
    return argmax(data, ref, 50)

def argmax100(data, ref):
    return argmax(data, ref, 100)

def argmax500(data, ref):
    return argmax(data, ref, 500)

# Kolmogorov-Smirnov ----------------------------------------------------------

def kolmogorov_smirnov(data, ref):

    distances = np.zeros(data.shape[0])

    for d, dist in enumerate(data):

        dist = np.sort(dist)
        ref = np.sort(ref)

        v = np.min(np.concatenate([dist, ref]))

        i_ref = 0
        i_dist = 0

        count_ref = 0
        count_dist = 0

        while i_ref < len(ref) and i_dist < len(dist):

            if ref[i_ref] < dist[i_dist]:
                count_ref += 1
                i_ref += 1
            else:
                count_dist += 1
                i_dist += 1
            
            distances[d] = max(distances[d], np.abs(count_dist - count_ref) )#/ len(reference))

    return distances

# Cramer-von Mises ------------------------------------------------------------

def cramer_von_mises(data, ref):

    distances = np.zeros(data.shape[0])

    for d, dist in enumerate(data):

        dist = np.sort(dist)
        ref = np.sort(ref)

        v = np.min(np.concatenate([dist, ref]))

        i_ref = 0
        i_dist = 0

        count_ref = 0
        count_dist = 0

        while i_ref < len(ref) and i_dist < len(dist):

            if ref[i_ref] < dist[i_dist]:
                count_ref += 1
                i_ref += 1
            else:
                count_dist += 1
                i_dist += 1
            
            distances[d] += np.abs(count_dist - count_ref) ** 2

    return distances

# Wilcoxon-Mann-Whitney -------------------------------------------------------

def wilcoxon_mann_whitney(data, ref):

    res = np.empty(len(data))

    for d, dist in enumerate(data):

        sorted_comb = np.unique(np.sort(np.concatenate([dist, ref])))

        r1 = np.sum(np.searchsorted(sorted_comb, dist) + 1)
        r2 = np.sum(np.searchsorted(sorted_comb, ref) + 1)

        n1 = len(dist)
        n2 = len(ref)

        u1 = n1*n2 + n1*(n1+1)/2 - r1
        u2 = n1*n2 + n2*(n2+1)/2 - r2

        res[d] = min(u1, u2)
    
    return res

# CDF difference area ---------------------------------------------------------

def cdf_diff_area(data, ref):
    """
    Compute the area between the 2 CDF 
    """
    #data
    num_simulations = data.shape[0]
    distances = np.zeros(num_simulations)
    
    ref = np.sort(ref)
    m = len(ref)
    cdf_ref_dist = np.arange(1, m + 1) / m

    for d in range(num_simulations):
        dist = data[d, :]
        dist = np.sort(dist)
        n = len(dist)
        cdf_dist = np.arange(1, n + 1) / n

        #interpolation 
        cdf_ref_dist_interp = np.interp(dist, ref, cdf_ref_dist)

        #test 
        cdf_diff = cdf_dist - cdf_ref_dist_interp
        area = np.trapz(cdf_diff, dist)
        distances[d] = area
    
    return distances

#==============================================================================
# ROC curve
#==============================================================================

def roc(t0, t1, tref, test_statistic, multiple_kernels=False):

    N = len(t0)

    # Computing test statistic of all the distributions
    if multiple_kernels:
        t0_values = np.zeros(t0.shape[1])
        t1_values = np.zeros(t0.shape[1])
        # Sum the test over the different kernels
        for k in range(3):
            t0_values += test_statistic(t0[k], tref[k])
            t1_values += test_statistic(t1[k], tref[k])
    else:
        t0_values = test_statistic(t0, tref)
        t1_values = test_statistic(t1, tref)

    # Computng the maximum treshold value
    sup = np.max(np.concatenate([t0_values, t1_values]))

    thresholds = np.linspace(0, sup, 1000)
    
    pfa = np.zeros(len(thresholds))
    pdet = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        pfa[i] = np.sum(t0_values > threshold) / N
        pdet[i] = np.sum(t1_values > threshold) / N

    return pfa, pdet, thresholds

# Plot ------------------------------------------------------------------------

def plot_roc(t0, t1, tref, test_statistics):

    _, axs = plt.subplots(1, 4, figsize=(25, 5))

    for k in range(4):
        for ts_name, ts in test_statistics.items():
            if k < 3:
                Pfa, Pdet, _ = roc(t0[k], t1[k], tref[k], ts)
                axs[k].set_title(f"Kernel {k+1}")
            else:
                Pfa, Pdet, _ = roc(t0, t1, tref, ts, multiple_kernels=True)
                axs[k].set_title(f"All kernels")
            axs[k].plot(Pfa, Pdet, label=ts_name)
            axs[k].set_xlabel("Pfa")
            axs[k].set_ylabel("Pdet")
            axs[k].legend()

    plt.suptitle(f"ROC curves")
    plt.show()