

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
from copy import deepcopy as copy
import astropy.units as u
from scipy import stats

from src.classes.context import Context
from . import contexts

def get_vectors(ctx:Context=None, nmc:int=1000, size:int=1000):

    if ctx is None:
        ctx = contexts.get_VLTI()
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

#==============================================================================
# ROC curve
#==============================================================================

def roc(t0:np.ndarray, t1:np.ndarray, test:callable):

    # Compute test statistics for each distribution
    t0_stats = np.array([test(t0[i], t0[i+1]) if i+1 < t0.shape[0] else test(t0[i], t0[0]) for i in range(t0.shape[0])])
    t1_stats = np.array([test(t1[i], t0[i]) for i in range(t1.shape[0])])

    # Define thresholds over the range of both statistics
    all_stats = np.concatenate([t0_stats, t1_stats])
    thresholds = np.linspace(np.min(all_stats), np.max(all_stats), 1000)

    pdet = []  # True Positive Rate (sensitivity)
    pfa = []  # False Positive Rate (1 - specificity)

    for thresh in thresholds:
        tp = np.sum(t1_stats > thresh)
        fn = np.sum(t1_stats <= thresh)
        fp = np.sum(t0_stats > thresh)
        tn = np.sum(t0_stats <= thresh)
        pdet.append(tp / (tp + fn))
        pfa.append(fp / (fp + tn))

    return np.array(pfa), np.array(pdet), thresholds

def plot_rocs(t0:np.ndarray, t1:np.ndarray, tests:dict=ALL_TESTS, figsize=(6,6)):

    plt.figure(figsize=figsize, constrained_layout=True)
    for name, test in tests.items():
        pfa, pdet, thresholds = roc(t0, t1, test)
        plt.plot(pfa, pdet, label=f"{name}")
        power = np.round(np.abs(np.trapz(pdet-pfa, pfa))*200,2)
        print(f"Power of {name}: {power}%")
    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

#==============================================================================
# Power over nb data
#==============================================================================

def test_power(ctx=None, tests=ALL_TESTS, nmc=100, bootstrap=10, resolution=10, maxpoints=1000):

    if ctx is None:
        ctx = contexts.get_VLTI()

        ctx.interferometer.kn.σ = np.zeros(14) * u.nm
        ctx.target.companions[0].c = 1e-4

    else:
        ctx = copy(ctx)

    nb_values = np.logspace(1, np.log10(maxpoints), resolution, endpoint=True).astype(int)

    auc_bootstrap = copy(tests)
    power_bootstrap = copy(tests)
    for ts_name, ts in tests.items():
        auc_bootstrap[ts_name] = []
        power_bootstrap[ts_name] = []

    # Bootstrap

    for b in range(bootstrap):
        print("="*10, f"\nBootstrap {b+1}/{bootstrap}...\n", "="*10, sep='')

        # Generate dataset

        t0, t1 = get_vectors(ctx=ctx, nmc=nmc, size=maxpoints)
        dataset = []
        for nb in nb_values:
            # Take nb random points from t0 and t1
            indices = np.random.choice(t0.shape[1], nb, replace=False)
            t0_subset = t0[:,indices]
            t1_subset = t1[:,indices]
            dataset.append((t0_subset, t1_subset))

        # Loop over tests

        print(f"Computing tests power...", end='\r')
        for ts_name, ts in tests.items():

            auc = np.zeros(len(dataset))
            power = np.zeros(len(dataset))

            # Loop over distributions size

            for i, (t0, t1) in enumerate(dataset):
                pfa, pdet, _ = roc(t0=t0, t1=t1, test=ts)

                pfa, pdet = zip(*sorted(zip(pfa, pdet)))
                pfa = np.array(pfa)
                pdet = np.array(pdet)
                
                auc[i] = np.round(np.abs(np.trapz(pdet-pfa, pfa))*200,2)
                power[i] = np.round(pdet[np.argmax(pfa[pfa <= 0.01])]*100,2)
            
            auc_bootstrap[ts_name].append(auc)
            power_bootstrap[ts_name].append(power)
        print("Done computing tests power ✅")

    # Plot the results

    _, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Associate each test to a color
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(tests)))

    for i, (ts_name, ts) in enumerate(tests.items()):

        auc_mean = np.mean(auc_bootstrap[ts_name], axis=0)
        power_mean = np.mean(power_bootstrap[ts_name], axis=0)

        auc_std = np.std(auc_bootstrap[ts_name], axis=0)
        power_std = np.std(power_bootstrap[ts_name], axis=0)

        axs[0].plot(nb_values, auc_mean, label=ts_name, color=colors[i])
        axs[0].fill_between(nb_values, auc_mean - auc_std, auc_mean + auc_std, color=colors[i], alpha=0.2)

        axs[1].plot(nb_values, power_mean, label=ts_name, color=colors[i])
        axs[1].fill_between(nb_values, power_mean - power_std, power_mean + power_std, color=colors[i], alpha=0.2)

    for ax in axs:
        ax.set_xscale('log')
        ax.set_xlabel("Number of data points")

    axs[0].set_ylabel("Global power [%]")
    axs[1].set_ylabel("Power [%]")
    axs[0].set_title("AUC (global power)")
    axs[1].set_title("Power ($P_{det}$ at $P_{fa}<1\%$)")
    plt.legend()

#==============================================================================
# P-value
#==============================================================================

def plot_p_values(t0, t1, tests=ALL_TESTS):

    col = min(2, len(tests))
    row = int(np.ceil(len(tests) / col))

    _, axs = plt.subplots(row, col, figsize=(5*col, 5*row))
    axs = axs.flatten()
    
    for t, (ts_name, ts) in enumerate(tests.items()):
        sup = 0
        values = []
        for u, v in zip(t1, t0):
            values.append(ts(u, v))
        sup = max(sup, np.max(values))
        thresholds = np.linspace(0, sup, 1000)
        p_values = np.zeros(len(thresholds))
        for i, threshold in enumerate(thresholds):
            p_values[i] = np.sum(values > threshold) / len(values)
        axs[t].plot(thresholds, p_values, label=ts_name)
        axs[t].hlines(0.05, 0, sup, color='red', linestyle='dashed')
        axs[t].set_xlabel("Test value")
        axs[t].set_ylabel("P-value")
        axs[t].set_title(f"P-values for {ts_name}")
        axs[t].legend()
    plt.show()