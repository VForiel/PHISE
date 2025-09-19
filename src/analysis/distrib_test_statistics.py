import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
from copy import deepcopy as copy
import astropy.units as u

from src.classes.context import Context
from src.modules.test_statistics import ALL_TESTS
from scipy.optimize import minimize
from scipy import stats

from src.modules import test_statistics as ts

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
        ctx = Context.get_VLTI()

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

        t0, t1 = ts.get_vectors(ctx=ctx, nmc=nmc, size=maxpoints)
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

#==============================================================================
# Neyman-Pearson
#==============================================================================

def np_benchmark(ctx:Context=None):

    # Get contexts ------------------------------------------------------------

    if ctx is None:
        ctx = Context.get_VLTI()

        ctx.interferometer.kn.σ = np.zeros(14) * u.nm
        ctx.target.companions[0].c = 1e-3
        ctx.monochromatic = False

    else:
        ctx = copy(ctx)

    ctx_star_only = copy(ctx)
    ctx_star_only.target.companions = []

    # Generate distributions --------------------------------------------------

    print("⌛ Generating distributions...")

    samples = 10_000
    bins = np.sqrt(samples).astype(int)
    
    h0_data = np.empty(samples)
    h1_data = np.empty(samples)

    for i in range(samples):
        print(f"{(i+1)/samples*100:.2f}% ({i+1}/{samples})", end='\r')
        _, k, _ = ctx_star_only.observe()
        h0_data[i] = k[0]
        _, k, _ = ctx.observe()
        h1_data[i] = k[0]

    print("✅ Distributions generated.")

    # Model definition --------------------------------------------------------

    # # Laplace distribution

    # def laplace(x, μ, b):
    #     return (1/(b)) * np.exp(-np.abs((x - μ))/(b))

    # def laplace_cost(params, data):
    #     μ, b = params
        
    #     # True histogram (empirical density)
    #     y, bin_edges = np.histogram(data, bins=bins, density=True)
    #     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    #     # Model histogram
    #     s = laplace(bin_centers, μ, b)
    #     s /= np.sum(s)

    #     return np.sum(np.abs(y - s))

    # # Cauchy distribution
    # def cauchy(x, x0, γ):
    #     return (1/(np.pi*γ * (1 + ((x - x0)/γ)**2)))

    # def cauchy_cost(params, data):
    #     x0, γ = params
        
    #     # True histogram (empirical density)
    #     y, bin_edges = np.histogram(data, bins=bins, density=True)
    #     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    #     # Model histogram
    #     s = cauchy(bin_centers, x0, γ)
    #     s /= np.sum(s)

    #     return np.sum(np.log(1+((y - s)/γ)**2))
    
    # model = cauchy
    # cost = cauchy_cost  

    # Fit model ---------------------------------------------------------------

    # print("⌛ Fitting distributions...")

    # # Fit for h0_data
    # res_h0 = minimize(cost, x0=[np.median(h0_data), np.std(h0_data)/2], args=(h0_data,), method='Nelder-Mead')
    # μ_h0, b_h0 = res_h0.x

    # if res_h0.success:
    #     print(f"✅ Fitted distrib under H0: μ={μ_h0:.3e}, b={b_h0:.3e}")
    # else:
    #     print("❌ Fitting for H0 did not converge:", res_h0.message)

    # # Fit for h1_data
    # res_h1 = minimize(cost, x0=[np.median(h1_data), np.std(h1_data)/2], args=(h1_data,), method='Nelder-Mead')
    # μ_h1, b_h1 = res_h1.x

    # if res_h1.success:
    #     print(f"✅ Fitted distrib under H1: μ={μ_h1:.3e}, b={b_h1:.3e}")
    # else:
    #     print("❌ Fitting for H1 did not converge:", res_h1.message)

    x0, γ0 = stats.cauchy.fit(h0_data)
    x1, γ1 = stats.cauchy.fit(h1_data)

    μ0, b0 = stats.laplace.fit(h0_data)
    μ1, b1 = stats.laplace.fit(h1_data)

    # Plot distributions ------------------------------------------------------

    x = np.linspace(min(np.min(h0_data), np.min(h1_data)), max(np.max(h0_data), np.max(h1_data)), 1000)

    plt.figure(figsize=(10, 6))
    _, h0_bins, _ = plt.hist(h0_data, bins=bins, density=True, alpha=0.5, label='h0 data', color='blue', log=True)
    _, h1_bins, _ = plt.hist(h1_data, bins=bins, density=True, alpha=0.5, label='h1 data', color='orange', log=True)

    plt.plot(x, stats.cauchy.pdf(x, loc=x0, scale=γ0), 'b--', label='h0 cauchy fit', linewidth=2)
    plt.plot(x, stats.cauchy.pdf(x, loc=x1, scale=γ1), 'r--', label='h1 cauchy fit', linewidth=2)

    plt.plot(x, stats.laplace.pdf(x, loc=μ0, scale=b0), 'b:', label='h0 laplace fit', linewidth=2)
    plt.plot(x, stats.laplace.pdf(x, loc=μ1, scale=b1), 'r:', label='h1 laplace fit', linewidth=2)

    plt.xlabel('Test Statistic Value')
    plt.ylabel('Density')
    plt.title('Distributions and Fits')
    plt.legend()
    plt.show()

    # Generate random distribution following the model ------------------------

    print("⌛ Generating random distributions from the fitted models...")

    nmc = 1000
    samples = 1000
    t0_cauchy = np.empty((nmc, samples))
    t1_cauchy = np.empty((nmc, samples))
    t0_laplace = np.empty((nmc, samples))
    t1_laplace = np.empty((nmc, samples))

    for i in range(nmc):
        t0_cauchy[i] = stats.cauchy.rvs(loc=x0, scale=γ0, size=samples)
        t1_cauchy[i] = stats.cauchy.rvs(loc=x1, scale=γ1, size=samples)
        t0_laplace[i] = stats.laplace.rvs(loc=μ0, scale=b0, size=samples)
        t1_laplace[i] = stats.laplace.rvs(loc=μ1, scale=b1, size=samples)

    print("✅ Random distributions generated.")

    # Plot ROC ----------------------------------------------------------------

    print("⌛ Plotting ROC curves...")

    def lr_cauchy(u, v):
        return np.sum(np.log(
            (1 + ((u - x0)/γ0)**2) /
            (1 + ((u - x1)/γ1)**2)
        ))
    
    def lr_laplace(u, v):
        return np.sum(
            (np.abs(u - μ1)/b1) - (np.abs(u - μ0)/b0)
        )

    tests = copy(ALL_TESTS)

    tests['Likelihood Ratio'] = lr_cauchy
    plot_rocs(t0_cauchy, t1_laplace, tests=tests, figsize=(6,6))

    tests['Likelihood Ratio'] = lr_laplace
    plot_rocs(t0_laplace, t1_laplace, tests=tests, figsize=(6,6))

