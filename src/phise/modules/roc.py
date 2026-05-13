import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
from copy import deepcopy as copy
import astropy.units as u
from .test_statistics import ALL_TESTS
from scipy import stats


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(y, x))
    return float(np.trapz(y, x))

#==============================================================================
# Generate ROC vectors
#==============================================================================

def generate_data(ctx_h1, ctx_h0=None, nmc: int = 1000, size: int = 1000, progress_callback=None, randomize_position: bool = False):
    """Generate two sets of statistic vectors under H0 and H1.

    This simulates observations with and without companion(s) to build
    statistic arrays of shape ``(nb_processed_outputs, nmc, size)`` under H0 (no companion)
    and H1 (with companions), then concatenates each set along the first
    axis.

    Notes:
        - Assumes a compatible ``Context`` object exists (see
          ``phise.classes.context.Context``) and relies on its ``observe()``
          method.
        - To avoid circular imports, the ``Context`` import is local and only
          used if ``ctx`` is ``None``.

    Args:
        ctx_h1: Observation context for H1.
        ctx_h0: Observation context for H0. If ``None``, a copy of ``ctx_h1`` with no companions is used.
        nmc: Number of Monte-Carlo realizations.
        size: Number of samples per realization.
        progress_callback (callable, optional): function accepting a float (0-1)
            representing the progress.
        randomize_position (bool, optional): If True, randomizes companion position (uniform in FOV) for each sample.
            If False, uses the fixed position defined in `ctx_h1`. Defaults to False.

    Returns:
        Tuple ``(T0, T1)`` where:
        - T0: Vectors under H0, shape ``(nb_raw_outputs + nb_processed_outputs, nmc, size)``.
        - T1: Vectors under H1, shape ``(nb_raw_outputs + nb_processed_outputs, nmc, size)``.

    Raises:
        ValueError: If ``ctx`` contains no companions.
    """

    if ctx_h0 is None:
        ctx_h0 = copy(ctx_h1)
        ctx_h0.target.companions = []

    nb_raw = ctx_h1.chip.nb_raw_outputs
    nb_proc = ctx_h1.chip.nb_processed_outputs
    T0 = np.zeros((nb_raw + nb_proc, nmc, size))
    T1 = np.zeros((nb_raw + nb_proc, nmc, size))

    fov = ctx_h1.interferometer.fov.to(u.mas).value

    for i in range(nmc):
        if progress_callback:
            progress_callback(i / nmc)
        else:
            print(f'⌛ Generating vectors... {round(i / nmc * 100, 2)}%', end='\r')
        
        for j in range(size):
            if randomize_position:
                for c in ctx_h1.target.companions:
                    c.θ = np.random.uniform(0, 2 * np.pi) * u.rad
                    c.ρ = np.random.uniform(fov / 10, fov) * u.mas

            raw_h0 = ctx_h0.observe()
            raw_h1 = ctx_h1.observe()

            proc_h0 = ctx_h0.interferometer.chip.process_outputs(raw_h0)
            proc_h1 = ctx_h1.interferometer.chip.process_outputs(raw_h1)

            T0[:nb_raw, i, j] = raw_h0
            T0[nb_raw:, i, j] = proc_h0
            T1[:nb_raw, i, j] = raw_h1
            T1[nb_raw:, i, j] = proc_h1

    if progress_callback:
        progress_callback(1.0)
    else:
        print('✅ Vectors generation complete')
    
    return (T0, T1)

#==============================================================================
# Compute ROC curve
#==============================================================================

def compute_roc_curve(t0: np.ndarray, t1: np.ndarray, test: callable):
    t0_stats = np.array([test(t0[i], t0[i + 1]) if i + 1 < t0.shape[0] else test(t0[i], t0[0]) for i in range(t0.shape[0])])
    t1_stats = np.array([test(t1[i], t0[i]) for i in range(t1.shape[0])])
    all_stats = np.concatenate([t0_stats, t1_stats])
    thresholds = np.linspace(np.min(all_stats), np.max(all_stats), 1000)
    pdet = []
    pfa = []
    for thresh in thresholds:
        tp = np.sum(t1_stats > thresh)
        fn = np.sum(t1_stats <= thresh)
        fp = np.sum(t0_stats > thresh)
        tn = np.sum(t0_stats <= thresh)
        pdet.append(tp / (tp + fn))
        pfa.append(fp / (fp + tn))
    return (np.array(pfa), np.array(pdet), thresholds)

#==============================================================================
# Plot ROC curves
#==============================================================================

def plot_roc_curves(t0: np.ndarray, t1: np.ndarray, tests: dict=ALL_TESTS, figsize=(6, 6), save_as=None):
    plt.figure(figsize=figsize, constrained_layout=True)
    for (name, test) in tests.items():
        (pfa, pdet, thresholds) = compute_roc_curve(t0, t1, test)
        plt.plot(pfa, pdet, label=f'{name}')
        power = np.round(np.abs(_trapezoid_integral(pdet - pfa, pfa)) * 200, 2)
        print(f'Power of {name}: {power}%')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    if save_as:
        utils.save_plot(save_as, "roc_curves.png")
    plt.show()

#==============================================================================
# Neyman Pearson lemma: optimal test statistic
#==============================================================================

def neyman_pearson_lemma(ctx_h1, output_index, ctx_h0=None, save_as=None, tests=ALL_TESTS):
    tests = copy(tests)

    if ctx_h0 is None:
        if ctx_h1.target.companions == []:
            raise ValueError('No companion found in the H1 context. Please add a companion or specify the H0 context')
        ctx_h0 = copy(ctx_h1)
        ctx_h0.target.companions = []

    # Generate reference distribution using the numerical model ---------------

    print('⌛ Generating distributions...')
    samples = 10_000
    bins = np.sqrt(samples).astype(int)
    h0_data_kn = np.empty(samples)
    h1_data_kn = np.empty(samples)
    for i in range(samples):
        print(f'{(i + 1) / samples * 100:.2f}% ({i + 1}/{samples})', end='\r')
        outs_h0 = ctx_h0.observe()
        ker_h0 = ctx_h0.interferometer.chip.process_outputs(outs_h0)
        h0_data_kn[i] = np.concatenate([outs_h0, ker_h0])[output_index]
        outs_h1 = ctx_h1.observe()
        ker_h1 = ctx_h1.interferometer.chip.process_outputs(outs_h1)
        h1_data_kn[i] = np.concatenate([outs_h1, ker_h1])[output_index]
    print('✅ Distributions generated.')

    # Cost function for imb fit -----------------------------------------------

    # def imb_cost(params, data):
    #     μ, σ, ν = params
    #     pdf_vals = imb(data, μ, σ, ν)
    #     pdf_vals /= np.trapz(pdf_vals, data)
    #     hist_vals, bin_edges = np.histogram(data, bins=bins, density=True)
    #     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #     model_vals = imb(bin_centers, μ, σ, ν)
    #     model_vals /= np.trapz(model_vals, bin_centers)
    #     cost = np.sum((hist_vals - model_vals) ** 2)
    #     return cost

    # Fit distributions -------------------------------------------------------

    # (x0, γ0) = stats.cauchy.fit(h0_data_kn)
    # (x1, γ1) = stats.cauchy.fit(h1_data_kn)
    # (μ0, b0) = stats.laplace.fit(h0_data_kn)
    # (μ1, b1) = stats.laplace.fit(h1_data_kn)
    # (β0, m0, s0) = stats.gennorm.fit(h0_data_kn)
    # (β1, m1, s1) = stats.gennorm.fit(h1_data_kn)
    kde_h0 = stats.gaussian_kde(h0_data_kn)
    kde_h1 = stats.gaussian_kde(h1_data_kn)
    # imb fit for h0
    # initial_guess_h0 = [np.median(h0_data_kn), np.std(h0_data_kn), 0.8]
    # result_h0 = minimize(imb_cost, initial_guess_h0, args=(h0_data_kn,), bounds=[(None, None), (1e-5, None), (2.1, None)])
    # μ_imb0, σ_imb0, ν_imb0 = result_h0.x
    # imb fit for h1
    # initial_guess_h1 = [np.median(h1_data_kn), np.std(h1_data_kn), 0.8]
    # result_h1 = minimize(imb_cost, initial_guess_h1, args=(h1_data_kn,), bounds=[(None, None), (1e-5, None), (2.1, None)])
    # μ_imb1, σ_imb1, ν_imb1 = result_h1.x

    # Init plot
    x = np.linspace(min(np.min(h0_data_kn), np.min(h1_data_kn)), max(np.max(h0_data_kn), np.max(h1_data_kn)), 1000)
    plt.figure(figsize=(10, 6))

    # Reference distributions
    plt.hist(h0_data_kn, bins=bins, density=True, alpha=0.5, label='h0 data', color='blue', log=True)
    plt.hist(h1_data_kn, bins=bins, density=True, alpha=0.5, label='h1 data', color='orange', log=True)
    
    # Fitted distributions
    # plt.plot(x, stats.cauchy.pdf(x, loc=x0, scale=γ0), 'b--', label='h0 cauchy fit', linewidth=2)
    # plt.plot(x, stats.cauchy.pdf(x, loc=x1, scale=γ1), 'r--', label='h1 cauchy fit', linewidth=2)
    # plt.plot(x, stats.laplace.pdf(x, loc=μ0, scale=b0), 'b:', label='h0 laplace fit', linewidth=2)
    # plt.plot(x, stats.laplace.pdf(x, loc=μ1, scale=b1), 'r:', label='h1 laplace fit', linewidth=2)
    # plt.plot(x, stats.gennorm.pdf(x, β0, m0, s0), 'b-.', label='h0 gennorm fit', linewidth=2)
    # plt.plot(x, stats.gennorm.pdf(x, β1, m1, s1), 'r-.', label='h1 gennorm fit', linewidth=2)
    # plt.plot(x, 0.5 * stats.cauchy.pdf(x, loc=x0, scale=γ0) + 0.5 * stats.laplace.pdf(x, loc=μ0, scale=b0), 'b.', label='h0 mix fit', linewidth=2)
    # plt.plot(x, 0.5 * stats.cauchy.pdf(x, loc=x1, scale=γ1) + 0.5 * stats.laplace.pdf(x, loc=μ1, scale=b1), 'r.', label='h1 mix fit', linewidth=2)
    plt.plot(x, kde_h0(x), 'b-', label='h0 KDE', linewidth=2)
    plt.plot(x, kde_h1(x), 'r-', label='h1 KDE', linewidth=2)
    # plt.plot(x, imb(x, μ_imb0, σ_imb0, ν_imb0), 'b-.', label='h0 IMB fit', linewidth=2)
    # plt.plot(x, imb(x, μ_imb1, σ_imb1, ν_imb1), 'r-.', label='h1 IMB fit', linewidth=2)
    
    # Finalize plot
    plt.xlabel('Test Statistic Value')
    plt.ylabel('Density')
    plt.title('Distributions and Fits')
    plt.legend()
    if save_as:
        utils.save_plot(save_as, "distributions.png")
    plt.show()

    # Generate random distributions from the fitted models
    print('⌛ Generating random distributions from the fitted models...')
    nmc = 1000
    samples = 1000
    t0_sim = np.empty((nmc, samples))
    t1_sim = np.empty((nmc, samples))
    # t0_cauchy = np.empty((nmc, samples))
    # t1_cauchy = np.empty((nmc, samples))
    # t0_laplace = np.empty((nmc, samples))
    # t1_laplace = np.empty((nmc, samples))
    # t0_gennorm = np.empty((nmc, samples))
    # t1_gennorm = np.empty((nmc, samples))
    # t0_imb = np.empty((nmc, samples))
    # t1_imb = np.empty((nmc, samples))
    for i in range(nmc):
        print(f'{(i + 1) / nmc * 100:.2f}% ({i + 1}/{nmc})', end='\r')
    #     t0_cauchy[i] = stats.cauchy.rvs(loc=x0, scale=γ0, size=samples)
    #     t1_cauchy[i] = stats.cauchy.rvs(loc=x1, scale=γ1, size=samples)
    #     t0_laplace[i] = stats.laplace.rvs(loc=μ0, scale=b0, size=samples)
    #     t1_laplace[i] = stats.laplace.rvs(loc=μ1, scale=b1, size=samples)
    #     # t0_gennorm[i] = stats.gennorm.rvs(beta=β0, loc=m0, scale=s0, size=samples)
    #     # t1_gennorm[i] = stats.gennorm.rvs(beta=β1, loc=m1, scale=s1, size=samples)
        for j in range(samples):
            outs0 = ctx_h0.observe()
            k0 = ctx_h0.interferometer.chip.process_outputs(outs0)
            t0_sim[i, j] = k0[0]
            outs1 = ctx_h1.observe()
            k1 = ctx_h1.interferometer.chip.process_outputs(outs1)
            t1_sim[i, j] = k1[0]

    #         # random in imb distribution
    #         def sample_imb_random(mu, sigma, nu, x_min=None, x_max=None, n_grid=2000):
    #             # Build sampling grid from empirical ranges if not provided
    #             if x_min is None:
    #                 x_min = min(np.min(h0_data_kn), np.min(h1_data_kn))
    #             if x_max is None:
    #                 x_max = max(np.max(h0_data_kn), np.max(h1_data_kn))
    #             x = np.linspace(x_min, x_max, n_grid)
    #             pdf = imb(x, mu, sigma, nu)
    #             pdf = np.clip(pdf, 0.0, None)
    #             dx = x[1] - x[0]
    #             cdf = np.cumsum(pdf) * dx
    #             if not np.isfinite(cdf[-1]) or cdf[-1] <= 0:
    #                 # fallback to normal jitter if CDF invalid
    #                 return np.random.normal(loc=mu, scale=max(sigma, 1e-9))
    #             cdf /= cdf[-1]
    #             u = np.random.rand()
    #             return float(np.interp(u, cdf, x))

    #         # draw one sample from each fitted IMB for h0 and h1
    #         t0_imb[i, j] = sample_imb_random(μ_imb0, σ_imb0, ν_imb0)
    #         t1_imb[i, j] = sample_imb_random(μ_imb1, σ_imb1, ν_imb1)
    print('✅ Random distributions generated.')
    print('⌛ Plotting ROC curves...')

    # Define likelihood ratio tests for fitted distributions
    # def lr_cauchy(u, v):
    #     return np.sum(np.log((1 + ((u - x0) / γ0) ** 2) / (1 + ((u - x1) / γ1) ** 2)))

    # def lr_laplace(u, v):
    #     return np.sum(np.abs(u - μ0) / b0 - np.abs(u - μ1) / b1)

    # def lr_gennorm(u, v):
    #     return np.sum(np.abs((u - m0) / s0) ** β0 - np.abs((u - m1) / s1) ** β1)

    def lr_kde(u, v):
        return np.sum(np.log(kde_h1(u) / kde_h0(u)))
    
    # def lr_imb(u, v):
    #     return np.sum(np.log(imb(u, μ_imb1, σ_imb1, ν_imb1) / imb(u, μ_imb0, σ_imb0, ν_imb0)))
    
    # Plot ROC curves
    # print('📊 Simulated case:')
    # plot_rocs(t0_sim, t1_sim, tests=tests, figsize=(4, 4))
    # print('📊 Cauchy case:')
    # tests['Likelihood Ratio'] = lr_cauchy
    # plot_rocs(t0_cauchy, t1_cauchy, tests=tests, figsize=(4, 4))
    # print('📊 Laplace case:')
    # tests['Likelihood Ratio'] = lr_laplace
    # plot_rocs(t0_laplace, t1_laplace, tests=tests, figsize=(4, 4))
    # print('📊 Gennorm case:')
    # tests['Likelihood Ratio'] = lr_gennorm
    # plot_rocs(t0_gennorm, t1_gennorm, tests=tests, figsize=(4, 4))
    print('📊 KDE case:')
    tests['Likelihood Ratio'] = lr_kde
    plot_roc_curves(t0_sim, t1_sim, tests=tests, figsize=(4, 4))
    # print('📊 IMB case:')
    # tests['Likelihood Ratio'] = lr_imb
    # plot_rocs(t0_imb, t1_imb, tests=tests, figsize=(4, 4))
    print('✅ ROC curves plotted.')