"""Module generated docstring."""
import numpy as np
import astropy.units as u
from copy import deepcopy as copy
import matplotlib.pyplot as plt
try:
    plt.rcParams['image.origin'] = 'lower'
except Exception:
    pass
from src import Context

def instant_distribution(ctx: Context=None, n=10000, stat=np.median, figsize=(10, 10)) -> np.ndarray:
    """
    Get the instantaneous distribution of the kernel nuller.

    Parameters
    ----------
    ctx : Context
        The context to use.
    n : int, optional
        The number of samples to take, by default 1000
    stat : function, optional
        The function to use to compute the statistic, by default np.median.

    Returns
    -------
    np.ndarray
        The instantaneous distribution of the kernel nuller.
    """
    if ctx is None:
        ctx = Context.get_VLTI()
        ctx.interferometer.kn.σ = np.zeros(14) * u.um
        ctx.target.companions[0].c = 0.1
    else:
        ctx = copy(ctx)
        if ctx.target.companions == []:
            raise ValueError('No companion in the context. Please add a companion to the target.')
    ctx.Δh = ctx.interferometer.camera.e.to(u.hour).value * u.hourangle
    ref_ctx = copy(ctx)
    ref_ctx.target.companions = []
    data = np.empty((n, 3))
    ref_data = np.empty((n, 3))
    for i in range(n):
        (_, k, b) = ctx.observe()
        data[i] = k / b
        (_, k, b) = ref_ctx.observe()
        ref_data[i] = k / b
    (_, axs) = plt.subplots(3, 1, figsize=figsize, sharex=True, constrained_layout=True)
    kmin_k = np.zeros(3)
    kmax_k = np.zeros(3)
    for k in range(3):
        combined = np.concatenate([data[:, k], ref_data[:, k]])
        (kmin_k[k], kmax_k[k]) = np.percentile(combined, [5, 95])
    kmin_k = np.min(kmin_k)
    kmax_k = np.max(kmax_k)
    for k in range(3):
        bins = np.linspace(kmin_k, kmax_k, 2 * int(np.sqrt(n)) + 1)
        combined = np.concatenate([data[:, k], ref_data[:, k]])
        weights_data = np.ones_like(data[:, k]) * (100.0 / data.shape[0])
        axs[k].hist(data[:, k], label='With companion(s)', bins=bins, alpha=0.5, color='blue', weights=weights_data)
        axs[k].axvline(stat(data[:, k]), color='blue', linestyle='--')
        weights_ref = np.ones_like(ref_data[:, k]) * (100.0 / ref_data.shape[0])
        axs[k].hist(ref_data[:, k], label='Star only', bins=bins, alpha=0.5, color='red', weights=weights_ref)
        axs[k].axvline(stat(ref_data[:, k]), color='red', linestyle='--')
        axs[k].set_ylabel('Occurrences (%)')
        axs[k].set_xlabel('Kernel-Null depth')
        axs[k].set_title(f'Kernel {k + 1}')
        axs[k].legend()
        (xmin_plot, xmax_plot) = np.percentile(combined, [5, 95])
        if xmin_plot == xmax_plot:
            xmin_plot -= 1e-12
            xmax_plot += 1e-12
        axs[k].set_xlim(xmin_plot, xmax_plot)
    plt.show()
    return (data, ref_data)

def time_evolution(ctx: Context=None, n=100, map=np.median) -> np.ndarray:
    """
    Get the time evolution of the kernel nuller.

    Parameters
    ----------
    ctx : Context
        The context to use.
    n : int, optional
        The number of samples to take at a given time, by default 1000.
    map : function, optional
        The function to use to map the data, by default np.median.

    Returns
    -------
    np.ndarray
        The time evolution of the kernel nuller. (n_h, 3)
    np.ndarray
        The reference time evolution of the kernel nuller (without input perturbation). (n_h, 3)
    """
    if ctx is None:
        ctx = Context.get_VLTI()
        ctx.interferometer.kn.σ = np.zeros(14) * u.um
        ctx.Γ = 10 * u.nm
    else:
        ctx = copy(ctx)
    ref_ctx = copy(ctx)
    ref_ctx.Γ = 0 * u.nm
    data = np.empty((len(ctx.get_h_range()), 3))
    ref_data = np.empty((len(ref_ctx.get_h_range()), 3))
    (_, k, b) = ctx.observation_serie(n=n)
    (_, ref_k, ref_b) = ref_ctx.observation_serie(n=1)
    k_depth = np.empty_like(k)
    ref_k_depth = np.empty_like(ref_k)
    for i in range(n):
        for h in range(len(ctx.get_h_range())):
            k_depth[i, h] = k[i, h] / b[i, h]
        for h in range(len(ref_ctx.get_h_range())):
            ref_k_depth[0, h] = ref_k[0, h] / ref_b[0, h]
    for h in range(len(ctx.get_h_range())):
        for k in range(3):
            data[h, k] = map(k_depth[:, h, k])
    for h in range(len(ref_ctx.get_h_range())):
        for k in range(3):
            ref_data[h, k] = ref_k_depth[0, h, k]
    (_, axs) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for k in range(3):
        axs[k].scatter(ctx.get_h_range(), data[:, k], label='Data')
        axs[k].plot(ref_ctx.get_h_range(), ref_data[:, k], label='Reference', linestyle='--')
        axs[k].set_ylabel(f'Kernel output')
        axs[k].set_xlabel('Time (hourangle)')
        axs[k].set_title(f'Kernel {k + 1}')
        axs[k].legend()
    plt.show()
    return (data, ref_data)