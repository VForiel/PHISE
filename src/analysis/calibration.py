# External libs
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from LRFutils import color
from scipy.stats import linregress
from copy import deepcopy as copy

# Internal libs
from .. import Context
from .. import calibration
from . import default_context

#==============================================================================
# Calibration approaches
#==============================================================================

# Genetic ---------------------------------------------------------------------

def genetic_approach(ctx:Context = None, β:float = 0.9, verbose=False, figsize=(10,5)):

    if ctx is None:
         ctx = default_context.get()
    else:
        ctx = copy(ctx)

    # Calibration is performed in a controled environment where there is no input cophasing error variation
    ctx.Γ = 0 * u.nm

    # Calibration is performed with only the on-axis source
    ctx.target.companions = []

    # Introduce random noise
    ctx.interferometer.kn.σ = np.abs(np.random.normal(0, 1, 14)) * ctx.interferometer.λ

    ctx = calibration.genetic(ctx=ctx, β=β, plot=True, verbose=verbose, figsize=figsize)

    print_kernel_null_depth(ctx)

    return ctx

# Obstruction -----------------------------------------------------------------

def obstruction_approach(ctx:Context = None, N:int = 1000):

    if ctx is None:
        ctx = default_context.get()
    else:
        ctx = copy(ctx)

    # Calibration is performed in a controled environment where there is no input cophasing error variation
    ctx.Γ = 0 * u.nm

    # Calibration is performed with only the on-axis source
    ctx.target.companions = []
        
    # Introduce random noise
    ctx.interferometer.kn.σ = np.abs(np.random.normal(0, 1, 14)) * ctx.interferometer.λ

    print(ctx.interferometer.kn.φ)
    print_kernel_null_depth(ctx)
    ctx = calibration.obstruction(ctx=ctx, N=N, plot=True)
    print(ctx.interferometer.kn.φ)
    print_kernel_null_depth(ctx)

    return ctx

#==============================================================================
# Calibration results
#==============================================================================

def print_kernel_null_depth(ctx:Context, N=1000):
    kernels = np.empty((N, 3))
    bright = np.empty(N)
    for i in range(N):
        _, k, b = ctx.observe()
        kernels[i] = k
        bright[i] = b

    k_mean = np.mean(kernels, axis=0)
    k_std = np.std(kernels, axis=0)
    b_mean = np.mean(bright)
    b_std = np.std(bright)

    print(f"Achieved Kernel-Null depth:")
    print("   Mean: " + " | ".join([f"{i / b_mean:.2e}" for i in k_mean]))
    print("   Std:  " + " | ".join([f"{i / b_mean:.2e}" for i in k_std]))

#==============================================================================
# Comparison of the two algorithms
#==============================================================================

def compare_approaches(ctx:Context = None):

    if ctx is None:
        ctx = default_context.get()
    else:
        ctx = copy(ctx)

    kn = ctx.interferometer.kn
    λ = ctx.interferometer.λ

    # Calibration is performed in a controled environment where there is no input cophasing error variation
    ctx.Γ = 0 * u.nm

    # Calibration is performed with only the on-axis source
    ctx.target.companions = []

    β_res = 10
    βs, dβ = np.linspace(0.5, 0.99, β_res, retstep=True)
    Ns = [10, 100, 1000, 10_000]#, 100_000]
    samples = 100

    # Genetic approach --------------------------------------------------------

    shots = []
    for β in βs:
        for i in range(samples):
            print(f'Gen.: β={β:.3f}, sample={i+1}/{samples}          ', end='\r')

            # Introduce random noise
            ctx.interferometer.kn.σ = np.abs(np.random.normal(0, 1, len(kn.σ))) * λ

            # Calibrate
            _, history = calibration.genetic(ctx=ctx, β=β, verbose=False, ret_history=True)

            # Calculate depth
            ψ = np.ones(4) * (1+0j) * np.sqrt(1/4)
            _, d, b = ctx.interferometer.kn.propagate_fields(ψ=ψ, λ=λ)
            di = np.abs(d)**2
            k = np.array([di[0] - di[1], di[2] - di[3], di[4] - di[5]])
            depth = np.sum(np.abs(k)) / np.abs(b)**2

            # Store results
            shots.append((len(history['depth']), depth))
        
    x, y = zip(*shots)
    x = np.array(x); y = np.array(y)
    plt.scatter(np.random.uniform(x-x/10, x+x/10), y, c='tab:blue', s=5, label='Genetic')

    slope, intercept, r_value, p_value, std_err = linregress(np.log10(x), np.log10(y))
    print(slope, intercept)
    x_values = np.linspace(min(x), max(x), 500)
    y_fit = 10**intercept * x_values**slope
    plt.plot(x_values, y_fit, 'tab:cyan', linestyle='--', label=f'Gen. fit')

    print("")
    
    # Obstraction approach ----------------------------------------------------

    shots = []
    for j, N in enumerate(Ns):
        for i in range(samples):
            print(f'Obs.: N={N}, sample={i+1}/{samples}          ', end='\r')

            # Introduce random noise
            ctx.interferometer.kn.σ = np.abs(np.random.normal(0, 1, len(kn.σ))) * λ

            # Calibrate
            calibration.obstruction(ctx=ctx, N=N, plot=False)

            # Calculate depth
            ψ = np.ones(4) * (1+0j) * np.sqrt(1/4)
            _, d, b = ctx.interferometer.kn.propagate_fields(ψ=ψ, λ=λ)
            di = np.abs(d)**2
            k = np.array([di[0] - di[1], di[2] - di[3], di[4] - di[5]])
            depth = np.sum(np.abs(k)) / np.abs(b)**2

            # Store results
            shots.append((7*N, depth))

    x, y = zip(*shots)
    x = np.array(x); y = np.array(y)
    plt.scatter(np.random.uniform(x-x/10, x+x/10), y, c='tab:orange', s=5, label='Obstruction')

    slope, intercept, r_value, p_value, std_err = linregress(np.log10(x), np.log10(y))
    print(slope, intercept)
    x_values = np.linspace(min(x), max(x), 500)
    y_fit = 10**intercept * x_values**slope
    plt.plot(x_values, y_fit, 'tab:red', linestyle='--', label=f'Obs. fit')

    plt.xlabel('# of iterations')
    plt.xscale('log')
    plt.ylabel('Depth')
    plt.yscale('log')
    plt.title('Efficiency of the calibration approaches')
    plt.legend()
    plt.show()