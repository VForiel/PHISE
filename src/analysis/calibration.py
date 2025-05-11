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

#==============================================================================
# Calibration approaches
#==============================================================================

def genetic_approach(ctx:Context = None, β:float = 0.9):

    if ctx is None:
        from .default_context import ctx

    # Introduce random noise
    ctx = copy(ctx)
    ctx.interferometer.kn.σ = np.abs(np.random.normal(0, 1, 14)) * ctx.interferometer.λ

    ctx = calibration.genetic(ctx=ctx, β=β, plot=True)

def obstruction_approach(ctx:Context = None, N:int = 1000):

    if ctx is None:
        from .default_context import ctx
        
    # Introduce random noise
    ctx = copy(ctx)
    ctx.interferometer.kn.σ = np.abs(np.random.normal(0, 1, 14)) * ctx.interferometer.λ

    ctx = calibration.obstruction(ctx=ctx, N=N, plot=True)

#==============================================================================
# Comparison of the two algorithms
#==============================================================================

def compare_approaches(ctx:Context = None):

    if ctx is None:
        from .default_context import ctx

    kn = ctx.interferometer.kn
    λ = ctx.interferometer.λ

    β_res = 10
    βs, dβ = np.linspace(0.5, 0.99, β_res, retstep=True)
    Ns = [10, 100, 1000, 10_000]#, 100_000]
    samples = 100

    # fig, axs = plt.subplots(1, 1, figsize=(5, 5))

    shots = []
    for β in βs:
        for i in range(samples):
            print(f'Gen.: β={β:.3f}, sample={i+1}/{samples}          ', end='\r')

            # Introduce random noise
            ctx_gen = copy(ctx)
            ctx_gen.interferometer.kn.σ = np.abs(np.random.normal(0, 1, len(kn.σ))) * λ

            # Calibrate
            ctx_gen, history = calibration.genetic(ctx=ctx, β=β, verbose=False, ret_history=True)

            # Calculate depth
            ψ = np.ones(4) * (1+0j) * np.sqrt(1/4)
            _, d, b = ctx_gen.interferometer.kn.propagate_fields(ψ=ψ, λ=λ)
            di = np.abs(d)**2
            k = np.array([di[0] - di[1], di[2] - di[3], di[4] - di[5]])
            depth = np.sum(np.abs(k)) / np.abs(b)**2

            # Store results
            shots.append((len(history['bright']), depth))
        
    x, y = zip(*shots)
    x = np.array(x); y = np.array(y)
    plt.scatter(np.random.uniform(x-x/10, x+x/10), y, c='tab:blue', s=5, label='Genetic')

    slope, intercept, r_value, p_value, std_err = linregress(np.log10(x), np.log10(y))
    print(slope, intercept)
    x_values = np.linspace(min(x), max(x), 500)
    y_fit = 10**intercept * x_values**slope
    plt.plot(x_values, y_fit, 'tab:cyan', linestyle='--', label=f'Gen. fit')

    print("")

    shots = []
    for j, N in enumerate(Ns):
        for i in range(samples):
            print(f'Obs.: N={N}, sample={i+1}/{samples}          ', end='\r')

            # Introduce random noise
            ctx_obs = copy(ctx)
            ctx_obs.interferometer.kn.σ = np.abs(np.random.normal(0, 1, len(kn.σ))) * λ

            # Calibrate
            ctx_obs = calibration.obstruction(ctx=ctx, N=N, plot=False)

            # Calculate depth
            ψ = np.ones(4) * (1+0j) * np.sqrt(1/4)
            _, d, b = ctx_obs.interferometer.kn.propagate_fields(ψ=ψ, λ=λ)
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