import numpy as np
import numba as nb
import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
from copy import deepcopy as copy
import astropy.units as u
from scipy import stats

from src.classes.context import Context
from . import contexts

def run(ctx:Context=None, Δλ=0.2*u.um, n=11, figsize=(5,5)):

    if n%2 == 0:
        n += 1

    if ctx is None:
        ctx = contexts.get_VLTI()
        ctx.interferometer.kn.σ = np.zeros(14) * u.m
        ctx.Γ = 0 * u.nm
        ctx.target.companions = []
        ctx.monochromatic = True
    else:
        ctx = copy(ctx)

    λ0 = ctx.interferometer.λ.to(u.um)
    λs = np.linspace(λ0.value - Δλ.value/2, λ0.value + Δλ.value/2, n) * u.um

    data = np.empty((n,))
    
    plt.figure(figsize=figsize)
    plt.axvline(λ0.to(u.nm).value, color='k', ls='--', label=r"$\lambda_0$")

    for i, λ in enumerate(λs):
        print(f"⌛ Calibrating at {round(λ.value,3)} um... {round(i/n * 100,2)}%", end="\r")
        ctx.interferometer.λ = λ
        # ctx.calibrate_obs(n=1_000)
        ctx.calibrate_gen(β=0.961, verbose=False)
        _, k, b = ctx.observe()
        data[i] = np.mean(np.abs(k) / b)

        if λ == λ0:
            data2 = np.empty((n,))
            for j, λ in enumerate(λs):
                ctx.interferometer.λ = λ
                _, k, b = ctx.observe()
                data2[j] = np.mean(np.abs(k) / b)
                
            plt.plot(λs.to(u.nm).value, data2, color='gray', alpha=0.3, label=r"$\lambda_{cal} = \lambda_0$")

    print(f"✅ Done.{' '*30}")

    plt.plot(λs.to(u.nm).value, data, 'o-', label=r"$\lambda_{cal} = \lambda$")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Mean Kernel-Null Depth")
    plt.yscale('log')
    plt.title("Wavelength scan")
    plt.legend()
    plt.show()
