# External libs
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.optimize import curve_fit
from copy import deepcopy as copy
from LRFutils import color

# Internal libs
from ..classes.context import Context
from . import phase

#==============================================================================
# Genetic method
#==============================================================================

def genetic(
        ctx:Context,
        β: float,
        verbose: bool = False,
        plot:bool = False,
        figsize:tuple = (10, 5),
        ret_history:bool = False,
    ) -> Context:
    """
    Optimize the phase shifters offsets to maximize the nulling performance

    Parameters
    ----------
    - ctx: Context of the calibration process
    - β: Decay factor for the step size (0.5 <= β < 1)
    - verbose: Boolean, if True, print the optimization process

    Returns
    -------
    - Context: New context with the optimized kernel nuller
    - Dict: Dictionary with the optimization history (optional)
    """

    ctx = copy(ctx)
    ctx.Δh = ctx.interferometer.camera.e.to(u.hour).value * u.hourangle

    ψ = np.sqrt(ctx.pf.to(1/ctx.interferometer.camera.e.unit).value) * (1 + 0j) # Perfectly cophased inputs
    total_execpted_photons = np.sum(np.abs(ψ)**2)

    ε = 1e-6 * ctx.interferometer.λ.unit # Minimum shift step size

    # Shifters that contribute to redirecting light to the bright output
    φb = [1, 2, 3, 4, 5, 7]

    # Shifters that contribute to the symmetry of the dark outputs
    φk = [6, 8, 9, 10, 11, 12, 13, 14]

    # History of the optimization
    depth_history = []
    shifters_history = []

    Δφ = ctx.interferometer.λ / 4
    while Δφ > ε:

        if verbose:
            print(color.black(color.on_red(f"--- New iteration ---")), f"Δφ={Δφ:.2e}")

        for i in φb + φk:
            log = ""

            # Getting observation with different phase shifts
            ctx.interferometer.kn.φ[i-1] += Δφ
            _, k_pos, b_pos = ctx.observe()

            ctx.interferometer.kn.φ[i-1] -= 2*Δφ
            _, k_neg, b_neg = ctx.observe()

            ctx.interferometer.kn.φ[i-1] += Δφ
            _, k_old, b_old = ctx.observe()

            # Computing throughputs
            b_pos = b_pos / total_execpted_photons
            b_neg = b_neg / total_execpted_photons
            b_old = b_old / total_execpted_photons
            k_pos = np.sum(np.abs(k_pos)) / total_execpted_photons
            k_neg = np.sum(np.abs(k_neg)) / total_execpted_photons
            k_old = np.sum(np.abs(k_old)) / total_execpted_photons

            # Save the history
            depth_history.append(np.sum(k_old) / np.sum(b_old))
            shifters_history.append(np.copy(ctx.interferometer.kn.φ.value))

            # Maximize the bright metric for group 1 shifters
            if i in φb:
                log += "Shift " + color.black(color.on_lightgrey(f"{i}")) + " Bright: " + color.black(color.on_green(f"{b_neg:.2e} | {b_old:.2e} | {b_pos:.2e}")) + " -> "

                if b_pos > b_old and b_pos > b_neg:
                    log += color.black(color.on_green(" + "))
                    ctx.interferometer.kn.φ[i-1] += Δφ
                elif b_neg > b_old and b_neg > b_pos:
                    log += color.black(color.on_green(" - "))
                    ctx.interferometer.kn.φ[i-1] -= Δφ
                else:
                    log += color.black(color.on_green(" = "))

            # Minimize the kernel metric for group 2 shifters
            else:
                log += "Shift " + color.black(color.on_lightgrey(f"{i}")) + " Kernel: " + color.black(color.on_blue(f"{k_neg:.2e} | {k_old:.2e} | {k_pos:.2e}")) + " -> "

                if k_pos < k_old and k_pos < k_neg:
                    ctx.interferometer.kn.φ[i-1] += Δφ
                    log += color.black(color.on_blue(" + "))
                elif k_neg < k_old and k_neg < k_pos:
                    ctx.interferometer.kn.φ[i-1] -= Δφ
                    log += color.black(color.on_blue(" - "))
                else:
                    log += color.black(color.on_blue(" = "))
            
            if verbose:
                print(log)

        Δφ *= β

    ctx.interferometer.kn.φ = phase.bound(ctx.interferometer.kn.φ, ctx.interferometer.λ)

    if plot:

        shifters_history = np.array(shifters_history)

        _, axs = plt.subplots(2,1, figsize=figsize)

        axs[0].plot(depth_history)
        axs[0].set_xlabel("Iterations")
        axs[0].set_ylabel("Kernel-Null depth")
        axs[0].set_yscale("log")
        axs[0].set_title("Performance of the Kernel-Nuller")

        for i in range(shifters_history.shape[1]):
            axs[1].plot(shifters_history[:,i], label=f"Shifter {i+1}")
        axs[1].set_xlabel("Iterations")
        axs[1].set_ylabel("Phase shift")
        axs[1].set_yscale("linear")
        axs[1].set_title("Convergence of the phase shifters")
        axs[1].legend(loc='upper right')

        plt.show()

    if ret_history:
        return ctx, {
            "depth": depth_history,
            "shifters": shifters_history,
        }
    else:
        return ctx

#==============================================================================
# Obstruction method
#==============================================================================

def obstruction(
        ctx:Context,
        N: int = 1_000,
        plot: bool = False,
        figsize:tuple[int] = (30,20),
    ) -> Context:
    """
    Optimize the phase shifters offsets to maximize the nulling performance

    Parameters
    ----------
    - ctx: Context of the calibration process
    - λ: Wavelength of the observation
    - N: Number of points for the least squares optimization
    - plot: Boolean, if True, plot the optimization process

    Returns
    -------
    - Context: New context with the optimized kernel nuller
    """

    ctx = copy(ctx)
    kn = ctx.interferometer.kn
    λ = ctx.interferometer.λ
    e = ctx.interferometer.camera.e
    total_photons = np.sum(ctx.pf.to(1/e.unit).value) * e.value

    if plot:
        fig, axs = plt.subplots(3, 3, figsize=figsize)
        for i in range(7):
            axs.flatten()[i].set_xlabel("Phase shift")
            axs.flatten()[i].set_ylabel("Throughput")

    def maximize_bright(n, plt_coords=None):

        x = np.linspace(0, λ.value,N)
        y = np.empty(N)

        for i in range(N):
            kn.φ[n-1] = i * λ / N
            _, _, b = ctx.observe()
            y[i] = b / total_photons

        def sin(x, x0):
            return np.max(y) * (np.sin((x-x0)/λ.value*2*np.pi)+1)/2
        popt, pcov = curve_fit(sin, x, y, p0=[0])

        kn.φ[n-1] = (np.mod(popt[0]+λ.value/4, λ.value) * λ.unit).to(kn.φ.unit)

        if plot:
            axs[*plt_coords].set_title(f"$|B(\phi{n})|$")
            axs[*plt_coords].scatter(x, y, label='Data', color='tab:blue')
            axs[*plt_coords].plot(x, sin(x, *popt), label='Fit', color='tab:orange')
            axs[*plt_coords].axvline(x=np.mod(popt[0]+λ.value/4, λ.value), color='k', linestyle='--', label='Optimal phase shift')
            axs[*plt_coords].set_xlabel(f"Phase shift ({λ.unit})")
            axs[*plt_coords].set_ylabel("Bright throughput")
            axs[*plt_coords].legend()

    def minimize_kernel(n, m, plt_coords=None):

        x = np.linspace(0,λ.value,N)
        y = np.empty(N)

        for i in range(N):
            kn.φ[n-1] = i * λ / N
            _, k, _ = ctx.observe()
            y[i] = k[m-1] / total_photons

        def sin(x, x0):
            return np.max(y) * np.sin((x-x0)/λ.value*2*np.pi)/2
        popt, pcov = curve_fit(sin, x, y, p0=[0])

        kn.φ[n-1] = (np.mod(popt[0], λ.value) * λ.unit).to(kn.φ.unit)

        if plot:
            axs[*plt_coords].set_title(f"$K_{m}(\phi{n})$")
            axs[*plt_coords].scatter(x, y, label='Data', color='tab:blue')
            axs[*plt_coords].plot(x, sin(x, *popt), label='Fit', color='tab:orange')
            axs[*plt_coords].axvline(x=np.mod(popt[0], λ.value), color='k', linestyle='--', label='Optimal phase shift')
            axs[*plt_coords].set_xlabel(f"Phase shift ({λ.unit})")
            axs[*plt_coords].set_ylabel("Kernel throughput")
            axs[*plt_coords].legend()

    def maximize_darks(n, ds, plt_coords=None):

        x = np.linspace(0, λ.value, N)
        y = np.empty(N)
    
        for i in range(N):
            kn.φ[n-1] = i * λ / N
            d, _, _ = ctx.observe()
            y[i] = np.sum(np.abs(d[np.array(ds)-1])) / total_photons

        def sin(x, x0):
            return np.max(y) * (np.sin((x-x0)/λ.value*2*np.pi)+1)/2
        popt, pcov = curve_fit(sin, x, y, p0=[0], maxfev = 100_000)

        kn.φ[n-1] = (np.mod(popt[0]+λ.value/4, λ.value) * λ.unit).to(kn.φ.unit)

        if plot:
            axs[*plt_coords].set_title(f"$|D_{ds[0]}(\phi{n})| + |D_{ds[0]}(\phi{n})|$")
            axs[*plt_coords].scatter(x, y, label='Data', color='tab:blue')
            axs[*plt_coords].plot(x, sin(x, *popt), label='Fit', color='tab:orange')
            axs[*plt_coords].axvline(x=np.mod(popt[0]+λ.value/4, λ.value), color='k', linestyle='--', label='Optimal phase shift')
            axs[*plt_coords].set_xlabel(f"Phase shift ({λ.unit})")
            axs[*plt_coords].set_ylabel(f"Dark pair {ds} throughput")
            axs[*plt_coords].legend()

    # Bright maximization
    ctx.interferometer.kn.input_attenuation = [1, 1, 0, 0]
    maximize_bright(2, plt_coords=(0,0))

    ctx.interferometer.kn.input_attenuation = [0, 0, 1, 1]
    maximize_bright(4, plt_coords=(0,1))

    ctx.interferometer.kn.input_attenuation = [1, 0, 1, 0]
    maximize_bright(7, plt_coords=(0,2))

    # Darks maximization
    ctx.interferometer.kn.input_attenuation = [1, 0, -1, 0]
    maximize_darks(8, [1,2], plt_coords=(1,0))

    # Kernel minimization
    ctx.interferometer.kn.input_attenuation = [1, 0, 0, 0]
    minimize_kernel(11, 1, plt_coords=(2,0))
    minimize_kernel(13, 2, plt_coords=(2,1))
    minimize_kernel(14, 3, plt_coords=(2,2))

    kn.φ = phase.bound(kn.φ, λ)

    if plot:
        axs[1,1].axis('off')
        axs[1,2].axis('off')
        plt.show()

    return ctx
