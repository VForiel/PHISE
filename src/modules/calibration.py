import numpy as np
import astropy.units as u
from LRFutils import color
import matplotlib.pyplot as plt

from . import signals
from . import phase
from ..classes import kernel_nuller
from ..classes import KernelNuller

#==============================================================================
# Deterministic genetic algorithm
#==============================================================================

def genetic(
        kn: KernelNuller,
        β: float,
        λ: u.Quantity,
        f: u.Quantity,
        Δt: u.Quantity,
        verbose: bool = False,
    ) -> tuple[u.Quantity, dict[str, np.ndarray[float]]]:
    """
    Optimize the phase shifters offsets to maximize the nulling performance

    Parameters
    ----------
    - kn: Kernel nuller object
    - β: Decay factor for the step size (0.5 <= β < 1)
    - λ: Wavelength of the observation
    - f: Flux of the star
    - Δt: Integration time
    - verbose: Boolean, if True, print the optimization process

    Returns
    -------
    - Array of optimized phase shifters offsets
    - Dict containing the history of the optimization
    """

    ψ = np.ones(4) * (1 + 0j) * np.sqrt(f.to(1/Δt.unit).value/4) # Perfectly cophased inputs

    ε = 1e-6 * λ.unit # Minimum shift step size

    # Shifters that contribute to redirecting light to the bright output
    φb = [1, 2, 3, 4, 5, 7]

    # Shifters that contribute to the symmetry of the dark outputs
    φk = [6, 8, 9, 10, 11, 12, 13, 14]

    # History of the optimization
    bright_history = []
    kernel_history = []
    shifters_history = []

    Δφ = λ / 4
    while Δφ > ε:

        if verbose:
            print(color.black(color.on_red(f"--- New iteration ---")), f"Δφ={Δφ:.2e}")

        for i in φb + φk:
            log = ""

            # Step vector
            s = np.zeros(14) * λ.unit
            s[i-1] = Δφ

            # Apply the step
            _, k_neg, b_neg = kernel_nuller.observe_njit(ψ=ψ, φ=kn.φ-s, σ=kn.σ, λ=λ, Δt=Δt.value)
            _, k_old, b_old = kernel_nuller.observe_njit(ψ=ψ, φ=kn.φ,   σ=kn.σ, λ=λ, Δt=Δt.value)
            _, k_pos, b_pos = kernel_nuller.observe_njit(ψ=ψ, φ=kn.φ+s, σ=kn.σ, λ=λ, Δt=Δt.value)

            # Total Kernels relative intensity
            k_neg = np.sum(np.abs(k_neg))
            k_old = np.sum(np.abs(k_old))
            k_pos = np.sum(np.abs(k_pos))

            # Save the history
            bright_history.append(b_old / f.to(1/Δt.unit).value)
            kernel_history.append(k_old / f.to(1/Δt.unit).value)
            shifters_history.append(np.copy(kn.φ))

            # Maximize the bright metric for group 1 shifters
            if i in φb:
                log += "Shift " + color.black(color.on_lightgrey(f"{i}")) + " Bright: " + color.black(color.on_green(f"{b_neg:.2e} | {b_old:.2e} | {b_pos:.2e}")) + " -> "

                if b_pos > b_old and b_pos > b_neg:
                    log += color.black(color.on_green(" + "))
                    kn.φ += s
                elif b_neg > b_old and b_neg > b_pos:
                    log += color.black(color.on_green(" - "))
                    kn.φ -= s
                else:
                    log += color.black(color.on_green(" = "))

            # Minimize the kernel metric for group 2 shifters
            else:
                log += "Shift " + color.black(color.on_lightgrey(f"{i}")) + " Kernel: " + color.black(color.on_blue(f"{k_neg:.2e} | {k_old:.2e} | {k_pos:.2e}")) + " -> "

                if k_pos < k_old and k_pos < k_neg:
                    kn.φ += s
                    log += color.black(color.on_blue(" + "))
                elif k_neg < k_old and k_neg < k_pos:
                    kn.φ -= s
                    log += color.black(color.on_blue(" - "))
                else:
                    log += color.black(color.on_blue(" = "))
            
            if verbose:
                print(log)

        Δφ *= β

    kn.φ = phase.bound(kn.φ, λ)

    return {
        "bright": np.array(bright_history),
        "kernel": np.array(kernel_history),
        "shifters": np.array(shifters_history),
    }

def plot_genertic_history(history: dict[str, np.ndarray[float]]):
    bright_evol = history["bright"]
    kernel_evol = history["kernel"]
    shifts_evol = history["shifters"]

    _, axs = plt.subplots(3,1, figsize=(15,15))

    axs[0].plot(bright_evol)
    axs[0].set_xlabel("Number of iterations")
    axs[0].set_ylabel("Bright throughput (%)")
    axs[0].set_yscale("log")
    axs[0].set_title("Optimization of the bright output")

    axs[1].plot(kernel_evol)
    axs[1].set_xlabel("Number of iterations")
    axs[1].set_ylabel("Kernels throughput (%)")
    axs[1].set_yscale("log")
    axs[1].set_title("Optimization of the kernels")

    print(np.mean(kernel_evol[-2000:]))

    for i in range(shifts_evol.shape[1]):
        axs[2].plot(shifts_evol[:,i], label=f"Shifter {i+1}")
    axs[2].set_xlabel("Number of iterations")
    axs[2].set_ylabel("Phase shift")
    axs[2].set_yscale("linear")
    axs[2].set_title("Convergeance of the phase shifters")
    axs[2].legend(loc='upper right')

    plt.show()

#==============================================================================
# Input obstruction algorithm
#==============================================================================

def obstruction(
        kn: KernelNuller,
        λ: u.Quantity,
        f: u.Quantity,
        Δt: u.Quantity,
        verbose: bool = False,
    ) -> tuple[u.Quantity, dict[str, np.ndarray[float]]]:
    """
    Optimize the phase shifters offsets to maximize the nulling performance

    Parameters
    ----------
    - kn: KernelNuller object
    - λ: Wavelength of the observation
    - f: Flux of the star
    - Δt: Integration time
    - verbose: Boolean, if True, print the optimization process

    Returns
    -------
    - Dict containing the history of the optimization
    """

    b_history = []
    k_history = []
    φ_history = []

    def maximize_bright(kn:KernelNuller, ψ, n):
        Δφ = λ/4
        ε = 1e-6 * λ

        _, k, b_old = kn.observe(ψ=ψ, λ=λ, Δt=Δt)

        log = ""
        while Δφ > ε:
            s = np.zeros(14) * λ.unit
            s[n-1] = Δφ

            kn.φ += s
            _, _, b_pos = kn.observe(ψ=ψ, λ=λ, Δt=Δt)
            kn.φ -= 2*s
            _, _, b_neg = kn.observe(ψ=ψ, λ=λ, Δt=Δt)
            kn.φ += s
            
            log += "Shift " + color.black(color.on_lightgrey(f"{n}")) + " Bright: " + color.black(color.on_green(f"{b_neg:.2e} | {b_old:.2e} | {b_pos:.2e}")) + " -> "

            if b_pos > b_old and b_pos > b_neg:
                kn.φ += s
                log += color.black(color.on_green(" + "))
            elif b_neg > b_old and b_neg > b_pos:
                kn.φ -= s
                log += color.black(color.on_green(" - "))
            else:
                log += color.black(color.on_green(" = "))
            log += '\n'

            kn.φ = phase.bound(kn.φ, λ)

            _, k, b_old = kn.observe(ψ=ψ, λ=λ, Δt=Δt)

            φ_history.append(kn.φ.copy())
            b_history.append(b_old / f.to(1/Δt.unit).value)
            k_history.append(np.sum(np.abs(k)) / f.to(1/Δt.unit).value)
            Δφ /= 2
        
        if verbose:
            print(log)

    def minimize_kernel(kn, ψ, n, i):
        Δφ = λ/4
        ε = 1e-6 * λ
        i = i-1

        _, k, b = kn.observe(ψ=ψ, λ=λ, Δt=Δt)
        k_old = np.abs(k[i])

        log = ""
        while Δφ > ε:
            s = np.zeros(14) * λ.unit
            s[n-1] = Δφ

            kn.φ += s
            _, k_pos, _ = kn.observe(ψ=ψ, λ=λ, Δt=Δt)
            kn.φ -= 2*s
            _, k_neg, _ = kn.observe(ψ=ψ, λ=λ, Δt=Δt)
            kn.φ += s

            k_pos = np.abs(k_pos[i])
            k_neg = np.abs(k_neg[i])
            
            log += "Shift " + color.black(color.on_lightgrey(f"{n}")) + " Kernel: " + color.black(color.on_green(f"{k_neg:.2e} | {k_old:.2e} | {k_pos:.2e}")) + " -> "
            if k_pos < k_old and k_pos < k_neg:
                kn.φ += s
                log += color.black(color.on_green(" + "))
            elif k_neg < k_old and k_neg < k_pos:
                kn.φ -= s
                log += color.black(color.on_green(" - "))
            else:
                log += color.black(color.on_green(" = "))
            log += '\n'

            kn.φ = phase.bound(kn.φ, λ)

            _, k, b = kn.observe(ψ=ψ, λ=λ, Δt=Δt)
            k_old = np.abs(k[i])

            φ_history.append(kn.φ.copy())
            b_history.append(b / f.to(1/Δt.unit).value)
            k_history.append(np.sum(np.abs(k)) / f.to(1/Δt.unit).value)
            Δφ /= 2
        
        if verbose:
            print(log)

    def maximize_darks(kn, ψ, n, ds):
        Δφ = λ/4
        ε = 1e-6 * λ
        ds = np.array(ds) - 1

        d, k, b = kn.observe(ψ=ψ, λ=λ, Δt=Δt)
        d_old = np.sum(np.abs(d[ds]))

        log = ""
        while Δφ > ε:
            s = np.zeros(14) * λ.unit
            s[n-1] = Δφ
            
            kn.φ += s
            d_pos, _, _ = kn.observe(ψ=ψ, λ=λ, Δt=Δt)
            kn.φ -= 2*s
            d_neg, _, _ = kn.observe(ψ=ψ, λ=λ, Δt=Δt)
            kn.φ += s

            d_pos = np.sum(np.abs(d_pos[ds]))
            d_neg = np.sum(np.abs(d_neg[ds]))
            
            log += "Shift " + color.black(color.on_lightgrey(f"{n}")) + f" Darks: " + color.black(color.on_green(f"{d_neg:.2e} | {d_old:.2e} | {d_pos:.2e}")) + " -> "
            if d_pos > d_old and d_pos > d_neg:
                kn.φ += s
                log += color.black(color.on_green(" + "))
            elif d_neg > d_old and d_neg > d_pos:
                kn.φ -= s
                log += color.black(color.on_green(" - "))
            else:
                log += color.black(color.on_green(" = "))
            log += '\n'

            kn.φ = phase.bound(kn.φ, λ)

            d, k, b = kn.observe(ψ=ψ, λ=λ, Δt=Δt)
            d_old = np.sum(np.abs(d[ds]))

            φ_history.append(kn.φ.copy())
            b_history.append(b / f.to(1/Δt.unit).value)
            k_history.append(np.sum(np.abs(k)) / f.to(1/Δt.unit).value)
            Δφ /= 2
        
        if verbose:
            print(log)

    a = (1+0j) * np.sqrt(f/4)

    # Bright maximization
    ψ = np.array([a, a, 0, 0])
    maximize_bright(kn, ψ, 2)

    ψ = np.array([0, 0, a, a])
    maximize_bright(kn, ψ, 4)

    ψ = np.array([a, 0, a, 0])
    maximize_bright(kn, ψ, 7)

    # Darks maximization
    ψ = np.array([a, 0, -a, 0])
    maximize_darks(kn, ψ, 8, [1,2])

    # Kernel minimization
    ψ = np.array([a, 0, 0, 0])
    minimize_kernel(kn, ψ, 11, 1)
    minimize_kernel(kn, ψ, 13, 2)
    minimize_kernel(kn, ψ, 14, 3)

    kn.φ = phase.bound(kn.φ, λ)

    return {'bright':np.array(b_history), 'kernel':np.array(k_history), 'shifts':np.array(φ_history)}

def plot_obstruction_history(history: dict[str, np.ndarray[float]]):
    bright_evol, kernel_evol, shifts_evol = history["bright"], history["kernel"], history["shifts"]

    _, axs = plt.subplots(3, 1, figsize=(15,15))

    axs[0].plot(bright_evol)
    axs[0].set_xlabel("Number of iterations")
    axs[0].set_ylabel("Bright throughput (%)")
    axs[0].set_yscale("log")
    axs[0].set_title("Optimization of the bright output")

    axs[1].plot(kernel_evol)
    axs[1].set_xlabel("Number of iterations")
    axs[1].set_ylabel("Kernels throughput (%)")
    axs[1].set_yscale("log")
    axs[1].set_title("Optimization of the kernels")

    for i in range(shifts_evol.shape[1]):
        axs[2].plot(shifts_evol[:,i], label=f"Shifter {i+1}")
    axs[2].set_xlabel("Number of iterations")
    axs[2].set_ylabel("Phase shift")
    axs[2].set_yscale("linear")
    axs[2].set_title("Convergeance of the phase shifters")
    axs[2].legend(loc='upper right')

    plt.show()