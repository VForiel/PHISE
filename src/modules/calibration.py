import numpy as np
import astropy.units as u
from LRFutils import color
import matplotlib.pyplot as plt

from . import signals
from . import phase
from ..classes import kernel_nuller
from ..classes import KernelNuller

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
    - σ: Array of 14 intrasic OPD (in wavelenght unit)
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

    return phase.bound(kn.φ, λ=λ), {
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