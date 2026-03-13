"""Trial-and-error calibration for photonic interferometers.

This module provides genetic algorithm optimization for phase shifter tuning
in kernel nulling interferometers (SuperKN architecture).
"""

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from ..phase import bound


def calibrate_gen(
    ctx,
    β: float,
    verbose: bool = False,
    plot: bool = False,
    figsize: tuple = (10, 10),
    save_as=None,
) -> dict:
    """Optimize phase shifter offsets to maximize nulling performance.

    Uses a genetic algorithm (trial-and-error) approach to find optimal phase
    shifter values for maximizing bright output and minimizing kernel null depth
    in a SuperKN kernel nuller architecture.

    The optimization works on two groups of shifters:
    - Group 1 (φb): Shifters that redirect light to bright output
    - Group 2 (φk): Shifters that maintain symmetry of dark outputs

    Args:
        ctx: Context object with interferometer, target, and observation settings
        β (float): Decay factor for the step size (0.5 <= β < 1).
                   Controls convergence speed: smaller β converges faster.
        verbose (bool): If ``True``, print optimization progress for each iteration.
        plot (bool): If ``True``, plot the optimization process (depth and shifter convergence).
        figsize (tuple): Figure size for plots, default (10, 10).
        save_as (str): Path to save the plot if plot is True.

    Returns:
        dict: Dictionary with optimization history containing:
            - 'depth': array of kernel-null depth values across iterations
            - 'shifters': array of phase shifter values across iterations
    """

    ctx.Δh = ctx.interferometer.camera.e.to(u.hour).value * u.hourangle

    ψ = (
        np.sqrt(ctx.pf.to(1 / ctx.interferometer.camera.e.unit).value)
        * (1 + 0j)
    )  # Perfectly cophased inputs
    total_execpted_photons = np.sum(np.abs(ψ) ** 2)

    ε = 1e-6 * ctx.interferometer.λ.unit  # Minimum shift step size

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
            print(f"--- New iteration --- Δφ={Δφ:.2e}")

        for i in φb + φk:
            log = ""

            # Getting observation with different phase shifts
            ctx.interferometer.chip.φ[i - 1] += Δφ
            outs_pos = ctx.observe()

            ctx.interferometer.chip.φ[i - 1] -= 2 * Δφ
            outs_neg = ctx.observe()

            ctx.interferometer.chip.φ[i - 1] += Δφ
            outs_old = ctx.observe()

            b_pos = outs_pos[0]
            b_neg = outs_neg[0]
            b_old = outs_old[0]
            k_old = (
                1 / 3 * np.sum(np.abs(ctx.interferometer.chip.process_outputs(outs_old)))
            )
            k_pos = (
                1 / 3 * np.sum(np.abs(ctx.interferometer.chip.process_outputs(outs_pos)))
            )
            k_neg = (
                1 / 3 * np.sum(np.abs(ctx.interferometer.chip.process_outputs(outs_neg)))
            )

            # Save the history
            depth_history.append(k_old / b_old)
            shifters_history.append(np.copy(ctx.interferometer.chip.φ.value))

            # Maximize the bright metric for group 1 shifters
            if i in φb:
                log += f"Shift {i} Bright: {b_neg:.2e} | {b_old:.2e} | {b_pos:.2e} -> "

                if b_pos > b_old and b_pos > b_neg:
                    log += " + "
                    ctx.interferometer.chip.φ[i - 1] += Δφ
                elif b_neg > b_old and b_neg > b_pos:
                    log += " - "
                    ctx.interferometer.chip.φ[i - 1] -= Δφ
                else:
                    log += " = "

            # Minimize the kernel metric for group 2 shifters
            else:
                log += f"Shift {i} Kernel: {k_neg:.2e} | {k_old:.2e} | {k_pos:.2e} -> "

                if k_pos < k_old and k_pos < k_neg:
                    ctx.interferometer.chip.φ[i - 1] += Δφ
                    log += " + "
                elif k_neg < k_old and k_neg < k_pos:
                    ctx.interferometer.chip.φ[i - 1] -= Δφ
                    log += " - "
                else:
                    log += " = "

            if verbose:
                print(log)

        Δφ *= β

    ctx.interferometer.chip.φ = bound(ctx.interferometer.chip.φ, ctx.interferometer.λ)

    if plot:

        shifters_history = np.array(shifters_history)

        _, axs = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)

        axs[0].plot(depth_history)
        axs[0].set_xlabel("Iterations")
        axs[0].set_ylabel("Kernel-Null depth")
        axs[0].set_yscale("log")
        axs[0].set_title("Performance of the Kernel-Nuller")

        for i in range(shifters_history.shape[1]):
            axs[1].plot(shifters_history[:, i], label=f"Shifter {i+1}")
        axs[1].set_xlabel("Iterations")
        axs[1].set_ylabel("Phase shift")
        axs[1].set_yscale("linear")
        axs[1].set_title("Convergence of the phase shifters")
        # axs[1].legend(loc='upper right')

        if save_as:
            from ..utils import save_plot
            save_plot(save_as, "genetic_calibration.png")

    return {
        "depth": np.array(depth_history),
        "shifters": np.array(shifters_history),
    }
