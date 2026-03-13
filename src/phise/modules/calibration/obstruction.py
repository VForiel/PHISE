"""Obstruction calibration for photonic interferometers.

This module provides least-squares based optimization for phase shifter tuning
in kernel nulling interferometers, accounting for partial obstructions and
beam combiner non-idealities.
"""

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import scipy.optimize

from ..phase import bound


def calibrate_obs(
    ctx,
    n: int = 1_000,
    plot: bool = False,
    figsize: tuple = (30, 20),
    save_as=None,
):
    """Optimize calibration via least squares sampling.

    Uses least-squares fitting to find optimal phase shifter values that maximize
    bright output and minimize kernel null depth in a SuperKN kernel nuller.
    This approach accounts for obstructions and non-ideal beam combiner behavior.

    The optimization:
    1. Bright maximization: Adjusts shifters 2, 4, 7 to maximize bright output
    2. Dark maximization: Adjusts shifter 8 to balance dark outputs
    3. Kernel minimization: Adjusts shifters 11, 13, 14 to minimize kernel null depth
    4. Global optimization: For non-monochromatic light, refines paired shifters

    Args:
        ctx: Context object with interferometer, target, and observation settings
        n (int): Number of sampling points for least squares fitting.
                 Higher values provide better fits but take longer.
        plot (bool): If ``True``, plot the optimization process for each shifter.
        figsize (tuple): Figure size for plots, default (30, 20).
        save_as (str): Path to save the plot if plot is True.

    Returns:
        None: Modifies ctx.interferometer.chip.φ in-place
    """

    chip = ctx.interferometer.chip
    input_attenuation_backup = chip.input_attenuation.copy()
    λ = ctx.interferometer.λ
    e = ctx.interferometer.camera.e
    total_photons = np.sum(ctx.pf.to(1 / e.unit).value) * e.value

    if plot:
        _, axs = plt.subplots(6, 3, figsize=figsize, constrained_layout=True)
        for i in range(7):
            axs.flatten()[i].set_xlabel("Phase shift")
            axs.flatten()[i].set_ylabel("Throughput")

    def maximize_bright(p, plt_coords=None):

        x = np.linspace(0, λ.value, n)
        y = np.empty(n)

        if isinstance(p, list):
            Δp = (chip.φ[p[1] - 1] - chip.φ[p[0] - 1]) % λ

        for i in range(n):

            if isinstance(p, list):
                chip.φ[p[0] - 1] = x[i] * λ.unit
                chip.φ[p[1] - 1] = (chip.φ[p[0] - 1] + Δp) % λ
            else:
                chip.φ[p - 1] = x[i] * λ.unit

            outs = ctx.observe()
            y[i] = outs[0] / total_photons

        def sin(x, x0):
            return (
                (np.sin((x - x0) / λ.value * 2 * np.pi) + 1) / 2 * (np.max(y) - np.min(y))
                + np.min(y)
            )

        # Fit sin using scipy.optimize.minimize
        popt = scipy.optimize.minimize(
            lambda x0: np.sum((y - sin(x, x0)) ** 2), x0=[0], method="Nelder-Mead"
        ).x

        if isinstance(p, list):
            chip.φ[p[0] - 1] = (
                np.mod(popt[0] + λ.value / 4, λ.value) * λ.unit
            ).to(chip.φ.unit)
            chip.φ[p[1] - 1] = (chip.φ[p[0] - 1] + Δp) % λ
        else:
            chip.φ[p - 1] = (
                np.mod(popt[0] + λ.value / 4, λ.value) * λ.unit
            ).to(chip.φ.unit)

        if plot:
            axs[plt_coords].set_title(rf"$|B(\phi{p})|$")
            axs[plt_coords].scatter(x, y, label="Data", color="tab:blue")
            axs[plt_coords].plot(x, sin(x, *popt), label="Fit", color="tab:orange")
            axs[plt_coords].axvline(
                x=np.mod(popt[0] + λ.value / 4, λ.value),
                color="k",
                linestyle="--",
                label="Optimal phase shift",
            )
            axs[plt_coords].set_xlabel(f"Phase shift ({λ.unit})")
            axs[plt_coords].set_ylabel("Bright throughput")
            axs[plt_coords].legend()

    def minimize_kernel(p, m, plt_coords=None):

        x = np.linspace(0, λ.value, n)
        y = np.empty(n)

        for i in range(n):
            chip.φ[p - 1] = x[i] * λ.unit
            outs = ctx.observe()
            ker = ctx.interferometer.chip.process_outputs(outs)
            y[i] = ker[m - 1]

        def sin(x, x0):
            return (
                (np.sin((x - x0) / λ.value * 2 * np.pi) + 1) / 2 * (np.max(y) - np.min(y))
                + np.min(y)
            )

        # Fit sin using scipy.optimize.minimize
        popt = scipy.optimize.minimize(
            lambda x0: np.sum((y - sin(x, x0)) ** 2), x0=[0], method="Nelder-Mead"
        ).x

        chip.φ[p - 1] = (np.mod(popt[0], λ.value) * λ.unit).to(chip.φ.unit)

        if plot:
            axs[plt_coords].set_title(rf"$K_{m}(\phi{p})$")
            axs[plt_coords].scatter(x, y, label="Data", color="tab:blue")
            axs[plt_coords].plot(x, sin(x, *popt), label="Fit", color="tab:orange")
            axs[plt_coords].axvline(
                x=np.mod(popt[0], λ.value),
                color="k",
                linestyle="--",
                label="Optimal phase shift",
            )
            axs[plt_coords].set_xlabel(f"Phase shift ({λ.unit})")
            axs[plt_coords].set_ylabel("Kernel throughput")
            axs[plt_coords].legend()

    def maximize_darks(p, ds, plt_coords=None):

        # Init data arrays
        x = np.linspace(0, λ.value, n)
        y = np.empty(n)

        # Sampling
        for i in range(n):
            # Set phase shift
            chip.φ[p - 1] = x[i] * λ.unit
            # Get outputs intensities
            outs = ctx.observe()
            # Compute |Di|² + |Dj|²
            y[i] = np.sum(np.abs(outs[np.array(ds)]))

        # Model
        def sin(x, x0):
            return (
                (np.sin((x - x0) / λ.value * 2 * np.pi) + 1) / 2 * (np.max(y) - np.min(y))
                + np.min(y)
            )

        # Fit sin using scipy.optimize.minimize
        popt = scipy.optimize.minimize(
            lambda x0: np.sum((y - sin(x, x0)) ** 2), x0=[0], method="Nelder-Mead"
        ).x

        # Update phase shift
        chip.φ[p - 1] = (np.mod(popt[0] + λ.value / 4, λ.value) * λ.unit).to(
            chip.φ.unit
        )

        # Plotting
        if plot:
            axs[plt_coords].set_title(rf"$|D_{ds[0]}(\phi{p})| + |D_{ds[1]}(\phi{p})|$")
            axs[plt_coords].scatter(x, y, label="Data", color="tab:blue")
            axs[plt_coords].plot(x, sin(x, *popt), label="Fit", color="tab:orange")
            axs[plt_coords].axvline(
                x=np.mod(popt[0] + λ.value / 4, λ.value),
                color="k",
                linestyle="--",
                label="Optimal phase shift",
            )
            axs[plt_coords].set_xlabel(f"Phase shift ({λ.unit})")
            axs[plt_coords].set_ylabel(f"Dark pair {ds} throughput")
            axs[plt_coords].legend()

    # Bright maximization
    ctx.interferometer.chip.input_attenuation = [1, 1, 0, 0]
    maximize_bright(2, plt_coords=(0, 0))

    ctx.interferometer.chip.input_attenuation = [0, 0, 1, 1]
    maximize_bright(4, plt_coords=(0, 1))

    ctx.interferometer.chip.input_attenuation = [1, 0, 1, 0]
    maximize_bright(7, plt_coords=(0, 2))

    # Darks maximization
    ctx.interferometer.chip.input_attenuation = [1, 0, 0, -1]
    maximize_darks(8, [1, 2], plt_coords=(1, 0))

    # Kernel minimization
    ctx.interferometer.chip.input_attenuation = [1, 0, 0, 0]
    minimize_kernel(11, 1, plt_coords=(2, 0))
    minimize_kernel(13, 2, plt_coords=(2, 1))
    minimize_kernel(14, 3, plt_coords=(2, 2))

    # Find global minimum
    if not ctx.monochromatic:
        # Bright maximization
        ctx.interferometer.chip.input_attenuation = [1, 1, 0, 0]
        maximize_bright([1, 2], plt_coords=(3, 0))
        ctx.interferometer.chip.input_attenuation = [0, 0, 1, 1]
        maximize_bright([3, 4], plt_coords=(3, 1))
        ctx.interferometer.chip.input_attenuation = [1, 0, 1, 0]
        maximize_bright([5, 7], plt_coords=(3, 2))

    chip.φ = bound(chip.φ, λ)
    chip.input_attenuation = input_attenuation_backup

    if plot:
        axs[1, 1].axis("off")
        axs[1, 2].axis("off")

        if save_as:
            from ..utils import save_plot
            save_plot(save_as, "obstruction_calibration.png")
        plt.show()
