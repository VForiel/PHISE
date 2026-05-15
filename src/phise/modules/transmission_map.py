"""Transmission map computation and visualisation for PHISE contexts.

This module provides standalone functions for computing and plotting the
transmission maps of a photonic kernel nuller.  All high-level functions
accept a :class:`~phise.classes.context.Context` as their first argument so
that they can be used independently of the class interface while the
:class:`~phise.classes.context.Context` retains thin wrapper methods for
backward compatibility.

Typical usage::

    import phise
    from phise.modules import transmission_map as tm

    ctx = phise.examples.contexts.VLTI()
    raw, proc = tm.get_transmission_maps(ctx, N=100)
    tm.plot_transmission_maps(ctx, N=100)
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
import numba as nb
import astropy.units as u
import matplotlib.pyplot as plt

from . import coordinates
from . import utils
from ..classes.archs import superkn as _superkn

# Avoid circular imports: Context is imported only for type checking.
if TYPE_CHECKING:
    from ..classes.context import Context


# =============================================================================
# Low-level JIT helpers
# =============================================================================

@nb.njit()
def get_unique_source_input_fields_jit(
    a: np.ndarray,
    ρ: float,
    θ: float,
    λ: float,
    p: np.ndarray,
) -> np.ndarray:
    """Compute complex amplitudes of input fields for a single point source.

    For each telescope *i* the complex amplitude is

    .. math::

        s_i = \\sqrt{a_i} \\, \\exp\\!\\left(\\frac{2\\pi i}{\\lambda}
              \\, p_{i,\\text{rot}} \\sin\\rho \\right)

    where :math:`p_{i,\\text{rot}}` is the projected baseline component
    after rotating by the parallactic angle :math:`-\\theta`.

    Parameters
    ----------
    a : np.ndarray, shape (n_telescopes,)
        Intensity (photons/s) collected by each telescope.
    ρ : float
        Angular separation of the source [rad].
    θ : float
        Parallactic angle of the source [rad].
    λ : float
        Observation wavelength [m].
    p : np.ndarray, shape (n_telescopes, 2)
        Projected telescope positions [m].

    Returns
    -------
    np.ndarray of complex128, shape (n_telescopes,)
        Complex field amplitudes.
    """

    assert len(a) == len(p), (
        "Length of amplitudes (len(a)) must equal the number of telescopes (len(p))."
    )

    s = np.empty(p.shape[0], dtype=np.complex128)

    for i, t in enumerate(p):
        # Rotate projected positions by the parallactic angle
        p_rot = t[0] * np.cos(-θ) - t[1] * np.sin(-θ)

        # Geometric phase delay from source position
        Φ = 2 * np.pi * p_rot * np.sin(ρ) / λ

        s[i] = np.exp(1j * Φ)

    return s * np.sqrt(a)


@nb.njit()
def get_transmission_map_jit(
    N: int,
    φ: np.ndarray,
    σ: np.ndarray,
    p: np.ndarray,
    λ: float,
    λ0: float,
    fov: float,
    output_order: np.ndarray,
    nb_raw_outputs: int,
    nb_processed_outputs: int,
    get_output_fields_fn,
    process_outputs_fn,
) -> tuple:
    """Compute raw and processed transmission maps over the field of view.

    For every pixel of an *N×N* grid covering the full field of view, the
    function evaluates the chip response to a unit-amplitude point source
    located at the corresponding sky position and assembles the result into
    two 3-D arrays.

    Parameters
    ----------
    N : int
        Number of pixels along each axis.
    φ : np.ndarray
        Injected OPD per telescope [m].
    σ : np.ndarray
        Intrinsic (manufacturing) OPD per telescope [m].
    p : np.ndarray, shape (n_telescopes, 2)
        Projected telescope positions [m].
    λ : float
        Observation wavelength [m].
    λ0 : float
        Chip reference (design) wavelength [m].
    fov : float
        Field of view [mas].
    output_order : np.ndarray of int
        Permutation array that reorders chip outputs.
    nb_raw_outputs : int
        Number of raw (physical) chip outputs.
    nb_processed_outputs : int
        Number of processed (kernel) outputs.
    get_output_fields_fn : callable (njit)
        Chip-specific JIT function ``(ψ, φ, σ, λ, λ0, output_order) →
        raw_output_fields``.
    process_outputs_fn : callable (njit)
        Chip-specific JIT function ``(raw_outs) → processed_outputs``.

    Returns
    -------
    raw_out_maps : np.ndarray, shape (nb_raw_outputs, N, N)
        Transmission maps for each raw output channel.
    processed_out_maps : np.ndarray, shape (nb_processed_outputs, N, N)
        Transmission maps for each processed (kernel) output.
    """

    _, _, θ_map, ρ_map = coordinates.get_maps_jit(N=N, fov=fov)

    # Convert angular separation from mas to radians
    ρ_map = ρ_map / 1000 / 3600 / 180 * np.pi

    raw_out_maps = np.empty((nb_raw_outputs, N, N))
    processed_out_maps = np.empty((nb_processed_outputs, N, N))

    for x in range(N):
        for y in range(N):
            θ = θ_map[x, y]
            ρ = ρ_map[x, y]

            ψ = get_unique_source_input_fields_jit(
                a=np.ones(4) / 4, ρ=ρ, θ=θ, λ=λ, p=p
            )
            raw_outs = np.abs(get_output_fields_fn(ψ, φ, σ, λ, λ0, output_order)) ** 2

            for i in range(nb_raw_outputs):
                raw_out_maps[i, x, y] = raw_outs[i]

            if nb_processed_outputs > 0:
                processed_outs = process_outputs_fn(raw_outs)
                for i in range(nb_processed_outputs):
                    processed_out_maps[i, x, y] = processed_outs[i]

    return raw_out_maps, processed_out_maps


@nb.njit()
def get_analytical_transmission_map_jit(
    N: int,
    p: np.ndarray,
    λ: float,
    fov: float,
) -> tuple:
    """Compute analytical transmission maps for a 4-telescope kernel nuller.

    Uses simplified closed-form expressions for the SuperKN architecture.
    This function provides insight into the fundamental transmission patterns
    and is faster than the full numerical simulation.

    Physics Background:
        The kernel nuller combines light from 4 telescopes through MMI beam
        combiners.  For an ideal chip at the design wavelength:

        1. **Bright output** — all inputs combined constructively.
        2. **Dark outputs** — inputs combined with π/2 phase shifts creating
           paired outputs.
        3. **Kernel outputs** — ``K_n = |D_{2n-1}|² - |D_{2n}|²``, robust
           to symmetric phase aberrations.

    Parameters
    ----------
    N : int
        Number of pixels along each axis (N×N map).
    p : np.ndarray, shape (4, 2)
        Projected telescope positions [m].
    λ : float
        Observation wavelength [m].
    fov : float
        Field of view [mas].

    Returns
    -------
    bright_map : np.ndarray, shape (N, N)
        Bright output transmission map.
    dark_maps : np.ndarray, shape (6, N, N)
        Dark output transmission maps.
    kernel_maps : np.ndarray, shape (3, N, N)
        Kernel output transmission maps.
    """
    π = np.pi

    _, _, θ_map, ρ_map = coordinates.get_maps_jit(N=N, fov=fov)

    # Convert mas → radians
    ρ_map = ρ_map / 1000 / 3600 / 180 * π

    bright_map = np.empty((N, N))
    dark_maps = np.empty((6, N, N))
    kernel_maps = np.empty((3, N, N))

    for x in range(N):
        for y in range(N):
            θ = θ_map[x, y]   # Position angle [rad]
            ρ = ρ_map[x, y]   # Angular separation [rad]

            # Input phases from source position (rotated projection)
            φ = np.empty(4)
            for i in range(4):
                p_rot = p[i, 0] * np.cos(-θ) - p[i, 1] * np.sin(-θ)
                φ[i] = 2 * π * p_rot * np.sin(ρ) / λ

            # Amplitude 0.5 → matches SuperKN normalisation convention
            ψ = np.empty(4, dtype=np.complex128)
            for i in range(4):
                ψ[i] = 0.5 * np.exp(1j * φ[i])

            b, d, k = _superkn.expected_outputs_jit(ψ)

            bright_map[x, y] = b
            dark_maps[:, x, y] = d
            kernel_maps[:, x, y] = k

    return bright_map, dark_maps, kernel_maps


# =============================================================================
# High-level public API
# =============================================================================

def get_transmission_maps(ctx: "Context", N: int) -> tuple:
    """Generate all kernel nuller transmission maps for a context.

    Parameters
    ----------
    ctx : Context
        Observation context providing chip, interferometer and projected
        telescope positions.
    N : int
        Map resolution (N×N pixels).

    Returns
    -------
    raw_out_maps : np.ndarray, shape (nb_raw_outputs, N, N)
        Transmission maps for each raw output channel.
    processed_out_maps : np.ndarray, shape (nb_processed_outputs, N, N)
        Transmission maps for each processed (kernel) output.
    """

    return get_transmission_map_jit(
        N=N,
        φ=ctx.chip.φ.to(u.m).value,
        σ=ctx.chip.σ.to(u.m).value,
        p=ctx.p.value,
        λ=ctx.interferometer.λ.to(u.m).value,
        λ0=ctx.chip.λ0.to(u.m).value,
        fov=ctx.interferometer.fov.to(u.mas).value,
        output_order=ctx.chip.output_order,
        nb_raw_outputs=ctx.chip.nb_raw_outputs,
        nb_processed_outputs=ctx.chip.nb_processed_outputs,
        get_output_fields_fn=ctx.chip.get_output_fields_jit,
        process_outputs_fn=ctx.chip.process_outputs_jit,
    )


def get_transmission_map_gradient_norm(ctx: "Context", N: int) -> tuple:
    """Compute the spatial gradient norm of each transmission map.

    The gradient highlights regions of rapid transmission variation, which
    are informative for companion detection sensitivity analysis.

    Parameters
    ----------
    ctx : Context
        Observation context.
    N : int
        Map resolution (N×N pixels).

    Returns
    -------
    raw_grad_maps : np.ndarray, shape (nb_raw_outputs, N, N)
        Gradient-norm maps for each raw output.
    processed_grad_maps : np.ndarray, shape (nb_processed_outputs, N, N)
        Gradient-norm maps for each processed output.
    """

    raw_out_maps, processed_out_maps = get_transmission_maps(ctx, N=N)

    raw_grad_maps = np.empty_like(raw_out_maps)
    processed_grad_maps = np.empty_like(processed_out_maps)

    for i in range(ctx.interferometer.chip.nb_raw_outputs):
        dnx, dny = np.gradient(raw_out_maps[i])
        raw_grad_maps[i] = np.sqrt(dnx ** 2 + dny ** 2)

    for i in range(ctx.interferometer.chip.nb_processed_outputs):
        ddx, ddy = np.gradient(processed_out_maps[i])
        processed_grad_maps[i] = np.sqrt(ddx ** 2 + ddy ** 2)

    return raw_grad_maps, processed_grad_maps


def plot_transmission_maps(
    ctx: "Context",
    N: int = 100,
    return_plot: bool = False,
    grad: bool = False,
    save_as=None,
):
    """Plot raw and processed output transmission maps side by side.

    A two-row grid is produced: raw outputs on the first row, processed
    (kernel) outputs on the second.  Star and companion positions are
    overlaid on each panel.  A throughput summary is printed (or returned
    when *return_plot* is ``True``).

    Parameters
    ----------
    ctx : Context
        Observation context to visualise.
    N : int, optional
        Map resolution.  Defaults to 100.
    return_plot : bool, optional
        When ``True`` the function returns ``(png_bytes, transmissions_str)``
        instead of displaying the figure.
    grad : bool, optional
        When ``True`` plot gradient-norm maps instead of raw transmission.
    save_as : str or None, optional
        Path prefix passed to :func:`phise.modules.utils.save_plot`.

    Returns
    -------
    (bytes, str) or None
        PNG image bytes and throughput summary string when *return_plot* is
        ``True``; ``None`` otherwise.
    """
    from ..classes.companion import Companion

    # Compute maps (gradient-norm or plain transmission)
    if grad:
        raw_out_maps, processed_out_maps = get_transmission_map_gradient_norm(ctx, N=N)
    else:
        raw_out_maps, processed_out_maps = get_transmission_maps(ctx, N=N)

    # Companion positions in angular units for scatter overlay
    companions_pos = []
    for c in ctx.target.companions:
        x, y = coordinates.ρθ_to_xy(ρ=c.ρ, θ=c.θ, fov=ctx.interferometer.fov)
        companions_pos.append((x * ctx.interferometer.fov / 2, y * ctx.interferometer.fov / 2))

    nb_raw_outs = ctx.interferometer.chip.nb_raw_outputs
    nb_processed_outs = ctx.interferometer.chip.nb_processed_outputs
    nb_columns = max(nb_raw_outs, nb_processed_outs)
    _, axs = plt.subplots(2, nb_columns, figsize=(5 * nb_columns, 10))

    fov = ctx.interferometer.fov
    extent = (-fov.value / 2, fov.value / 2, -fov.value / 2, fov.value / 2)

    for i in range(nb_columns):

        if i >= nb_raw_outs:
            axs[0, i].axis("off")
        else:
            im = axs[0, i].imshow(
                raw_out_maps[i],
                aspect="equal",
                cmap="hot" if not grad else "gray",
                extent=extent,
            )
            axs[0, i].set_title(ctx.interferometer.chip._raw_output_labels[i])
            plt.colorbar(im, ax=axs[0, i])
            axs[0, i].set_xlabel(r"$\theta_x$" + f" ({fov.unit})")
            axs[0, i].set_ylabel(r"$\theta_y$" + f" ({fov.unit})")
            axs[0, i].scatter(0, 0, color="yellow", marker="*", edgecolors="black", s=100)
            for x, y in companions_pos:
                axs[0, i].scatter(x, y, color="blue", edgecolors="black")

        if i >= nb_processed_outs:
            axs[1, i].axis("off")
        else:
            im = axs[1, i].imshow(
                processed_out_maps[i],
                aspect="equal",
                cmap="bwr" if not grad else "gray",
                extent=extent,
            )
            axs[1, i].set_title(ctx.interferometer.chip._processed_output_labels[i])
            axs[1, i].set_aspect("equal")
            plt.colorbar(im, ax=axs[1, i])
            axs[1, i].set_xlabel(r"$\theta_x$" + f" ({fov.unit})")
            axs[1, i].set_ylabel(r"$\theta_y$" + f" ({fov.unit})")
            axs[1, i].scatter(0, 0, color="yellow", marker="*", edgecolors="black", s=100)
            for x, y in companions_pos:
                axs[1, i].scatter(x, y, color="blue", edgecolors="black")

    # Build throughput summary string
    transmissions = ""
    companions = (
        [Companion(name=ctx.target.name + " Star", c=1, θ=0 * u.deg, ρ=0 * u.mas)]
        + ctx.target.companions
    )
    for c in companions:
        θ = c.θ.to(u.rad)
        ρ = c.ρ.to(u.rad)
        p = ctx.p.to(u.m)
        λ = ctx.interferometer.λ.to(u.m)

        ψ = get_unique_source_input_fields_jit(
            a=np.ones(4) / 4,
            ρ=ρ.value,
            θ=θ.value,
            λ=λ.value,
            p=p.value,
        )

        out_fields = ctx.interferometer.chip.get_output_fields(ψ=ψ, λ=ctx.interferometer.λ)
        raw_outs = np.abs(out_fields) ** 2
        processed_outs = ctx.interferometer.chip.process_outputs(raw_outs)

        linebreak = "<br>" if return_plot else "\n   "
        transmissions += "<h2>" if return_plot else ""
        transmissions += f"\n{c.name} throughputs:"
        transmissions += "</h1>" if return_plot else "\n----------" + linebreak
        transmissions += (
            ",   ".join(
                [
                    f"{ctx.interferometer.chip._raw_output_labels[o]}: {raw_outs[o]*100:.2f}%"
                    for o in range(nb_raw_outs)
                ]
            )
            + linebreak
        )
        transmissions += ",   ".join(
            [
                f"{ctx.interferometer.chip._processed_output_labels[o]}: {processed_outs[o]*100:.2f}%"
                for o in range(nb_processed_outs)
            ]
        )

    if save_as:
        utils.save_plot(save_as, "transmission_maps.png")

    if return_plot:
        plot = BytesIO()
        plt.savefig(plot, format="png")
        plt.close()
        return plot.getvalue(), transmissions

    plt.show()
    print(transmissions)


def plot_analytical_transmission_maps(
    ctx: "Context",
    N: int,
    return_plot: bool = False,
    save_as=None,
):
    """Plot analytical transmission maps (Bright + 6 Darks + 3 Kernels).

    Uses the closed-form SuperKN model.  A 2×5 grid is produced:

    * Row 1: Bright, D1, D2, D3, D4
    * Row 2: D5, D6, K1, K2, K3

    Parameters
    ----------
    ctx : Context
        Observation context.
    N : int
        Map resolution (N×N pixels).
    return_plot : bool, optional
        When ``True`` return ``(png_bytes, transmissions_str)`` instead of
        showing the figure.
    save_as : str or None, optional
        Path prefix for :func:`phise.modules.utils.save_plot`.

    Returns
    -------
    (bytes, str) or None
        PNG image bytes and throughput summary string when *return_plot* is
        ``True``; ``None`` otherwise.
    """
    from ..classes.companion import Companion

    p = ctx.p.value
    λ = ctx.interferometer.λ.to(u.m).value
    fov = ctx.interferometer.fov

    bright_map, dark_maps, kernel_maps = get_analytical_transmission_map_jit(
        N=N, p=p, λ=λ, fov=fov
    )

    # Companion overlay positions
    companions_pos = []
    for c in ctx.target.companions:
        x, y = coordinates.ρθ_to_xy(ρ=c.ρ, θ=c.θ, fov=ctx.interferometer.fov)
        companions_pos.append((x * ctx.interferometer.fov / 2, y * ctx.interferometer.fov / 2))

    _, axs = plt.subplots(2, 5, figsize=(25, 10))

    fov_val = fov.value
    extent = (-fov_val / 2, fov_val / 2, -fov_val / 2, fov_val / 2)

    labels = [
        "Bright",
        "Dark 1", "Dark 2", "Dark 3", "Dark 4",
        "Dark 5", "Dark 6",
        "Kernel 1", "Kernel 2", "Kernel 3",
    ]
    maps = [
        bright_map,
        dark_maps[0], dark_maps[1], dark_maps[2], dark_maps[3],
        dark_maps[4], dark_maps[5],
        kernel_maps[0], kernel_maps[1], kernel_maps[2],
    ]
    cmaps = ["hot"] + ["gray"] * 6 + ["bwr"] * 3

    for i in range(10):
        row = i // 5
        col = i % 5
        ax = axs[row, col]

        im = ax.imshow(maps[i], aspect="equal", cmap=cmaps[i], extent=extent)
        ax.set_title(labels[i])
        plt.colorbar(im, ax=ax)
        ax.set_xlabel(r"$\theta_x$" + f" ({fov.unit})")
        ax.set_ylabel(r"$\theta_y$" + f" ({fov.unit})")
        ax.scatter(0, 0, color="yellow", marker="*", edgecolors="black", s=100)
        for x, y in companions_pos:
            ax.scatter(x, y, color="blue", edgecolors="black")

    # Throughput summary
    transmissions = ""
    companions = (
        [Companion(name=ctx.target.name + " Star", c=1, θ=0 * u.deg, ρ=0 * u.mas)]
        + ctx.target.companions
    )

    for c in companions:
        θ = c.θ.to(u.rad).value
        ρ = c.ρ.to(u.rad).value

        φ = np.empty(4)
        for i in range(4):
            p_rot = p[i, 0] * np.cos(-θ) - p[i, 1] * np.sin(-θ)
            φ[i] = 2 * np.pi * p_rot * np.sin(ρ) / λ

        ψ = np.empty(4, dtype=np.complex128)
        for i in range(4):
            ψ[i] = 0.5 * np.exp(1j * φ[i])

        b, d, k = _superkn.expected_outputs_jit(ψ)

        linebreak = "<br>" if return_plot else "\n   "
        transmissions += "<h2>" if return_plot else ""
        transmissions += f"\n{c.name} analytical throughputs:"
        transmissions += "</h1>" if return_plot else "\n----------" + linebreak

        transmissions += f"Bright: {b*100:.2f}%" + linebreak
        transmissions += (
            ", ".join([f"D{i+1}: {val*100:.2f}%" for i, val in enumerate(d)]) + linebreak
        )
        transmissions += ", ".join([f"K{i+1}: {val*100:.2f}%" for i, val in enumerate(k)]) + linebreak

    if save_as:
        utils.save_plot(save_as, "analytical_transmission_maps.png")

    if return_plot:
        plot = BytesIO()
        plt.savefig(plot, format="png")
        plt.close()
        return plot.getvalue(), transmissions

    plt.show()
    print(transmissions)
