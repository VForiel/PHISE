"""Coordinate utilities for (u, v) maps, polar angle and separation.

This module provides utilities to generate normalized spatial coordinate
maps (u, v), the polar angle α and angular separation θ over a given field
of view, plus conversion helpers between (α, θ) and (u, v).

Docstrings follow the Google style (compatible with Sphinx Napoleon).
"""
import numpy as np
import numba as nb
import astropy.units as u
import matplotlib.pyplot as plt
try:
    plt.rcParams['image.origin'] = 'lower'
except Exception:
    # Some mocked matplotlib backends or minimal stubs may not accept assignments
    pass

@nb.njit()
def get_maps_njit(N: int, fov: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate (u, v), α and θ maps in polar representation (njit version).

    Args:
        N: Grid resolution (samples per axis).
        fov: Total field of view (same unit as θ; scalar value, e.g. in mas or rad).

    Returns:
        Tuple with:
        - x_map (np.ndarray): Normalized u map of shape (N, N) in [-1, 1].
        - y_map (np.ndarray): Normalized v map of shape (N, N) in [-1, 1].
        - alpha (np.ndarray): Polar angle α map in radians, shape (N, N).
        - theta (np.ndarray): Separation θ map, shape (N, N), expressed in ``fov`` units.
    """
    x_map = np.zeros((N, N))
    y_map = np.zeros((N, N))
    for (i, v) in enumerate(np.linspace(-1, 1, N)):
        x_map[:, i] = v
        y_map[i, :] = v
    θ_map = np.sqrt(x_map ** 2 + y_map ** 2) * fov / 2
    α_map = np.arctan2(y_map, x_map) % (2 * np.pi)
    return x_map, y_map, α_map, θ_map

def get_maps(N: int, fov: u.Quantity) -> tuple[np.ndarray, np.ndarray, u.Quantity, u.Quantity]:
    """Generate (u, v), α and θ maps with units handling.

    This interface converts and annotates outputs with ``astropy`` units.

    Args:
        N: Grid resolution (samples per axis).
        fov: Total field of view as an ``astropy.units.Quantity`` (e.g. mas, rad).

    Returns:
        Tuple with:
        - x_map (np.ndarray): Normalized u map of shape (N, N) in [-1, 1].
        - y_map (np.ndarray): Normalized v map of shape (N, N) in [-1, 1].
        - alpha (Quantity): α map in radians (``u.rad``), shape (N, N).
        - theta (Quantity): θ map, shape (N, N), in ``fov`` units.
    """
    (x_map, y_map, α_map, θ_map) = get_maps_njit(N=N, fov=fov.value)
    α_map *= u.rad
    θ_map *= fov.unit
    return x_map, y_map, α_map, θ_map

def plot_uv_map(extent: tuple[float, float, float, float]):
    """Display u, v, θ and α maps for a quick visual inspection.

    Note:
        This function internally generates the maps via ``get_maps`` and
        displays them in a 2x2 figure. The ``extent`` argument controls the
        (u, v) axis frame passed to ``matplotlib.pyplot.imshow``.

    Args:
        extent: Axis bounds (u_min, u_max, v_min, v_max) passed to imshow.

    Returns:
        None. Shows a matplotlib figure.
    """
    (x, y, α, θ) = get_maps()
    (_, axs) = plt.subplots(2, 2, figsize=(13, 10))
    im = axs[0, 0].imshow(x, extent=extent, cmap='viridis')
    axs[0, 0].set_title('U map (px)')
    axs[0, 0].set_xlabel('U')
    axs[0, 0].set_ylabel('V')
    plt.colorbar(im, ax=axs[0, 0])
    im = axs[0, 1].imshow(y, extent=extent, cmap='viridis')
    axs[0, 1].set_title('V map (px)')
    axs[0, 1].set_xlabel('U')
    axs[0, 1].set_ylabel('V')
    plt.colorbar(im, ax=axs[0, 1])
    im = axs[1, 0].imshow(θ.value, extent=extent, cmap='viridis')
    axs[1, 0].set_title(f'Theta map ({θ.unit})')
    axs[1, 0].set_xlabel('U')
    axs[1, 0].set_ylabel('V')
    plt.colorbar(im, ax=axs[1, 0])
    im = axs[1, 1].imshow(α.value, extent=extent, cmap='viridis')
    axs[1, 1].set_title(f'Alpha map ({α.unit})')
    axs[1, 1].set_xlabel('U')
    axs[1, 1].set_ylabel('V')
    plt.colorbar(im, ax=axs[1, 1])
    plt.show()

@nb.njit()
def αθ_to_xy_njit(α: float, θ: float, fov: float) -> tuple[float, float]:
    """Convert (α, θ) to normalized Cartesian coordinates (u, v) (njit version).

    Args:
        α: Parallactic angle in radians.
        θ: Angular separation in radians.
        fov: Total field of view in radians (used for normalization).

    Returns:
        Tuple ``(u, v)`` representing normalized positions in [-1, 1].
    """
    x = 2 * θ / fov * np.cos(α)
    y = 2 * θ / fov * np.sin(α)
    return x, y

def αθ_to_xy(α: u.Quantity, θ: u.Quantity, fov: u.Quantity) -> tuple[u.Quantity, u.Quantity]:
    """Convert (α, θ) to Cartesian coordinates (u, v) with units.

    Args:
        α: Parallactic angle in angle units (converted to ``u.rad``).
        θ: Angular separation (converted to radians for computation).
        fov: Total field of view (converted to radians for normalization).

    Returns:
        Tuple ``(u, v)`` as unitless Quantities (normalized values),
        corresponding to normalized coordinates in [-1, 1].
    """
    α = α.to(u.rad).value
    θ = θ.to(u.rad).value
    fov = fov.to(u.rad).value
    return αθ_to_xy_njit(α=α, θ=θ, fov=fov)