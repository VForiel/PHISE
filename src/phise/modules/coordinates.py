"""Module generated docstring."""
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
def get_maps_njit(N: int, fov: float) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Generate a map of theta and alpha values for a given resolution.

    Parameters
    ----------
    - N: Resolution of the map
    - fov: Field of view in mas

    Returns
    -------
    - Normalized U map (resolution x resolution)
    - Normalized V map (resolution x resolution)
    - Alpha map (resolution x resolution, in radians)
    - Theta map (resolution x resolution, in fov unit)
    """
    x_map = np.zeros((N, N))
    y_map = np.zeros((N, N))
    for (i, v) in enumerate(np.linspace(-1, 1, N)):
        x_map[:, i] = v
        y_map[i, :] = v
    θ_map = np.sqrt(x_map ** 2 + y_map ** 2) * fov / 2
    α_map = np.arctan2(y_map, x_map) % (2 * np.pi)
    return (x_map, y_map, α_map, θ_map)

def get_maps(N: int, fov: u.Quantity) -> tuple[np.ndarray[float], np.ndarray[float], u.Quantity, u.Quantity]:
    """
    Generate a map of theta and alpha values for a given resolution.

    Parameters
    ----------
    - N: Resolution of the map
    - fov: Range of field of view values

    Returns
    -------
    - Normalized U map (resolution x resolution)
    - Normalized V map (resolution x resolution)
    - Alpha map (resolution x resolution, in radian)
    - Theta map (resolution x resolution, in fov unit)
    """
    (x_map, y_map, α_map, θ_map) = get_maps_njit(N=N, fov=fov.value)
    α_map *= u.rad
    θ_map *= fov.unit
    return (x_map, y_map, α_map, θ_map)

def plot_uv_map(extent: tuple[float, float, float, float]):
    """"plot_uv_map.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
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
    """
    Convert alpha and theta values to the x and y angle from the center of the field of view.

    Parameters
    ----------
    - α: Parallactic angle (rad)
    - θ: Angular separation (rad)
    - fov: Field of view (rad)

    Returns
    -------
    - U value
    - V value
    """
    x = 2 * θ / fov * np.cos(α)
    y = 2 * θ / fov * np.sin(α)
    return (x, y)

def αθ_to_xy(α: u.Quantity, θ: u.Quantity, fov: u.Quantity) -> tuple[u.Quantity, u.Quantity]:
    """
    Convert alpha and theta values to the x and y angle from the center of the field of view.

    Parameters
    ----------
    - α: Parallactic angle
    - θ: Angular separation
    - fov: Field of view

    Returns
    -------
    - U value
    - V value
    """
    α = α.to(u.rad).value
    θ = θ.to(u.rad).value
    fov = fov.to(u.rad).value
    return αθ_to_xy_njit(α=α, θ=θ, fov=fov)