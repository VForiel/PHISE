"""Cartographie de la contribution des noyaux sur le ciel.

Fournit des fonctions pour calculer et afficher les cartes de
transmission des différents kernels et la zone de contribution totale.
"""
import numpy as np
import matplotlib.pyplot as plt
try:
    plt.rcParams['image.origin'] = 'lower'
except Exception:
    pass
from copy import deepcopy as copy
from astropy import units as u
from phise.classes import Context
from phise.modules import coordinates

def plot(ctx: Context=None, resolution: int=100, n=100, map=np.median):
    """
    Plot the contribution zones of the kernels in the sky.

    Parameters
    ----------
    ctx : Context, optional
        The context to use for the plot. If None, a default context is used.
    resolution : int, optional
        The resolution of the plot. Default is 100.
    map : function, optional
        The function to use for mapping the images. Default is np.median.
    """
    if ctx is None:
        ref_ctx = Context.get_VLTI()
        ref_ctx.interferometer.kn.σ = np.zeros(14) * u.nm
    else:
        ref_ctx = copy(ctx)
    images = np.zeros((3, resolution, resolution))
    (_, _, ρ_map, _) = coordinates.get_maps(N=resolution, fov=ref_ctx.interferometer.fov)
    ρ_map = ρ_map.value / np.max(ρ_map.value)
    h_range = ref_ctx.get_h_range()
    for (i, h) in enumerate(h_range):
        ctx = copy(ref_ctx)
        ctx.h = h
        raw_data = np.empty((n, 3))
        for j in range(n):
            (_, k, _) = ctx.observe()
            raw_data[j] = k
        transmission_maps = ctx.get_transmission_maps(N=resolution)[2]
        for k in range(3):
            data = map(raw_data[:, k])
            transmission_map = transmission_maps[k]
            images[k] += transmission_map * data / len(h_range)
    images[images < 0] = 0
    max_im = np.max(images)
    (_, axs) = plt.subplots(1, 4, figsize=(25, 5))
    fov = ref_ctx.interferometer.fov.to(u.mas)
    extent = [-fov.value / 2, fov.value / 2, -fov.value / 2, fov.value / 2]
    for k in range(3):
        img = images[k]
        img[img < 0] = 0
        im = axs[k].imshow(img, cmap='hot', vmax=max_im, extent=extent)
        axs[k].set_title(f'Kernel {k + 1}')
        plt.colorbar(im, ax=axs[k])
        for companion in ref_ctx.target.companions:
            (planet_x, planet_y) = coordinates.ρθ_to_xy(ρ=companion.ρ, θ=companion.θ, fov=fov)
            axs[k].scatter(planet_x * fov / 2, planet_y * fov / 2, color='tab:blue', edgecolors='black')
    stack = np.prod(images, axis=0) ** (1 / 3)
    im = axs[3].imshow(stack, cmap='hot', extent=extent)
    axs[3].set_title('Contribution zones')
    plt.colorbar(im, ax=axs[3])
    for companion in ref_ctx.target.companions:
        (planet_x, planet_y) = coordinates.ρθ_to_xy(ρ=companion.ρ, θ=companion.θ, fov=fov)
        axs[3].scatter(planet_x * fov / 2, planet_y * fov / 2, color='tab:blue', edgecolors='black')
    plt.show()