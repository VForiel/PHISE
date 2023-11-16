import numpy as np
import matplotlib.pyplot as plt

def gauss(x, mu, sigma):
    return np.exp(-(x-mu)**2/(2*sigma**2))

def multi_imshow(x, title, cmap="inferno"):
    
    N = len(x)
    Ncol = int(np.ceil(np.sqrt(N)))
    Nrow = int(np.ceil(N / Ncol))

    fig, axs = plt.subplots(Nrow, Ncol, figsize=(12,8))

    if Nrow == 1:
        axs = np.array([axs])

    fig.suptitle(title, fontsize=14)

    for i, e in enumerate(x):
        j = i // Ncol
        k = i % Ncol

        im = axs[j,k].imshow(np.abs(e), cmap=cmap)
        axs[j,k].set_title(f"Output {i}")
        cbar = fig.colorbar(im, ax=axs[j,k])
        im.set_clim(np.min(np.abs(x)), np.max(np.abs(x)))
    plt.show()