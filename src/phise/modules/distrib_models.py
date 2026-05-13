import numpy as np
import scipy

def imb(z, μ, σ, ν):
    # Felix Dannert et al. 2025
    # Well describe kernel outputs without planet

    ν=0.8

    v = (ν - 1)/2
    a = 2**((1-ν)/2) * np.sqrt(ν)
    b = σ * np.sqrt(np.pi) * scipy.special.gamma(ν/2)
    c = np.abs((z-μ) / (σ * np.sqrt(ν)))
    k = scipy.special.kv(v, c)

    return (a / b) * c**v * k