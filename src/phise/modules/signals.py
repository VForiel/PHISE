"""Module generated docstring."""
import numpy as np
import numba as nb
import astropy.units as u
from astropy import constants as const

def as_str(signals: np.ndarray[complex]) -> str:
    """
    Convert signals to a string.

    Parameters
    ----------
    - signals : Signals to convert.

    Returns
    -------
    - String representation of the signals.
    """
    res = ''
    for (i, s) in enumerate(signals):
        res += f' - Telescope {i}:   {np.abs(s):.2e} *exp(i* {np.angle(s) / np.pi:.2f} *pi)   ->   {np.abs(s) ** 2:.2e}\n'
    return res[:-1]

def photon_flux(λ: u.Quantity, Δλ: u.Quantity, f: u.Quantity, a: u.Quantity, η: float, m: float) -> u.Quantity:
    """
    Compute the number of photons per second coming from a star and captured by a telescope.

    Parameters
    ----------
    - λ: Wavelength of the light
    - Δλ: Spectral width
    - f: Flux of the object
    - a: Area of the telescope
    - η: Optical efficiency
    - m: Magnitude of the star

    Returns
    -------
    - Number of photons detected by the telescope per second
    """
    h = const.h.to(u.J * u.s)
    c = const.c.to(u.m / u.s)
    δν = c / λ ** 2 * Δλ
    e = h * c / λ
    return (f * a * δν * η * 10 ** (-m.value / 2.5) / e).to(1 / u.s)