import numpy as np
import numba as nb
import astropy.units as u
from astropy import constants as const

def get_input_fields(
        a: float,
        θ: u.Quantity,
        α: u.Quantity,
        λ: u.Quantity,
        p: u.Quantity,
    ) -> np.ndarray[complex]:
    """
    Get the phase of the 4 inputs according to the object and telescope positions.

    Parameters
    ----------
    - a: Amplitude of the signal (prop. to #photons/s)
    - θ: Angular separation
    - α: Parallactic angle
    - λ: Wavelength
    - p: Projected telescope positions

    Returns
    -------
    - Array of acquired signals (complex amplitudes)
    """
    α = α.to(u.rad).value
    θ = θ.to(u.rad).value
    λ = λ.to(u.m).value
    p = p.to(u.m).value
    return get_input_fields_njit(a, θ, α, λ, p)

@nb.njit()
def get_input_fields_njit(
    a: float,
    θ: float,
    α: float,
    λ: float,
    p: np.ndarray[float],
) -> np.ndarray[complex]:
    """
    Get the complexe amplitude of the input signals according to the object and telescopes positions.

    Parameters
    ----------
    - a: Amplitude of the signal (prop. to #photons/s)
    - θ: Angular separation (in radian)
    - α: Parallactic angle (in radian)
    - λ: Wavelength (in meter)
    - r: Projected telescope positions (in meter)

    Returns
    -------
    - Array of acquired signals (complex amplitudes).
    """

    # Array of complex signals
    s = np.empty(p.shape[0], dtype=np.complex128)

    for i, t in enumerate(p):

        # Rotate the projected telescope positions by the parallactic angle
        p_rot = t[0] * np.cos(-α) - t[1] * np.sin(-α)

        # Compute the phase delay according to the object position
        Φ = 2 * np.pi * p_rot * np.sin(θ) / λ

        # Build the complex amplitude of the signal
        s[i] = a * np.exp(1j * Φ)

    return s / np.sqrt(p.shape[0])

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

    res = ""
    for i, s in enumerate(signals):
        res += f" - Telescope {i}:   {np.abs(s):.2e} *exp(i* {np.angle(s)/np.pi:.2f} *pi)   ->   {np.abs(s)**2:.2e}\n"
    return res[:-1]

def nb_photons(
        λ:u.Quantity,
        Δλ:u.Quantity,
        f:u.Quantity,
        a:u.Quantity,
        η:float,
        m:float,
    ) -> u.Quantity:
    """
    Compute the number of photons detected by the telescope per second.

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
    c = const.c.to(u.m/u.s)
    δν = c/(λ**2) * Δλ
    e = h*c/λ

    return (f * a * δν * η * 10**(-m.value/2.5) / e).to(1/u.s)