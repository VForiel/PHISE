"""Signal utilities: formatting and photon flux."""
import numpy as np
import numba as nb
import astropy.units as u
from astropy import constants as const

def as_str(signals: np.ndarray) -> str:
    """Return a compact text representation of a signal vector.

    Each complex entry is displayed as amplitude · exp(i·phase) and its
    intensity |s|².

    Args:
        signals: Complex amplitudes array, shape (N,) or (N, M).

    Returns:
        Formatted string listing telescopes and their signals.
    """
    res = ''
    for (i, s) in enumerate(signals):
        res += f' - Telescope {i}:   {np.abs(s):.2e} *exp(i* {np.angle(s) / np.pi:.2f} *pi)   ->   {np.abs(s) ** 2:.2e}\n'
    return res[:-1]

def photon_flux(
    λ: u.Quantity,
    Δλ: u.Quantity,
    f: u.Quantity,
    a: u.Quantity,
    η: float,
    m: float,
) -> u.Quantity:
    """Compute the photon detection rate for a star (photons per second).

    Formula: Ṅ ≈ f · a · η · δν · 10^{-m/2.5} / (h c / λ), with δν ≈ c Δλ / λ².

    Args:
        λ: Central wavelength (length Quantity, e.g. ``u.m``).
        Δλ: Spectral width (same length unit as ``λ``).
        f: Object spectral flux (e.g. W·m⁻²·Hz⁻¹ or compatible).
        a: Telescope collecting area (area Quantity, e.g. ``u.m**2``).
        η: Optical throughput (0–1).
        m: Stellar magnitude (astronomical logarithmic scale).

    Returns:
        Detection rate in ``photons / s`` (Quantity ``1 / u.s``).
    """
    h = const.h.to(u.J * u.s)
    c = const.c.to(u.m / u.s)
    δν = c / λ ** 2 * Δλ
    e = h * c / λ
    return (f * a * δν * η * 10 ** (-m.value / 2.5) / e).to(1 / u.s)