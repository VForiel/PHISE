"""Phase operations: shifting, wrapping and random perturbations.

Utility functions to apply wavelength-dependent phase shifts, wrap phases
into an interval, and add Gaussian noise.
"""
from astropy import units as u
import numpy as np
import numba as nb
from typing import Union

@nb.njit()
def shift_jit(
    ψ: Union[complex, np.ndarray],
    δφ: Union[float, np.ndarray],
    λ: float,
) -> Union[complex, np.ndarray]:
    """Phase rotation (njit) of an electric field.

    Args:
        ψ: Complex electric field (scalar or array).
        δφ: Applied phase shift (same unit as ``λ`` once converted).
        λ: Wavelength (scalar, in length units).

    Returns:
        Field with phase shift applied, same shape as ``ψ``.
    """
    return ψ * np.exp(1j * 2 * np.pi * δφ / λ)

def shift(
    ψ: Union[complex, np.ndarray],
    δφ: u.Quantity,
    λ: u.Quantity,
) -> Union[complex, np.ndarray]:
    """Apply a phase shift with unit handling.

    Converts ``Quantity`` inputs to numeric values then calls
    :func:`shift_jit`.

    Args:
        ψ: Complex electric field (scalar or array).
        δφ: Phase shift (length ``Quantity``, e.g. meters).
        λ: Wavelength (length ``Quantity``) used in 2π δφ/λ.

    Returns:
        Field with phase shift applied, same shape as ``ψ``.
    """
    δφ = δφ.to(λ.unit).value
    λ = λ.value
    return shift_jit(ψ, δφ, λ)

def bound(φ: u.Quantity, λ: u.Quantity) -> u.Quantity:
    """Wrap a phase into [0, λ[ while preserving units.

    Args:
        φ: Phase (length ``Quantity``) to wrap.
        λ: Wavelength (length ``Quantity``) defining the interval.

    Returns:
        Wrapped phase in [0, λ[ with the unit of ``φ``.
    """
    return bound_jit(φ.value, λ.to(φ.unit).value) * φ.unit

@nb.njit()
def bound_jit(φ: float, λ: float) -> float:
    """Wrap a scalar phase into [0, λ[ (njit version).

    Args:
        φ: Scalar phase to wrap.
        λ: Scalar wavelength defining the interval.

    Returns:
        Wrapped phase in [0, λ[ (implicit same unit as inputs).
    """
    return np.mod(φ, λ)

def perturb(φ: np.ndarray, rms: u.Quantity) -> u.Quantity:
    """Add Gaussian noise to phases with standard deviation ``rms``.

    Args:
        φ: Array of phases as ``Quantity`` (all with the same unit).
        rms: Noise standard deviation, same unit as ``φ``.

    Returns:
        ``Quantity`` array of the same shape as ``φ`` with noisy phases.
    """
    rms = rms.to(φ.unit).value
    err = np.random.normal(0, rms, size=len(φ)) * φ.unit
    return φ + err