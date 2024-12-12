from astropy import units as u
import numpy as np
import numba as nb

# Phase shift -----------------------------------------------------------------

@nb.njit()
def shift_njit(
    ψ: complex | np.ndarray[complex],
    δφ: float | np.ndarray[float],
    λ: float,
) -> complex | np.ndarray[complex]:
    """
    De-phase the input beam by heating the fiber with an electrical current.

    Parameters
    ----------
    - ψ: input beam complex amplitude
    - δφ: phase to add (in same unit as wavelenght)
    - λ: wavelength

    Returns
    -------
    - Output beam complex amplitude
    """
    return ψ * np.exp(1j * 2 * np.pi * δφ / λ)


def shift(
    ψ: complex | np.ndarray[complex],
    δφ: u.Quantity,
    λ: u.Quantity,
) -> complex | np.ndarray[complex]:
    """
    De-phase the input beam by heating the fiber with an electrical current.

    Parameters
    ----------
    - ψ: input beam complex amplitude
    - δφ: phase to add
    - λ: wavelength

    Returns
    -------
    - Output beam complex amplitude
    """
    δφ = δφ.to(λ.unit).value
    λ = λ.value
    return shift_njit(ψ, δφ, λ)

# Bound phase -----------------------------------------------------------------

# We only consider relative phases, so we only consider phase shift in [0,wavelenght[

def bound(
        φ:u.Quantity,
        λ:u.Quantity,
    ) -> u.Quantity:
    """Bring a phase to the interval [0, wavelenght[.

    Parameters
    ----------
    - φ: Phase to bound (any distance unit)
    - λ: Wavelenght of the light (any distance unit)

    Returns
    -------
    - Phase in the interval [0, wavelenght]
    """
    return bound_njit(φ.value, λ.to(φ.unit).value) * φ.unit

@nb.njit()
def bound_njit(
        φ:float,
        λ:float,
    ) -> float:
    """Bring a phase to the interval [0, wavelenght[.

    Parameters
    ----------
    - φ: Phase to bound (in distance unit)
    - λ: Wavelenght of the light (same unit as phase) 

    Returns
    -------
    - Phase in the interval [0, wavelenght]
    """
    return np.mod(φ, λ)

# Perturbation ----------------------------------------------------------------

def perturb(
        φ:np.ndarray[u.Quantity],
        rms:u.Quantity,
    ) -> u.Quantity:
    """Add a random perturbation to a phase.

    Parameters
    ----------
    - φ: Phase to perturb
    - rms: Perturbation RMS

    Returns
    -------
    - Perturbed phase
    """

    rms = rms.to(φ.unit).value
    err = np.random.normal(0, rms, size=len(φ)) * φ.unit

    return φ + err