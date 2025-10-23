"""Module generated docstring."""
from astropy import units as u
import numpy as np
import numba as nb
from typing import Union

@nb.njit()
def shift_njit(ψ: Union[complex, np.ndarray[complex]], δφ: Union[float, np.ndarray[float]], λ: float) -> Union[complex, np.ndarray[complex]]:
    """Version numba-jittée de la rotation de phase d'un champ électrique.

    Applique un déphasage δφ au champ ψ à la longueur d'onde λ.
    """
    return ψ * np.exp(1j * 2 * np.pi * δφ / λ)

def shift(ψ: Union[complex, np.ndarray[complex]], δφ: u.Quantity, λ: u.Quantity) -> Union[complex, np.ndarray[complex]]:
    """Appliquer un déphasage (interface utilisateur).

    Convertit les Quantity en valeurs numériques puis appelle
    `shift_njit`.
    """
    δφ = δφ.to(λ.unit).value
    λ = λ.value
    return shift_njit(ψ, δφ, λ)

def bound(φ: u.Quantity, λ: u.Quantity) -> u.Quantity:
    """Ramener une phase dans l'intervalle [0, λ[.

    Les deux arguments sont des Quantities en unités de longueur.
    """
    return bound_njit(φ.value, λ.to(φ.unit).value) * φ.unit

@nb.njit()
def bound_njit(φ: float, λ: float) -> float:
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

def perturb(φ: np.ndarray[u.Quantity], rms: u.Quantity) -> u.Quantity:
    """Ajouter un bruit gaussien aux phases avec dispersion `rms`.

    Renvoie un Quantity de même forme que `φ`.
    """
    rms = rms.to(φ.unit).value
    err = np.random.normal(0, rms, size=len(φ)) * φ.unit
    return φ + err