"""Module generated docstring."""
import numpy as np
import numba as nb

@nb.njit()
def nuller_2x2(beams: np.ndarray[complex]) -> np.ndarray[complex]:
    """
    Simulate a 2 input beam nuller.

    Parameters
    ----------
    - beams: Array of 2 input beams complex amplitudes

    Returns
    -------
    - Array of 2 output beams complex amplitudes
        - 1st output is the bright channel
        - 2nd output is the dark channel
    """
    N = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    return N @ beams

@nb.njit()
def cross_recombiner_2x2(beams: np.array) -> np.array:
    """
    Simulate a 2x2 cross recombiner MMI

    Parameters
    ----------
    - beams: Array of 2 input beams complex amplitudes

    Returns
    -------
    - Array of 2 output beams complex amplitudes
    """
    θ: float = np.pi / 2
    S = 1 / np.sqrt(2) * np.array([[np.exp(1j * θ / 2), np.exp(-1j * θ / 2)], [np.exp(-1j * θ / 2), np.exp(1j * θ / 2)]])
    return S @ beams