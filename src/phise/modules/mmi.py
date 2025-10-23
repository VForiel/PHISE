"""Recombination blocks (MMI, nuller) for 2x2 combinations.

Contains ideal matrices to simulate a 2x2 nuller (bright/dark) and a 2x2
cross recombiner.
"""
import numpy as np
import numba as nb

@nb.njit()
def nuller_2x2(beams: np.ndarray) -> np.ndarray:
    """Ideal 2x2 nuller (Hadamard) applied to two complex beams.

    Args:
        beams: Array of shape (2,) or (2, M) of input complex amplitudes.

    Returns:
        Complex array of shape (2,) or (2, M):
        - index 0: bright (constructive) output,
        - index 1: dark (destructive) output.
    """
    N = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    return N @ beams

@nb.njit()
def cross_recombiner_2x2(beams: np.ndarray) -> np.ndarray:
    """2x2 cross recombiner (MMI) with ideal quadrature phase.

    Args:
        beams: Array of shape (2,) or (2, M) of input complex amplitudes.

    Returns:
        Complex array of shape (2,) or (2, M) for MMI outputs.
    """
    θ: float = np.pi / 2
    S = 1 / np.sqrt(2) * np.array([[np.exp(1j * θ / 2), np.exp(-1j * θ / 2)], [np.exp(-1j * θ / 2), np.exp(1j * θ / 2)]])
    return S @ beams