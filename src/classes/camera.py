# Trick to import Target but avoiding circular import
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .interferometer import Interferometer

# External libs
import numpy as np
import astropy.units as u
import numba as nb
import math

class Camera:
    def __init__(self, e:u.Quantity = 1 * u.s, name:str = "Unnamed Camera"):
        """
        Initialize the camera object.

        Parameters
        ----------
        - e: Exposure time
        - name: Name of the camera
        """

        self._parent_interferometer = None

        self.e = e
        self.name = name

    # To string ---------------------------------------------------------------

    def __str__(self) -> str:
        res = f'Camera "{self.name}"\n'
        res += f'  Exposure time: {self.e:.2f}'
        return res

    def __repr__(self) -> str:
        return self.__str__()

    # e property --------------------------------------------------------------

    @property
    def e(self) -> u.Quantity:
        return self._e
    
    @e.setter
    def e(self, e:u.Quantity):
        if not isinstance(e, u.Quantity):
            raise TypeError("e must be an astropy Quantity")
        try:
            e = e.to(u.s)
        except u.UnitConversionError:
            raise ValueError("e must be in a time unit")
        self._e = e

    # parent_interferometer property ------------------------------------------

    @property
    def parent_interferometer(self) -> Interferometer:
        return self._parent_interferometer
    
    @parent_interferometer.setter
    def parent_interferometer(self, _):
        raise ValueError("parent_interferometer is read-only")
    
    # Acquire -----------------------------------------------------------------

    def acquire_pixel(self, ψ: np.ndarray[complex]) -> int:
        """
        Acquire the intensities from the complex visibilities.

        Parameters
        ----------
        ψ : np.ndarray[complex]
            Complex electric fields in [s^(-1/2)].

        Returns
        -------
        int
            Number of photons detected during the integration time.
        """

        # Expected intensity (photons/s), integrated over the exposure time (s)
        expected_photons = np.sum(np.abs(ψ)**2) * self.e.to(u.s).value

        # Add photon noise
        if expected_photons <= 2e9:
            detected_photons = np.random.poisson(expected_photons)
        else:
            detected_photons = int(expected_photons + np.random.normal(0, math.sqrt(expected_photons)))

        # print("Acquiring pixel...\n   ψ =", ψ, "\n   Detected =", detected_photons)

        return detected_photons
        # return acquire_pixel_njit(ψ, self.e.to(u.s).value)

#==============================================================================
# Numba functions
#==============================================================================

# @nb.njit()
# def acquire_pixel_njit(ψ: np.ndarray[complex], e: float) -> int:
#     """
#     Acquire the intensities from the complex visibilities.

#     Parameters
#     ----------
#     ψ: np.ndarray[complex]
#         Complex visibilities [s^(-1/2)]
#     - e: float
#         Exposure time [s]

#     Returns
#     -------
#     int
#         Number of photons detected
#     """

#     # Get intensities
#     I = int(np.sum(np.abs(ψ)**2) * e)

#     # Add photon noise
#     if I <= 2147020237: # Using poisson noise
#         I = int(np.random.poisson(I))
#     else: # Using gaussian noise
#         I = int(np.random.normal(I, np.sqrt(I)))

#     return I