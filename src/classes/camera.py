# Trick to import Target but avoiding circular import
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .interferometer import Interferometer

# External libs
import numpy as np
import astropy.units as u
import numba as nb

class Camera:
    def __init__(self, e:u.Quantity, name:str = "Unnamed Camera"):
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

    def acquire_pixel(self, ψ:np.ndarray[complex]) -> int:
        """
        Acquire the image from the interferometer.

        Parameters
        ----------
        ψ: np.ndarray[complex]
            Complex visibilities [s^(-1/2)]

        Returns
        -------
        int
            Number of photons detected
        """

        return acquire_pixel_njit(ψ, self.e.to(u.s).value)

#==============================================================================
# Numba functions
#==============================================================================

@nb.njit()
def acquire_pixel_njit(ψ: np.ndarray[complex], e: float) -> int:
    """
    Acquire the intensities from the complex visibilities.

    Parameters
    ----------
    ψ: np.ndarray[complex]
        Complex visibilities [s^(-1/2)]
    - e: float
        Exposure time [s]

    Returns
    -------
    int
        Number of photons detected
    """

    # Get intensities
    I = int(np.sum(np.abs(ψ)**2) * e)

    # Add photon noise
    if I <= 2147020237: # Using poisson noise
        I = int(np.random.poisson(I))
    else: # Using gaussian noise
        I = int(np.random.normal(I, np.sqrt(I)))

    return I