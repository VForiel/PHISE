# External libs
import numpy as np
import astropy.units as u
import numba as nb

# Internal libs
from .interferometer import Interferometer

class Camera:
    def __init__(self, e:u.Quantity, name:str = "Unnamed"):
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

    def acquire(self, ψ:np.ndarray[complex]) -> np.ndarray[int]:
        """
        Acquire the image from the interferometer.

        Parameters
        ----------
        - ψ: Complex visibility [s^(-1/2)]

        Returns
        -------
        - Image
        """

        return acquire_njit(ψ, self.e.to(u.s).value)

#==============================================================================
# Numba functions
#==============================================================================

@nb.njit()
def acquire_njit(ψ: np.ndarray[complex], e: float) -> np.ndarray[int]:
    """
    Acquire the intensities from the complex visibilities.

    Parameters
    ----------
    - ψ: Complex visibilities [s^(-1/2)]
    - e: Exposure time [s]

    Returns
    -------
    - np.ndarray[int]: Intensities
    """

    # Get intensities
    I = np.abs(ψ)**2 * e

    # Add photon noise
    Ip = I * (I <= 2147020237) # Using poisson noise
    In = I * (I > 2147020237) # Using normal noise (I too high to use poisson)

    for i in range(I.shape[0]):
        I[i] = int(np.random.poisson(Ip[i]))
        I[i] += int(np.random.normal(In[i], np.sqrt(In[i])))

    return I