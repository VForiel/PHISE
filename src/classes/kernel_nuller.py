# External libs
import numpy as np
import numba as nb
import astropy.units as u

# Internal libs
from ..modules import mmi
from ..modules import phase

#==============================================================================
# Kernel-Nuller class
#==============================================================================

class KernelNuller():
    def __init__(
            self,
            φ: np.ndarray[u.Quantity],
            σ: np.ndarray[u.Quantity],
            output_order: np.ndarray[int] = None,
            name: str = "Unnamed",
        ):
        """Kernel-Nuller object.

        Parameters
        ----------
        - φ: Array of 14 injected OPD
        - σ: Array of 14 intrasic OPD error
        - output_order: Order of the outputs
        - name: Name of the Kernel-Nuller object
        """
        self.φ = φ
        self.σ = σ
        self.output_order = output_order if output_order is not None else np.array([0, 1, 2, 3, 4, 5])
        self.name = name

    # φ property --------------------------------------------------------------

    @property
    def φ(self):
        return self._φ
    
    @φ.setter
    def φ(self, φ:np.ndarray[u.Quantity]):
        if type(φ) != u.Quantity:
            raise ValueError("φ must be a Quantity")
        try:
            φ.to(u.m)
        except u.UnitConversionError:
            raise ValueError("φ must be in a distance unit")
        if φ.shape != (14,):
            raise ValueError("φ must have a shape of (14,)")
        if np.any(φ < 0):
            raise ValueError("φ must be positive")
        self._φ = φ

    # σ property --------------------------------------------------------------

    @property
    def σ(self):
        return self._σ
    
    @σ.setter
    def σ(self, σ:np.ndarray[u.Quantity]):
        if type(σ) != u.Quantity:
            raise ValueError("σ must be a Quantity")
        try:
            σ.to(u.m)
        except u.UnitConversionError:
            raise ValueError("σ must be in a distance unit")
        if σ.shape != (14,):
            raise ValueError("σ must have a shape of (14,)")
        self._σ = σ

    # Output order property --------------------------------------------------

    @property
    def output_order(self):
        return self._output_order
    
    @output_order.setter
    def output_order(self, output_order:np.ndarray[int]):
        try:
            output_order = np.array(output_order, dtype=int)
        except:
            raise ValueError(f"output_order must be an array of integers, not {type(output_order)}")
        if output_order.shape != (6,):
            raise ValueError(f"output_order must have a shape of (6,), not {output_order.shape}")
        if not np.all(np.sort(output_order) == np.arange(6)):
            raise ValueError(f"output_order must contain all the integers from 0 to 5, not {output_order}")
        if output_order[0] - output_order[1] not in [-1, 1] \
                or output_order[2] - output_order[3] not in [-1, 1] \
                or output_order[4] - output_order[5] not in [-1, 1]:
            raise ValueError(f"output_order contain an invalid configuration of output pairs. Found {output_order}")
        self._output_order = output_order

    # Name property -----------------------------------------------------------

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name: str):
        if not isinstance(name, str):
            raise ValueError("name must be a string")
        self._name = name

    # Electric fields propagation ---------------------------------------------

    def propagate_fields(
            self,
            ψ: np.ndarray[complex],
            λ: u.Quantity,
        ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], float]:
        """
        Simulate a 4 telescope Kernel-Nuller field propagation (at a given wavelength) using a numeric approach

        Parameters
        ----------
        - ψ: Array of 4 input signals complex amplitudes
        - λ: Wavelength of the light

        Returns
        -------
        - Array of 3 null outputs electric fields
        - Array of 6 dark outputs electric fields
        - Bright output electric fields
        """
        φ = self.φ.to(λ.unit).value
        σ = self.σ.to(λ.unit).value
        λ = λ.value

        return propagate_fields_njit(ψ=ψ, φ=φ, σ=σ, λ=λ, output_order=self.output_order)

#==============================================================================
# Numba functions
#==============================================================================

# Electric fields propagation -------------------------------------------------

@nb.njit()
def propagate_fields_njit(
        ψ: np.ndarray[complex],
        φ: np.ndarray[float],
        σ: np.ndarray[float],
        λ: float,
        output_order:np.ndarray[int]
    ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], float]:
    """
    Simulate a 4 telescope Kernel-Nuller propagation using a numeric approach

    Parameters
    ----------
    - ψ: Array of 4 input signals complex amplitudes
    - φ: Array of 14 injected OPD (in wavelenght unit)
    - σ: Array of 14 intrasic OPD error (in wavelenght unit)
    - λ: Wavelength of the light
    - output_order: Order of the outputs

    Returns
    -------
    - Array of 3 null outputs electric fields
    - Array of 6 dark outputs electric fields
    - Bright output electric fields
    """

    φ = phase.bound_njit(φ + σ, λ)

    # First layer of pahse shifters
    nuller_inputs = phase.shift_njit(ψ, φ[:4], λ)

    # First layer of nulling
    N1 = mmi.nuller_2x2(nuller_inputs[:2])
    N2 = mmi.nuller_2x2(nuller_inputs[2:])

    # Second layer of phase shifters
    N1_shifted = phase.shift_njit(N1, φ[4:6], λ)
    N2_shifted = phase.shift_njit(N2, φ[6:8], λ)

    # Second layer of nulling
    N3 = mmi.nuller_2x2(np.array([N1_shifted[0], N2_shifted[0]]))
    N4 = mmi.nuller_2x2(np.array([N1_shifted[1], N2_shifted[1]]))

    nulls = np.array([N3[1], N4[0], N4[1]], dtype=np.complex128)
    bright = N3[0]

    # Beam splitting
    R_inputs = np.array([N3[1], N3[1], N4[0], N4[0], N4[1], N4[1]]) * 1 / np.sqrt(2)

    # Last layer of phase shifters
    R_inputs = phase.shift_njit(R_inputs, φ[8:], λ)

    # Beam mixing
    R1_output = mmi.cross_recombiner_2x2(np.array([R_inputs[0], R_inputs[2]]))
    R2_output = mmi.cross_recombiner_2x2(np.array([R_inputs[1], R_inputs[4]]))
    R3_output = mmi.cross_recombiner_2x2(np.array([R_inputs[3], R_inputs[5]]))

    darks = np.array(
        [
            R1_output[0],
            R1_output[1],
            R2_output[0],
            R2_output[1],
            R3_output[0],
            R3_output[1],
        ],
        dtype=np.complex128,
    )

    darks = darks[output_order]

    return nulls, darks, bright
