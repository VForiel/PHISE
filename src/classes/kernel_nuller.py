# External libs
import numpy as np
import numba as nb
import astropy.units as u
import matplotlib.pyplot as plt
from io import BytesIO
from LRFutils import color
from copy import deepcopy as copy

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

        self._parent_interferometer = None

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

    # Parent interferometer property ------------------------------------------

    @property
    def parent_interferometer(self):
        return self._parent_interferometer
    
    @parent_interferometer.setter
    def parent_interferometer(self, parent_interferometer):
        raise ValueError("parent_interferometer is read-only")

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
    
    # Plot phases -------------------------------------------------------------

    def plot_phase(
            self,
            λ,
            ψ=np.array([0.5+0j, 0.5+0j, 0.5+0j, 0.5+0j]),
            plot=True
        ):

        ψ1 = np.array([ψ[0], 0, 0, 0])
        ψ2 = np.array([0, ψ[1], 0, 0])
        ψ3 = np.array([0, 0, ψ[2], 0])
        ψ4 = np.array([0, 0, 0, ψ[3]])

        n1, d1, b1 = self.propagate_fields(ψ1, λ)
        n2, d2, b2 = self.propagate_fields(ψ2, λ)
        n3, d3, b3 = self.propagate_fields(ψ3, λ)
        n4, d4, b4 = self.propagate_fields(ψ4, λ)

        # Using first signal as reference
        n2 = np.abs(n2) * np.exp(1j * (np.angle(n2) - np.angle(n1)))
        n3 = np.abs(n3) * np.exp(1j * (np.angle(n3) - np.angle(n1)))
        n4 = np.abs(n4) * np.exp(1j * (np.angle(n4) - np.angle(n1)))
        d2 = np.abs(d2) * np.exp(1j * (np.angle(d2) - np.angle(d1)))
        d3 = np.abs(d3) * np.exp(1j * (np.angle(d3) - np.angle(d1)))
        d4 = np.abs(d4) * np.exp(1j * (np.angle(d4) - np.angle(d1)))
        b2 = np.abs(b2) * np.exp(1j * (np.angle(b2) - np.angle(b1)))
        b3 = np.abs(b3) * np.exp(1j * (np.angle(b3) - np.angle(b1)))
        b4 = np.abs(b4) * np.exp(1j * (np.angle(b4) - np.angle(b1)))
        n1 = np.abs(n1) * np.exp(1j * 0)
        d1 = np.abs(d1) * np.exp(1j * 0)
        b1 = np.abs(b1) * np.exp(1j * 0)

        _, axs = plt.subplots(2, 6, figsize=(20, 7.5), subplot_kw={'projection': 'polar'})

        # Bright output
        axs[0,0].scatter(np.angle(b1), np.abs(b1), color="yellow", label='Input 1', alpha=0.5)
        axs[0,0].plot([0, np.angle(b1)], [0, np.abs(b1)], color="yellow", alpha=0.5)
        axs[0,0].scatter(np.angle(b2), np.abs(b2), color="green", label='Input 2', alpha=0.5)
        axs[0,0].plot([0, np.angle(b2)], [0, np.abs(b2)], color="green", alpha=0.5)
        axs[0,0].scatter(np.angle(b3), np.abs(b3), color="red", label='Input 3', alpha=0.5)
        axs[0,0].plot([0, np.angle(b3)], [0, np.abs(b3)], color="red", alpha=0.5)
        axs[0,0].scatter(np.angle(b4), np.abs(b4), color="blue", label='Input 4', alpha=0.5)
        axs[0,0].plot([0, np.angle(b4)], [0, np.abs(b4)], color="blue", alpha=0.5)
        axs[0,0].set_title('Bright output')

        for n in range(3):
            axs[0,n+1].scatter(np.angle(n1[n]), np.abs(n1[n]), color="yellow", label='Input 1', alpha=0.5)
            axs[0,n+1].plot([0, np.angle(n1[n])], [0, np.abs(n1[n])], color="yellow", alpha=0.5)
            axs[0,n+1].scatter(np.angle(n2[n]), np.abs(n2[n]), color="green", label='Input 2', alpha=0.5)
            axs[0,n+1].plot([0, np.angle(n2[n])], [0, np.abs(n2[n])], color="green", alpha=0.5)
            axs[0,n+1].scatter(np.angle(n3[n]), np.abs(n3[n]), color="red", label='Input 3', alpha=0.5)
            axs[0,n+1].plot([0, np.angle(n3[n])], [0, np.abs(n3[n])], color="red", alpha=0.5)
            axs[0,n+1].scatter(np.angle(n4[n]), np.abs(n4[n]), color="blue", label='Input 4', alpha=0.5)
            axs[0,n+1].plot([0, np.angle(n4[n])], [0, np.abs(n4[n])], color="blue", alpha=0.5)
            axs[0,n+1].set_title(f'Null output {n+1}')

        for d in range(6):
            axs[1,d].scatter(np.angle(d1[d]), np.abs(d1[d]), color="yellow", label='I1', alpha=0.5)
            axs[1,d].plot([0, np.angle(d1[d])], [0, np.abs(d1[d])], color="yellow", alpha=0.5)
            axs[1,d].scatter(np.angle(d2[d]), np.abs(d2[d]), color="green", label='I2', alpha=0.5)
            axs[1,d].plot([0, np.angle(d2[d])], [0, np.abs(d2[d])], color="green", alpha=0.5)
            axs[1,d].scatter(np.angle(d3[d]), np.abs(d3[d]), color="red", label='I3', alpha=0.5)
            axs[1,d].plot([0, np.angle(d3[d])], [0, np.abs(d3[d])], color="red", alpha=0.5)
            axs[1,d].scatter(np.angle(d4[d]), np.abs(d4[d]), color="blue", label='I4', alpha=0.5)
            axs[1,d].plot([0, np.angle(d4[d])], [0, np.abs(d4[d])], color="blue", alpha=0.5)
            axs[1,d].set_title(f'Dark output {d+1}')

        m = np.max(np.concatenate([
            np.abs(n1), np.abs(n2), np.abs(n3), np.abs(n4),
            np.abs(d1), np.abs(d2), np.abs(d3), np.abs(d4),
            np.array([np.abs(b1), np.abs(b2), np.abs(b3), np.abs(b4)])
        ]))

        for ax in axs.flatten():
            ax.set_ylim(0, m)

        axs[0, 4].axis("off")
        axs[0, 5].axis("off")

        axs[0,0].legend()

        if not plot:
            plot = BytesIO()
            plt.savefig(plot, format='png')
            plt.close()
            return plot.getvalue()
        plt.show()

    # Rebind outputs ----------------------------------------------------------

    def rebind_outputs(self, λ):
        """
        Correct the output order of the KernelNuller object. To do so, we successively obstruct two inputs and add a π/4 phase over one of the two remaining inputs. Doing so, 

        Parameters
        ----------
        - self: KernelNuller object
        - λ: Wavelength of the observation

        Returns
        -------
        - KernelNuller object
        """

        # Identify kernels (correct kernel swapping) ~~~~~~~~~~~~~~~~~~~~~~~~~~

        # E1 + E4 -> D1 & D2 should be dark (K1)
        ψ = np.zeros(4, dtype=complex)
        ψ[0] = ψ[3] = (1+0j) * np.sqrt(1/2)

        _, d, _ = self.propagate_fields(ψ=ψ, λ=λ)
        k1 = np.argsort((d * np.conj(d)).real)[:2]

        # E1 + E3 -> D3 & D4 should be dark (K2)
        ψ = np.zeros(4, dtype=complex)
        ψ[0] = ψ[2] = (1+0j) * np.sqrt(1/2)

        _, d, _ = self.propagate_fields(ψ=ψ, λ=λ)
        k2 = np.argsort((d * np.conj(d)).real)[:2]

        # E1 + E2 -> D5 & D6 should be dark (K3)
        ψ = np.zeros(4, dtype=complex)
        ψ[0] = ψ[1] = (1+0j) * np.sqrt(1/2)

        _, d, _ = self.propagate_fields(ψ=ψ, λ=λ)
        k3 = np.argsort((d * np.conj(d)).real)[:2]

        # Check kernel sign (correc kernel inversion) ~~~~~~~~~~~~~~~~~~~~~~~~~

        # E1 + E2*exp(-iπ/4) -> K1 and K2 should be positive
        ψ = np.zeros(4, dtype=complex)
        ψ[0] = ψ[1] = (1+0j) * np.sqrt(1/2)
        ψ[1] *= np.exp(- 1j * np.pi / 2)
        _, d, _ = self.propagate_fields(ψ=ψ, λ=λ)

        dk1 = d[k1]
        diff = np.abs(dk1[0] - dk1[1])
        if diff < 0:
            k1 = np.flip(k1)

        dk2 = d[k2]
        diff = np.abs(dk2[0] - dk2[1])
        if diff < 0:
            k2 = np.flip(k2)

        # E1 + E3*exp(-iπ/4) -> K3 should be positive
        ψ = np.zeros(4, dtype=complex)
        ψ[0] = ψ[1] = (1+0j) * np.sqrt(1/2)
        ψ[2] *= np.exp(- 1j * np.pi / 2)
        _, d, _ = self.propagate_fields(ψ=ψ, λ=λ)

        dk3 = d[k3]
        diff = np.abs(dk3[0] - dk3[1])
        if diff < 0:
            k3 = np.flip(k3)

        # Reconstruct the kernel order ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.output_order = np.concatenate([k1, k2, k3])

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
