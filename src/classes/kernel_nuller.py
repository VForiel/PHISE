import numpy as np
import numba as nb
import astropy.units as u
from astropy import constants as const
import matplotlib.pyplot as plt
from io import BytesIO

from ..modules import mmi
from ..modules import phase
from .body import Body

class KernelNuller():
    def __init__(self, φ:np.ndarray[u.Quantity], σ:np.ndarray[u.Quantity]):
        """Kernel-Nuller object.

        Parameters
        ----------
        - φ: Array of 14 injected OPD
        - σ: Array of 14 intrasic OPD error
        """
        self._φ = φ
        self.σ = σ

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
        self._φ = φ

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

    # Electric fields propagation -------------------------------------------------

    def propagate_fields(
            self,
            ψ: np.ndarray[complex],
            λ: u.Quantity,
        ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], float]:
        """
        Simulate a 4 telescope Kernel-Nuller propagation using a numeric approach

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

        return propagate_fields_njit(ψ=ψ, φ=φ, σ=σ, λ=λ)

    # Observation -----------------------------------------------------------------

    def observe(
            self,
            ψ: np.ndarray[complex],
            λ: u.Quantity,
            f: float = None,
            Δt:u.Quantity = 1*u.s,
        ) -> np.ndarray[float]:
        """
        Simulate a 4 telescope Kernel-Nuller propagation using a numeric approach

        Parameters
        ----------
        - Ψ: Array of 4 input beams complex amplitudes
        - λ: Wavelength of the light
        - Δt: Exposure time 

        Returns
        -------
        - Array of 6 dark outputs intensities
        - Array of 3 kernels output intensities
        - Bright output intensity
        """
        φ = self.φ.to(λ.unit).value
        σ = self.σ.to(λ.unit).value
        dt = dt.to(u.s).value
        return observe_njit(ψ, φ, σ, λ.value, Δt)
    
    # Plotting --------------------------------------------------------------------

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

#==============================================================================
# Numba functions
#==============================================================================

# Electric fields propagation -------------------------------------------------

# @nb.njit()
def propagate_fields_njit(
        ψ: np.ndarray[complex],
        φ: np.ndarray[float],
        σ: np.ndarray[float],
        λ: float,
    ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], float]:
    """
    Simulate a 4 telescope Kernel-Nuller propagation using a numeric approach

    Parameters
    ----------
    - ψ: Array of 4 input signals complex amplitudes
    - φ: Array of 14 injected OPD (in wavelenght unit)
    - σ: Array of 14 intrasic OPD error (in wavelenght unit)
    - λ: Wavelength of the light

    Returns
    -------
    - Array of 3 null outputs electric fields
    - Array of 6 dark outputs electric fields
    - Bright output electric fields
    """

    print("i:", np.sum(ψ * np.conjugate(ψ)))

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

    nuller_output = np.array([*N3, *N4], dtype=np.complex128)
    print(nuller_output.shape, nuller_output)
    print("n:", np.sum(nuller_output * np.conjugate(nuller_output)))

    # Beam splitting
    R_inputs = np.array([N3[1], N3[1], N4[0], N4[0], N4[1], N4[1]]) * 1 / np.sqrt(2)

    # Last layer of phase shifters
    R_inputs = phase.shift_njit(R_inputs, φ[8:], λ)

    # Beam mixing
    R1_output = mmi.cross_recombiner_2x2(np.array([R_inputs[0], R_inputs[2]]))
    R2_output = mmi.cross_recombiner_2x2(np.array([R_inputs[1], R_inputs[4]]))
    R3_output = mmi.cross_recombiner_2x2(np.array([R_inputs[3], R_inputs[5]]))

    recombiner_output = np.array([*R1_output, *R2_output, *R3_output, bright], dtype=np.complex128)
    print(recombiner_output.shape, recombiner_output)
    print("r:", np.sum(recombiner_output * np.conjugate(recombiner_output)))

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

    return nulls, darks, bright

# Observation -----------------------------------------------------------------

# @nb.njit()
def observe_njit(
    ψ: np.ndarray[complex],
    φ: u.Quantity,
    σ: u.Quantity,
    λ: u.Quantity,
    Δt:float = 1,
) -> np.ndarray[float]:
    """
    Simulate a 4 telescope Kernel-Nuller propagation using a numeric approach

    Parameters
    ----------
    - ψ: Array of 4 input beams complex amplitudes (in photon/[Δt] unit)
    - φ: Array of 14 injected OPD
    - σ: Array of 14 intrasic OPD
    - λ: Wavelength of the light
    - Δt: Exposure time in seconds

    Returns
    -------
    - Array of 6 dark outputs intensities
    - Array of 3 kernels outputs intensities
    - Bright output intensity
    """

    _, d, b = propagate_fields_njit(ψ, φ, σ, λ)

    # Get intensities
    d = np.abs(d) ** 2
    b = np.abs(b) ** 2

    # Add photon noise
    for i in range(3):
        d[i] = np.random.poisson(np.floor(d[i] * Δt))
    print(np.floor(b * Δt))
    b = np.random.poisson(np.floor(b * Δt))

    # Create kernels
    k = np.array([d[0]-d[1], d[2]-d[3], d[4]-d[5]])

    return d, k, b