import numpy as np
import numba as nb
import astropy.units as u
from typing import Tuple, Any, Optional
import matplotlib.pyplot as plt
try:
    plt.rcParams['image.origin'] = 'lower'
except Exception:
    pass
from io import BytesIO
from copy import deepcopy as copy
from ...modules import mmi
from ...modules import phase

from ..chip import Chip
from ...modules.phase import shift, bound

class PhaseList(np.ndarray):
    ...

#==============================================================================
# Numba-accelerated functions
#==============================================================================

@nb.njit()
def get_output_fields_jit(
        ψ: np.ndarray[complex],
        φ: np.ndarray[float],
        σ: np.ndarray[float],
        λ: float,
        λ0: float,
        output_order: np.ndarray[int]
    ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], float]:
    """Simulate a 4-telescope Kernel Nuller propagation (numeric approach).

    Note: Does not account for input attenuation and OPD.

    Args:
        ψ (np.ndarray[complex]): Array of 4 input complex amplitudes.
        φ (np.ndarray[float]): Array of 14 injected OPDs (wavelength units).
        σ (np.ndarray[float]): Array of 14 intrinsic OPD errors (wavelength units).
        λ (float): Wavelength of the light.
        λ0 (float): Reference wavelength (wavelength units).
        output_order (np.ndarray[int]): Order of the outputs.

    Returns:
        tuple: (nulls, darks, bright)
            - nulls: Array of 3 null outputs (complex fields)
            - darks: Array of 6 dark outputs (complex fields)
            - bright: Bright output (complex field)
    """
    # λ_ratio = λ0 / λ
    
    # Obtained experimentally
    Cin = np.array([[0.993235600471687+0.009237092175691658j, -0.021264580807434843+0.03769657190120298j, -0.0051583647234613155+-0.0042329905582810185j, 0.017463492523078383+-0.004980876913586356j],
        [-0.0346680758683959+-0.018530357766726683j, 0.9889093932007929+-0.061123465436237485j, 0.026738344890038627+-0.0014393856537241413j, 0.004404623948204579+0.019764651573309613j],
        [-0.028412984445477973+0.019923133158339813j, 0.03070602561442529+-0.012119194765006948j, 1.0579343864170707+-0.011527258095511929j, 0.039989544678528464+0.007601157088749202j],
        [0.018285951246696873+-0.019141658304574972j, -0.009439220467160905+-0.012771119165619722j, 0.03584699301231649+-0.016710419975799848j, 1.0238623657867116+-0.024683838470046862j]], dtype=np.complex128)

    M = np.array([[0.5+0j, 0.5+0j, 0.5+0j, 0.5+0j],
        [0.5+0j, 0+-0.5j, 0+0.5j, -0.5+0j],
        [0.5+0j, 0+0.5j, 0+-0.5j, -0.5+0j],
        [0.5+0j, -0.5+0j, -0.5+0j, 0.5+0j]], dtype=np.complex128)

    P = np.array([
        [np.exp(1j * (φ[0] + σ[0]) / λ0 * 2 * np.pi), 0, 0, 0],
        [0, np.exp(1j * (φ[1] + σ[1]) / λ0 * 2 * np.pi), 0, 0],
        [0, 0, np.exp(1j * (φ[2] + σ[2]) / λ0 * 2 * np.pi), 0],
        [0, 0, 0, np.exp(1j * (φ[3] + σ[3]) / λ0 * 2 * np.pi)]
    ], dtype=np.complex128)

    Cout = np.array([[0.9730778790933375+-0.056722375163116436j, -0.028797392063705185+-0.01948778967495159j, -0.002666088781798851+-0.003540266278734302j, -0.024288111942176616+-0.005162868250002706j],
        [-0.021892304669666688+0.021443098378544353j, 0.9767787260134642+-0.01383016643950051j, 0.015854522865303682+0.007823830803290452j, 0.0013032043159456922+-0.03342304110006464j],
        [0.0014192768381037528+-0.013512061286457603j, 0.01313567563123009+-0.0202219623179085j, 0.9582292019742786+-0.03906121603845227j, 0.0228181198166124+-0.0424606000937866j],
        [-0.03282391798177125+0.004483601864926152j, -0.009184126593575653+-0.015909090105478144j, 0.02563732784075978+0.009597828972331662j, 0.9967355720799685+-0.041485655747843714j]], dtype=np.complex128)

    # Override with ideal matrices for testing
    # Cin = np.eye(4, dtype=np.complex128)
    # Cout = np.eye(4, dtype=np.complex128)

    return Cout @ M @ P @ Cin @ ψ
    
@nb.njit()
def process_outputs_jit(out: np.ndarray[complex]) -> np.ndarray[float]:
    """Compute kernel outputs from outputs intensities.

    Args:
        darks (np.ndarray[complex]): Array of raw outputs (complex fields).
    """
    k = out[2] - out[3]
    return np.array([k], dtype=np.float64)

#==============================================================================
# N4x4_T8 class
#==============================================================================

class N4x4_T8(Chip):
    """Kernel nuller representation for 4 telescopes.

    Args:
        φ (u.Quantity): (4,) array of applied OPDs (length units).
        σ (u.Quantity): (4,) array of intrinsic OPD errors.
        λ0 (u.Quantity): Reference wavelength at which matrices are defined.
        output_order (np.ndarray[int] | None): Output ordering (6 elements)
            defining output pairs.
        input_attenuation (np.ndarray[float] | None): Attenuations on the
            4 optical inputs.
        input_opd (u.Quantity | None): Relative OPDs applied to the 4 inputs.
        name (str): Descriptive name.
    """
    __slots__ = ('_parent_interferometer', '_φ', '_σ', '_λ0', '_output_order', '_input_attenuation', '_input_opd', '_name', '_raw_output_labels', '_processed_output_labels', 'nb_raw_outputs', 'nb_processed_outputs')

    def __init__(
            self,
            φ: np.ndarray[u.Quantity],
            σ: np.ndarray[u.Quantity],
            λ0: u.Quantity,
            output_order:np.ndarray[int]=None,
            input_attenuation:np.ndarray[float]=None,
            input_opd:np.ndarray[u.Quantity]=None,
            name:str='Unnamed Kernel-Nuller'
        ):

        self._raw_output_labels = ['Bright', 'Null', 'Dark 1', 'Dark 2']
        self._processed_output_labels = ['Kernel 1']

        self.nb_inputs = 4
        self.nb_raw_outputs = 4
        self.nb_processed_outputs = 1

        self._parent_interferometer = None
        self.φ = φ
        self.σ = σ
        self.λ0 = λ0
        self.output_order = output_order if output_order is not None else np.array([0, 1, 2, 3])
        self.input_attenuation = input_attenuation if input_attenuation is not None else np.array([1.0, 1.0, 1.0, 1.0])
        self.input_opd = input_opd if input_opd is not None else np.zeros(4) * u.m
        self.name = name

        super().__init__()

    #==========================================================================
    # Attributes
    #==========================================================================

    # Phase shifters ----------------------------------------------------------

    @property
    def φ(self):
        """Applied OPD/phase per nuller element.

        Returns:
            u.Quantity: Shape (4,) in length units (e.g., meters).
        """
        return self._φ

    @φ.setter
    def φ(self, φ: np.ndarray[u.Quantity]):
        """Set applied OPDs.

        Args:
            φ (u.Quantity): Shape (4,) in a length unit.

        Raises:
            ValueError: If not a Quantity, not in length units, wrong shape,
                or contains negative values.
        """
        if type(φ) != u.Quantity:
            raise ValueError('φ must be a Quantity')
        try:
            φ.to(u.m)
        except u.UnitConversionError:
            raise ValueError('φ must be in a distance unit')
        if φ.shape != (4,):
            raise ValueError('φ must have a shape of (4,)')
        if np.any(φ < 0):
            raise ValueError('φ must be positive')
        self._φ = φ

    # Perturbations -----------------------------------------------------------

    @property
    def σ(self):
        """Intrinsic OPD errors of the nuller.

        Returns:
            u.Quantity: Shape (4,) in same unit as ``φ``.
        """
        return self._σ

    @σ.setter
    def σ(self, σ: np.ndarray[u.Quantity]):
        """Set intrinsic OPD errors.

        Args:
            σ (u.Quantity): Shape (4,) in a length unit.

        Raises:
            ValueError: If not a Quantity, not in length units, or wrong shape.
        """
        if type(σ) != u.Quantity:
            raise ValueError('σ must be a Quantity')
        try:
            σ.to(u.m)
        except u.UnitConversionError:
            raise ValueError('σ must be in a distance unit')
        if σ.shape != (4,):
            raise ValueError('σ must have a shape of (4,)')
        self._σ = σ

    # Design wavelength -------------------------------------------------------

    @property
    def λ0(self):
        """Reference wavelength of the model.

        Returns:
            u.Quantity: Reference wavelength (e.g., meters).
        """
        return self._λ0

    @λ0.setter
    def λ0(self, λ0: u.Quantity):
        """Set reference wavelength.

        Args:
            λ0 (u.Quantity): Wavelength in a convertible length unit.

        Raises:
            TypeError: If not an ``astropy.units.Quantity``.
            ValueError: If not convertible to a length unit.
        """
        if not isinstance(λ0, u.Quantity):
            raise TypeError('λ0 must be an astropy Quantity')
        try:
            λ0 = λ0.to(u.m)
        except u.UnitConversionError:
            raise ValueError('λ0 must be in a distance unit')
        self._λ0 = λ0

    # Output ordering ---------------------------------------------------------

    @property
    def output_order(self):
        """Output order of the nuller.

        Returns:
            np.ndarray[int]: Length-4 array describing the output order and
                pair structure.
        """
        return self._output_order

    @output_order.setter
    def output_order(self, output_order: np.ndarray[int]):
        """Set output order.

        Args:
            output_order (np.ndarray[int]): Permutation of [0..3] with valid
                pair structure.

        Raises:
            ValueError: If not an integer array, wrong shape, not a permutation
                of 0..3, or invalid pair configuration.
        """
        try:
            output_order = np.array(output_order, dtype=int)
        except:
            raise ValueError(f'output_order must be an array of integers, not {type(output_order)}')
        if output_order.shape != (self.nb_raw_outputs,):
            raise ValueError(f'output_order must have a shape of ({self.nb_raw_outputs},), not {output_order.shape}')
        if not np.all(np.sort(output_order) == np.arange(self.nb_raw_outputs)):
            raise ValueError(f'output_order must contain all the integers from 0 to {self.nb_raw_outputs - 1}, not {output_order}')
        
        # Specific criteria for valid output pairs
        if  not ((output_order == np.array([0, 1, 2, 3], dtype=int)).all()
              or (output_order == np.array([1, 2, 3, 0], dtype=int)).all()
              or (output_order == np.array([2, 3, 0, 1], dtype=int)).all()
              or (output_order == np.array([3, 0, 1, 2], dtype=int)).all()):
            raise ValueError(f'output_order must be a valid permutation (a cyclic permutation of [0, 1, 2, 3]), not {output_order}')
        
        self._output_order = output_order

    def rebind_outputs(self, λ):
        """Correct output ordering of the SuperKN object.

        Successively obstruct two inputs and add a π/4 phase over one of the two
        remaining inputs to determine output pairing and ordering.

        Args:
            λ (u.Quantity): Observation wavelength.

        Returns:
            None: Updates ``self.output_order`` in place.
        """
        # Warning not implemented
        print("Warning: rebind_outputs is not implemented")
        pass

    # Input properties --------------------------------------------------------

    @property
    def input_attenuation(self):
        """Input attenuations.

        Returns:
            np.ndarray[float]: Length-4 multiplicative attenuation factors.
        """
        return self._input_attenuation

    @input_attenuation.setter
    def input_attenuation(self, input_attenuation: np.ndarray[float]):
        """Set input attenuations.

        Args:
            input_attenuation (np.ndarray[float]): Length-4 attenuation factors.

        Raises:
            ValueError: If not convertible to float array or wrong shape.
        """
        try:
            input_attenuation = np.array(input_attenuation, dtype=float)
        except:
            raise ValueError(f'input_attenuation must be an array of floats, not {type(input_attenuation)}')
        if input_attenuation.shape != (4,):
            raise ValueError(f'input_attenuation must have a shape of (4,), not {input_attenuation.shape}')
        self._input_attenuation = input_attenuation

    @property
    def input_opd(self):
        """Relative OPD applied on each input.

        Returns:
            u.Quantity: Shape (4,) in length units.
        """
        return self._input_opd

    @input_opd.setter
    def input_opd(self, input_opd: np.ndarray[u.Quantity]):
        """Set input OPDs.

        Args:
            input_opd (u.Quantity): Shape (4,) in a length unit.

        Raises:
            ValueError: If not a Quantity, not in length units, or wrong shape.
        """
        if type(input_opd) != u.Quantity:
            raise ValueError('input_opd must be a Quantity')
        try:
            input_opd.to(u.m)
        except u.UnitConversionError:
            raise ValueError('input_opd must be in a distance unit')
        if input_opd.shape != (4,):
            raise ValueError('input_opd must have a shape of (4,)')
        self._input_opd = input_opd

    # Name --------------------------------------------------------------------

    @property
    def name(self):
        """Descriptive instance name.

        Returns:
            str: Kernel nuller name.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Set instance name.

        Args:
            name (str): Readable name.

        Raises:
            ValueError: If not a string.
        """
        if not isinstance(name, str):
            raise ValueError('name must be a string')
        self._name = name

    def __str__(self) -> str:
        res = f'Kernel-Nuller "{self.name}"\n'
        res += f"  φ: [{', '.join([f'{i:.2e}' for i in self.φ.value])}] {self.φ.unit}\n"
        res += f"  σ: [{', '.join([f'{i:.2e}' for i in self.σ.value])}] {self.σ.unit}\n"
        res += f"  Output order: [{', '.join([f'{i}' for i in self.output_order])}]\n"
        res += f"  Input attenuation: [{', '.join([f'{i:.2e}' for i in self.input_attenuation])}]\n"
        res += f"  Input OPD: [{', '.join([f'{i:.2e}' for i in self.input_opd.value])}] {self.input_opd.unit}"
        return res.replace('e+00', '')

    def __repr__(self) -> str:
        return self.__str__()
    
    # Parent interferometer ---------------------------------------------------

    @property
    def parent_interferometer(self):
        """Parent interferometer associated with this kernel nuller.

        Read-only property set during association with an Interferometer object.
        """
        return self._parent_interferometer

    @parent_interferometer.setter
    def parent_interferometer(self, parent_interferometer):
        """Setter is disabled; ``parent_interferometer`` is read-only.

        Raises:
            ValueError: Always raised; property is read-only.
        """
        raise ValueError('parent_interferometer is read-only')

    # Shifters role -----------------------------------------------------------

    @property
    def bright_shifters_indices(self) -> list[int]:
        """Indices of shifters primarily controlling bright outputs."""
        return [0, 1, 2, 3]

    @property
    def kernel_shifters_indices(self) -> list[int]:
        """Indices of shifters primarily controlling kernel null depths."""
        return []
    
    #==========================================================================
    # Methods
    #==========================================================================

    # Wave propagation --------------------------------------------------------

    @property
    def get_output_fields_jit(self):
        """Return the @njit function for computing N4x4_T8 output fields."""
        return get_output_fields_jit

    @property
    def process_outputs_jit(self):
        """Return the @njit function for processing N4x4_T8 outputs."""
        return process_outputs_jit

    def get_output_fields(self, ψ: np.ndarray[complex], λ: u.Quantity, φ: Optional[u.Quantity]=None, σ: Optional[u.Quantity]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Propagate input fields through the kernel nuller.
        Args:
            ψ (np.ndarray[complex]): Input complex fields for the 4 channels (shape (4,)).
            λ (u.Quantity): Wavelength for propagation.
            φ (Optional[u.Quantity]): Override for phase shifters OPDs.
            σ (Optional[u.Quantity]): Override for intrinsic OPD errors.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, float]: Output complex
            fields (shape (4,)).
        """
        if φ is None:
            φ_val = self.φ.to(λ.unit).value
        else:
            φ_val = φ.to(λ.unit).value

        if σ is None:
            σ_val = self.σ.to(λ.unit).value
        else:
            σ_val = σ.to(λ.unit).value

        λ0 = self.λ0.to(λ.unit).value
        ψ = ψ.copy()
        ψ *= self.input_attenuation
        
        ψ *= np.exp(-1j * 2 * np.pi * self.input_opd.to(λ.unit).value / λ.value)

        return get_output_fields_jit(ψ=ψ.astype(np.complex128), φ=φ_val, σ=σ_val, λ=λ.value, λ0=λ0, output_order=self.output_order)
    
    def expected_outputs(self, ψ: np.ndarray[complex]) -> tuple[float, np.ndarray[float], np.ndarray[float]]:
        """
        Compute expected outputs from input fields using analytical model.
        
        Args:
            ψ (np.ndarray[complex]): Input complex fields for the 4 channels (shape (4,)).
        
        Returns:
            tuple:
                - bright (float): Bright output intensity.
                - darks (np.ndarray[float]): Dark outputs intensities (shape (6,)).
                - kernels (np.ndarray[float]): Kernel outputs intensities (shape (3,)).
        """
        return expected_outputs_jit(ψ)

    def process_outputs(self, out: np.ndarray[float]) -> np.ndarray[float]:
        """
        Compute processed kernel outputs from raw output intensities.
        Args:
            out (np.ndarray[float]): Raw output intensities (shape (7,)).
        Returns:
            np.ndarray[float]: Processed kernel outputs (shape (4,)).
        """
        return process_outputs_jit(out)
    
    # Plotting ----------------------------------------------------------------

    def plot_output_phase(self, λ: u.Quantity, ψ: Optional[np.ndarray]=None, plot: bool = True, n_cols: Optional[int] = None, ref_input1: bool = True) -> Optional[Any]:
        """Plot output phases and amplitudes of the nuller.

        Computes output responses for each isolated input and plots the phase
        and amplitude of null, dark, and bright outputs on polar diagrams.

        Args:
            λ (u.Quantity): Wavelength for the simulation.
            ψ (Optional[np.ndarray]): Input complex amplitudes (default [0.5,...]).
            plot (bool): If ``True``, display the figure; if ``False``, return the image bytes.
            n_cols (Optional[int]): Number of columns for the plot grid. If None, all plots are on a single row.
            ref_input1 (bool): If ``True``, use Input 1 (Bright output) as global phase reference (set to 0°).
        """
        if ψ is None:
            ψ = np.array([0.5 + 0j, 0.5 + 0j, 0.5 + 0j, 0.5 + 0j])
        ψ1 = np.array([ψ[0], 0, 0, 0])
        ψ2 = np.array([0, ψ[1], 0, 0])
        ψ3 = np.array([0, 0, ψ[2], 0])
        ψ4 = np.array([0, 0, 0, ψ[3]])

        out1 = self.get_output_fields(ψ1, λ)
        out2 = self.get_output_fields(ψ2, λ)
        out3 = self.get_output_fields(ψ3, λ)
        out4 = self.get_output_fields(ψ4, λ)

        # Global Phase Reference: Input 1 (Bright)
        if ref_input1 and len(out1) > 0:
             ref_phase = np.angle(out1[0])
             phasor = np.exp(-1j * ref_phase)
             out1 = out1 * phasor
             out2 = out2 * phasor
             out3 = out3 * phasor
             out4 = out4 * phasor
        
        n_out = len(out1)
        outs = np.array([out1, out2, out3, out4])

        # Grid layout configuration
        if n_cols is None:
            n_cols = n_out
        
        n_rows = int(np.ceil(n_out / n_cols))
        
        _, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), subplot_kw={'projection': 'polar'})
        
        # Flatten for easy iteration
        if n_out == 1:
            axs = [axs]
        else:
            axs = np.atleast_1d(axs).flatten()

        m = np.max(np.abs(outs))

        for i in range(len(axs)):
            ax = axs[i]
            if i < n_out:
                colors = ['gold', 'forestgreen', 'red', 'blue'] # Adjusted for visibility
                for j, out in enumerate(outs):
                    val = out[i]
                    # Vector arrow
                    ax.annotate(
                        "",
                        xy=(np.angle(val), np.abs(val)),
                        xytext=(0, 0),
                        textcoords='data',
                        arrowprops=dict(arrowstyle="->", color=colors[j], lw=2.0, alpha=0.8)
                    )
                    # Marker (tip) - mainly for legend
                    ax.scatter(np.angle(val), np.abs(val), color=colors[j], label=f'Input {j + 1}', s=20, alpha=1.0)
                
                ax.set_title(f'{self._raw_output_labels[i]} output')
                ax.set_ylim(0, m * 1.1)
            else:
                ax.axis('off')
        
        # Legend positioning depends on layout
        if n_rows > 1:
            axs[0].legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize='small')
        else:
            axs[0].legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize='small')
            
        if not plot:
            plot = BytesIO()
            plt.savefig(plot, format='png')
            plt.close()
            return plot.getvalue()
        plt.show()

    # Null calibration --------------------------------------------------------

    def calibrate(self, method='Hooke&Jeeves', verbose:bool=False, plot=False, input_fields:np.ndarray=None, hooke_jeeves_metric=None, β:float=0.5):
        if method.lower() == 'hooke&jeeves':
            return self._calibrate_hooke_jeeves(verbose=verbose, plot=plot, input_fields=input_fields, metric=hooke_jeeves_metric, β=β)
        else:
            raise ValueError(f'Unknown calibration method: {method}. Supported methods: Hooke&Jeeves')

    def _calibrate_hooke_jeeves(
        self,
        β = 0.5,
        ε:u.Quantity = None,
        verbose: bool = False,
        plot: bool = False,
        input_fields: np.ndarray = None,
        metric = None,
    ) -> dict:
        """
        Optimize phase shifter offsets using the analytical model to maximize nulling performance.
        Mimics null_calibration_gen but uses predict_output and scipy.optimize.
        Default metric: Null-Depth = sum(null outputs)
        Outputs are assumed to be: 0=Bright, 1,2,3=Nulls (Standard Arch6)
        """

        ctx = self.parent_interferometer.parent_ctx

        # Handle defaults
        if metric is None:
            def metric(outs):
                outs[outs <= 1] = 1  # Avoid log(0) issues
                return np.sum(outs[1:4])

        if input_fields is None:
            input_fields = np.ones(4, dtype=complex)

        if ε is None:
            ε = 1e-6 * ctx.interferometer.λ.unit  
        

        # Reduce observation window to one camera acquisition (static assumption)
        ctx.Δh = ctx.interferometer.camera.e.to(u.hour).value * u.hourangle

        # History of the optimization
        metric_history = []
        depths_history = []
        shifters_history = []

        ctx.interferometer.chip.φ = np.zeros(4) * ctx.interferometer.λ

        Δφ = ctx.interferometer.λ / 4
        while Δφ > ε:

            if verbose:
                print(f"--- New iteration --- Δφ={Δφ:.2e}")

            for i in range(4):
                log = ""

                # Getting observation with different phase shifts
                ctx.interferometer.chip.φ[i] += Δφ
                ctx.interferometer.chip.φ = ctx.interferometer.chip.φ % ctx.interferometer.λ
                outs_pos = ctx.observe()

                ctx.interferometer.chip.φ[i] -= 2 * Δφ
                ctx.interferometer.chip.φ = ctx.interferometer.chip.φ % ctx.interferometer.λ
                outs_neg = ctx.observe()

                ctx.interferometer.chip.φ[i] += Δφ
                ctx.interferometer.chip.φ = ctx.interferometer.chip.φ % ctx.interferometer.λ
                outs_old = ctx.observe()

                m_pos = metric(outs_pos)
                m_neg = metric(outs_neg)
                m_old = metric(outs_old)

                # Save the history
                metric_history.append(m_old)
                depths_history.append(np.abs(outs_old[1:]) / outs_old[0])
                shifters_history.append(np.copy(ctx.interferometer.chip.φ.value / ctx.interferometer.λ.value * 2 * np.pi))

                # Minimize the metric
                log += f"Shift {i} Metric: {m_neg:.2e} | {m_old:.2e} | {m_pos:.2e} -> "

                if m_pos < m_old and m_pos < m_neg:
                    log += " + "
                    ctx.interferometer.chip.φ[i] += Δφ
                elif m_neg < m_old and m_neg < m_pos:
                    log += " - "
                    ctx.interferometer.chip.φ[i] -= Δφ
                else:
                    log += " = "

                if verbose:
                    print(log)

            Δφ *= β

        metric_history = np.array(metric_history)
        depths_history = np.array(depths_history)
        shifters_history = np.array(shifters_history)

        fig = None
        if plot:

            fig, axs = plt.subplots(3, 1, constrained_layout=True)#, figsize=(8, 12))

            axs[0].plot(metric_history)
            axs[0].set_xlabel("Iterations")
            axs[0].set_ylabel("Metric")
            axs[0].set_yscale("log")
            axs[0].set_title("Performance of the Kernel-Nuller")

            for i in range(depths_history.shape[1]):
                axs[1].plot(depths_history[:, i], label=f"N{i+1}")
            axs[1].plot(np.mean(depths_history, axis=1), label="Mean", color='black', linestyle='--')
            axs[1].set_xlabel("Iterations")
            axs[1].set_ylabel("Null depth")
            axs[1].set_yscale("log")
            axs[1].set_title("Convergence of the Null depth")
            axs[1].legend(loc='upper right')

            for i in range(shifters_history.shape[1]):
                axs[2].plot(shifters_history[:, i], label=f"Shifter {i+1}")
            axs[2].set_xlabel("Iterations")
            axs[2].set_ylabel("Phase shift")
            axs[2].set_yscale("linear")
            axs[2].set_title("Convergence of the phase shifters")

        return {
            "metric": np.array(metric_history),
            "depths": np.array(depths_history),
            "shifters": np.array(shifters_history),
            "figure": fig
        }