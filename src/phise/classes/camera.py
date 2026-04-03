from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from .interferometer import Interferometer
import numpy as np
import astropy.units as u
import numba as nb
import math

class Camera:
    """Virtual camera used to simulate photon detection.

        Args:
            e (u.Quantity): Exposure time as an Astropy quantity in a
                time unit (e.g. ``1 * u.s``). If ``None``, a default of ``1 s``
                is used when possible.
            ideal (bool): If ``True``, the camera is considered ideal and
                returns the expected integer value without noise (truncated).
                If ``False``, Poisson counting noise is simulated.
            name (str): Human-readable name for the camera.
            ron (float): Read-Out Noise (electrons rms). Default: 0.0.
            dc (float): Dark current (electrons/pixel/s). Default: 0.0.
            fwc (float): Full-Well Capacity (electrons). Default: infinity.
            qe (float): Quantum Efficiency [0, 1]. Default: 1.0.

        Raises:
            TypeError: If ``ideal`` is not a boolean or ``name`` is not a
                string.
            ValueError: If ``e`` cannot be converted to a time unit.
        """

    __slots__ = ('_parent_interferometer', '_e', '_e_unit', '_name', '_ideal', '_ron', '_dc', '_fwc', '_qe')

    def __init__(self, e: u.Quantity = None, ideal: bool = False, name: str = 'Unnamed Camera', ron: float = 0.0, dc: float = 0.0, fwc: float = math.inf, qe: float = 1.0):
        self._parent_interferometer = None
        self.e = e if e is not None else 1 * u.s
        self.ideal = ideal
        self.name = name
        self.ron = ron
        self.dc = dc
        self.fwc = fwc
        self.qe = qe

    def __str__(self) -> str:
        res = f'Camera "{self.name}"\n'
        res += f'  Exposure time: {self.e:.2f}\n'
        res += f'  QE: {self.qe*100:.1f}% | RON: {self.ron} e- | DC: {self.dc} e-/s'
        return res

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def e(self) -> u.Quantity:
        """Camera exposure time in seconds.

        Returns:
            u.Quantity: Exposu re time expressed in seconds. Conversion is
                handled in the setter.
        """
        return (self._e * u.s).to(self._e_unit)

    @e.setter
    def e(self, e: u.Quantity):
        """Set the exposure time.

        Args:
            e (u.Quantity): Time quantity (e.g. ``0.5 * u.s``).

        Raises:
            TypeError: If ``e`` is not an ``astropy.units.Quantity``.
            ValueError: If the quantity cannot be converted to a time unit.
        """
        if not isinstance(e, u.Quantity):
            raise TypeError('e must be an astropy Quantity')
        try:
            e_val = e.to(u.s).value
        except u.UnitConversionError:
            raise ValueError('e must be in a time unit')
        if e_val <= 0:
            raise ValueError('e must be positive')
        self._e_unit = e.unit
        self._e = e_val

    @property
    def parent_interferometer(self) -> Interferometer:
        """Read-only reference to the parent interferometer."""
        return self._parent_interferometer

    @parent_interferometer.setter
    def parent_interferometer(self, _):
        raise ValueError('parent_interferometer is read-only')

    @property
    def ideal(self) -> bool:
        """Whether the camera is in ideal (noise-free) mode."""
        return self._ideal

    @ideal.setter
    def ideal(self, ideal: bool):
        if not isinstance(ideal, bool):
            raise TypeError('ideal must be a boolean')
        self._ideal = ideal

    @property
    def name(self) -> str:
        """Human-readable camera name."""
        return self._name

    @name.setter
    def name(self, name: str):
        if not isinstance(name, str):
            raise TypeError('name must be a string')
        self._name = name

    @property
    def ron(self) -> float:
        """Read-Out Noise in electrons root mean square."""
        return self._ron

    @ron.setter
    def ron(self, val: float):
        self._ron = float(val)

    @property
    def dc(self) -> float:
        """Dark current in electrons/pixel/second."""
        return self._dc

    @dc.setter
    def dc(self, val: float):
        self._dc = float(val)

    @property
    def fwc(self) -> float:
        """Full-Well Capacity (saturation limit in electrons)."""
        return self._fwc

    @fwc.setter
    def fwc(self, val: float):
        self._fwc = float(val)

    @property
    def qe(self) -> float:
        """Quantum Efficiency [0, 1]."""
        return self._qe

    @qe.setter
    def qe(self, val: float):
        self._qe = float(val)

    def acquire(self, ψ: np.ndarray[complex]) -> int:
        """Simulate acquisition of a pixel from complex electric fields.

        Computes the expected number of photo-electrons based on the field
        amplitude, exposure time, and quantum efficiency, and adds dark current.
        Detection is simulated depending on 'ideal' flag, with Poisson
        statistics and Read-Out Noise (RON). Includes saturation checks.

        Args:
            ψ (np.ndarray[complex]): 1D array (or broadcastable) of complex
                electric field amplitudes (units: s**(-1/2)).

        Returns:
            int: Number of detected electrons during the exposure.
        """
        try:
            return acquire_jit(ψ, self._e, ideal=self._ideal, qe=self._qe, dc=self._dc, ron=self._ron, fwc=self._fwc)
        except TypeError:
            return acquire_jit.__func__(ψ, self._e, ideal=self._ideal, qe=self._qe, dc=self._dc, ron=self._ron, fwc=self._fwc)

@staticmethod
@nb.njit()
def acquire_jit(ψ: np.ndarray[complex], e: float, ideal=False, qe=1.0, dc=0.0, ron=0.0, fwc=math.inf) -> int:
    """JIT-compiled version of ``acquire`` for Numba integration."""
    expected_pe = np.sum(np.abs(ψ) ** 2) * e * qe
    expected_dc = dc * e
    total_expected = expected_pe + expected_dc

    if ideal:
        detected_electrons = int(total_expected)
    else:
        if total_expected <= 2000000000.0:
            detected_electrons = np.random.poisson(total_expected)
        else:
            detected_electrons = int(total_expected + np.random.normal(0.0, math.sqrt(total_expected)))
        
        if ron > 0.0:
            detected_electrons = int(detected_electrons + np.random.normal(0.0, ron))

    if detected_electrons < 0:
        detected_electrons = 0
    if detected_electrons > fwc:
        detected_electrons = int(fwc)

    return detected_electrons