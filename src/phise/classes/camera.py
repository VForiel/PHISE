"""Virtual camera for interferometric simulations.

This module provides the `Camera` class, a minimal sensor that converts
complex electric fields (visibilities) into a detected photon count over an
exposure time. It is designed to be used within an `Interferometer` object
that acts as its parent.

Responsibilities
- Store the exposure time (`e`) as an Astropy quantity in seconds.
- Simulate photon detection with `acquire_pixel` from an array of complex
    electric fields.
- Support an `ideal` mode (no noise, truncated integer value) or a realistic
    mode (Poisson noise / Gaussian approximation for very large counts).

Example
    >>> from phise.classes.camera import Camera
    >>> import numpy as np
    >>> import astropy.units as u
    >>> cam = Camera(e=0.5 * u.s, ideal=False, name="CamSim")
    >>> psi = np.array([1+0j, 0.5+0.5j])  # complex electric fields
    >>> n = cam.acquire_pixel(psi)

Raises
- TypeError: If `e` is not an `astropy.units.Quantity` or if `ideal`/`name`
    are not of the expected types.
- ValueError: If `e` cannot be converted to a time unit.

Notes
- Inputs and outputs are intentionally minimal:
    - `ψ` (psi) is a numpy.ndarray of complex numbers where |ψ|^2 yields a flux in s^-1 per element.
    - `acquire_pixel` returns an integer equal to the number of detected photons
        during exposure `e`.
"""
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

    Encapsulates simple sensor parameters (exposure time, name, ideal mode)
    and provides a method to convert complex electric fields into detected
    photon counts over the exposure.
    """

    __slots__ = ('_parent_interferometer', '_e', '_name', '_ideal')

    def __init__(self, e: Optional[u.Quantity] = None, ideal: bool = False, name: str = 'Unnamed Camera'):
        """Initialize the camera.

        Args:
            e (Optional[u.Quantity]): Exposure time as an Astropy quantity in a
                time unit (e.g. ``1 * u.s``). If ``None``, a default of ``1 s``
                is used when possible.
            ideal (bool): If ``True``, the camera is considered ideal and
                returns the expected integer value without noise (truncated).
                If ``False``, Poisson counting noise is simulated.
            name (str): Human-readable name for the camera.

        Raises:
            TypeError: If ``ideal`` is not a boolean or ``name`` is not a
                string.
            ValueError: If ``e`` cannot be converted to a time unit.
        """
        self._parent_interferometer = None
        # avoid evaluating `1 * u.s` at import time which may fail when astropy is mocked
        if e is None:
            try:
                e = 1 * u.s
            except Exception:
                e = None
        self.e = e
        self.ideal = ideal
        self.name = name

    def __str__(self) -> str:
        res = f'Camera "{self.name}"\n'
        res += f'  Exposure time: {self.e:.2f}'
        return res

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def e(self) -> u.Quantity:
        """Camera exposure time in seconds.

        Returns:
            u.Quantity: Exposure time expressed in seconds. Conversion is
                handled in the setter.
        """
        return self._e

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
            e = e.to(u.s)
        except u.UnitConversionError:
            raise ValueError('e must be in a time unit')
        self._e = e

    @property
    def parent_interferometer(self) -> Interferometer:
        """Read-only reference to the parent interferometer.

        The setter is read-only; the attribute is set by the parent object
        (e.g., an instance of ``Interferometer``).
        """
        return self._parent_interferometer

    @parent_interferometer.setter
    def parent_interferometer(self, _):
        """Setter is disabled; ``parent_interferometer`` is read-only.

        Raises:
            ValueError: Always raised; property is read-only.
        """
        raise ValueError('parent_interferometer is read-only')

    @property
    def ideal(self) -> bool:
        """Whether the camera is in ideal (noise-free) mode.

        Returns:
            bool: ``True`` when detection noise is disabled; useful for
                deterministic tests.
        """
        return self._ideal

    @ideal.setter
    def ideal(self, ideal: bool):
        """Set the ideal mode for the camera.

        Args:
            ideal (bool): ``True`` for a noise-free sensor, ``False`` to
                simulate counting noise.

        Raises:
            TypeError: If ``ideal`` is not a boolean.
        """
        if not isinstance(ideal, bool):
            raise TypeError('ideal must be a boolean')
        self._ideal = ideal

    @property
    def name(self) -> str:
        """Human-readable camera name.

        Returns:
            str: Object name.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Set the camera name.

        Args:
            name (str): Human-readable name.

        Raises:
            TypeError: If ``name`` is not a string.
        """
        if not isinstance(name, str):
            raise TypeError('name must be a string')
        self._name = name

    def acquire_pixel(self, ψ: np.ndarray[complex]) -> int:
        """Simulate acquisition of a pixel from complex electric fields.

        Computes the expected number of photons as the sum of powers |ψ|^2
        multiplied by the exposure time ``e``. Detection is then simulated as:

        - If ``ideal``: return the truncated integer of the expectation
          (deterministic).
        - Else: draw from a Poisson law for reasonable expectations (<= 2e9),
          or use a Gaussian approximation for very large expectations to avoid
          performance issues.

        Args:
            ψ (np.ndarray[complex]): 1D array (or broadcastable) of complex
                electric field amplitudes (units: s**(-1/2)).

        Returns:
            int: Number of detected photons during the exposure.

        Notes:
            - The method returns an integer >= 0. The numeric threshold (2e9)
              is an empirical switch to the normal approximation when the
              Poisson draw becomes too costly.
            - For reproducibility, seed the RNG before calling (e.g.
              ``np.random.seed(...)``).
        """
        expected_photons = np.sum(np.abs(ψ) ** 2) * self.e.to(u.s).value
        if self.ideal:
            detected_photons = int(expected_photons)
        elif expected_photons <= 2000000000.0:
            detected_photons = np.random.poisson(expected_photons)
        else:
            detected_photons = int(expected_photons + np.random.normal(0, math.sqrt(expected_photons)))
        return detected_photons