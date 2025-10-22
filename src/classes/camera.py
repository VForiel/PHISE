"""Module generated docstring."""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from .interferometer import Interferometer
import numpy as np
import astropy.units as u
import numba as nb
import math

class Camera:
    """"Camera class.

Attributes
----------
(Automatically added placeholder.)
"""
    __slots__ = ('_parent_interferometer', '_e', '_name', '_ideal')

    def __init__(self, e: Optional[u.Quantity]=None, ideal=False, name: str='Unnamed Camera'):
        """
        Initialize the camera object.

        Parameters
        ----------
        - e: Exposure time
        - ideal: Whether the camera is ideal (no noise) or not
        - name: Name of the camera
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
        """"e.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        return self._e

    @e.setter
    def e(self, e: u.Quantity):
        """"e.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
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
        """"parent_interferometer.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        return self._parent_interferometer

    @parent_interferometer.setter
    def parent_interferometer(self, _):
        """"parent_interferometer.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        raise ValueError('parent_interferometer is read-only')

    @property
    def ideal(self) -> bool:
        """"ideal.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        return self._ideal

    @ideal.setter
    def ideal(self, ideal: bool):
        """"ideal.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        if not isinstance(ideal, bool):
            raise TypeError('ideal must be a boolean')
        self._ideal = ideal

    @property
    def name(self) -> str:
        """"name.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        return self._name

    @name.setter
    def name(self, name: str):
        """"name.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        if not isinstance(name, str):
            raise TypeError('name must be a string')
        self._name = name

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
        expected_photons = np.sum(np.abs(ψ) ** 2) * self.e.to(u.s).value
        if self.ideal:
            detected_photons = int(expected_photons)
        elif expected_photons <= 2000000000.0:
            detected_photons = np.random.poisson(expected_photons)
        else:
            detected_photons = int(expected_photons + np.random.normal(0, math.sqrt(expected_photons)))
        return detected_photons