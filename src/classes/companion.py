"""Module generated docstring."""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from target import Target
from astropy import units as u

class Companion:
    """"Companion class.

Attributes
----------
(Automatically added placeholder.)
"""
    __slots__ = ('_parent_target', '_c', '_θ', '_α', '_name')

    def __init__(self, c: float, θ: u.Quantity, α: u.Quantity, name: str='Unnamed Companion'):
        """Any light source in the sky (unresolved).

        Parameters
        ----------
        - c: Contrast with respect to the host star
        - θ: Angular separation
        - α: Parallactic angle
        """
        self._parent_target = None
        self.θ = θ
        self.α = α
        self.c = c
        self.name = name

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        res = f'Companion "{self.name}"\n'
        res += f'  Contrast: {self.c:.2f}\n'
        res += f'  Angular separation: {self.θ:.2f}\n'
        res += f'  Parallactic angle: {self.α:.2f}'
        return res

    @property
    def c(self) -> float:
        """"c.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        return self._c

    @c.setter
    def c(self, c: float):
        """"c.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        if not isinstance(c, (int, float)):
            raise TypeError('c must be a float')
        if c < 0:
            raise ValueError('c must be positive')
        self._c = float(c)

    @property
    def θ(self) -> u.Quantity:
        """"θ.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        return self._θ

    @θ.setter
    def θ(self, θ: u.Quantity):
        """"θ.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        if not isinstance(θ, u.Quantity):
            raise TypeError('θ must be an astropy Quantity')
        try:
            θ = θ.to(u.mas)
        except u.UnitConversionError:
            raise ValueError('θ must be an angle')
        self._θ = θ

    @property
    def α(self) -> u.Quantity:
        """"α.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        return self._α

    @α.setter
    def α(self, α: u.Quantity):
        """"α.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        if not isinstance(α, u.Quantity):
            raise TypeError('α must be an astropy Quantity')
        try:
            α = α.to(u.rad)
        except u.UnitConversionError:
            raise ValueError('α must be an angle')
        self._α = α

    @property
    def parent_target(self) -> Target:
        """"parent_target.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        return self._parent_target

    @parent_target.setter
    def parent_target(self, target: Target):
        """"parent_target.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        raise ValueError('parent_target is read-only')

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