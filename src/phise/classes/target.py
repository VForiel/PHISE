"""Module generated docstring."""
import astropy.units as u
from copy import deepcopy as copy
from .companion import Companion

class Target:
    """"Target class.

Attributes
----------
(Automatically added placeholder.)
"""
    __slots__ = ('_parent_ctx', '_f', '_δ', '_companions', '_name')

    def __init__(self, f: u.Quantity, δ: u.Quantity, companions: list[Companion], name: str='Unnamed Target'):
        """
        A target star with a given magnitude and declination, and a list of companions.

        Parameters
        ----------
        f : `astropy.units.Quantity`
            Spectral flux of the star.
        δ : `astropy.units.Quantity`
            Declination of the star.
        companions : list of `Companion`
            List of Companion objects.
        name : str, optional
            Name of the scene (default is "Unnamed").
        """
        self._parent_ctx = None
        self.f = f
        self.δ = δ
        self.companions = copy(companions)
        for companion in self.companions:
            companion._parent_target = self
        self.name = name

    def __str__(self) -> str:
        res = f'Target "{self.name}"\n'
        res += f'  f: {self.f:.2e}\n'
        res += f'  δ: {self.δ:.2e}\n'
        res += f'  Companions:\n'
        lines = []
        for companion in self.companions:
            lines += str(companion).split('\n')
        res += f'    ' + '\n    '.join(lines)
        return res

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def f(self) -> u.Quantity:
        """"f.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        return self._f

    @f.setter
    def f(self, f: u.Quantity):
        """"f.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        if not isinstance(f, u.Quantity):
            raise TypeError('f must be an astropy Quantity')
        try:
            f = f.to(u.W * u.m ** (-2) * u.nm ** (-1))
        except u.UnitConversionError:
            raise ValueError('f must be in spectral flux units (equivalent to W/m^2/nm)')
        self._f = f
        if self.parent_ctx is not None:
            self.parent_ctx.update_photon_flux()

    @property
    def δ(self) -> u.Quantity:
        """"δ.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        return self._δ

    @δ.setter
    def δ(self, δ: u.Quantity):
        """"δ.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        if not isinstance(δ, u.Quantity):
            raise TypeError('δ must be an astropy Quantity')
        try:
            δ = δ.to(u.deg)
        except u.UnitConversionError:
            raise ValueError('δ must be in degrees')
        self._δ = δ
        if self.parent_ctx is not None:
            self.parent_ctx.project_telescopes_position()

    @property
    def companions(self) -> list[Companion]:
        """"companions.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        return self._companions

    @companions.setter
    def companions(self, companions: list[Companion]):
        """"companions.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        if not all((isinstance(companion, Companion) for companion in companions)):
            raise TypeError('`companions` must be a list of Companion objects.')
        try:
            companions = list(companions)
        except TypeError:
            raise TypeError('companions must be a list of Companion objects')
        self._companions = copy(companions)
        for companion in self._companions:
            companion._parent_target = self

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

    @property
    def parent_ctx(self) -> list:
        """"parent_ctx.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        return self._parent_ctx

    @parent_ctx.setter
    def parent_ctx(self, parent_ctx):
        """"parent_ctx.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        raise AttributeError('parent_ctx is read-only')