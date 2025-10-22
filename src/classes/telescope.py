"""Module generated docstring."""
import astropy.units as u
import numpy as np

class Telescope:
    """"Telescope class.

Attributes
----------
(Automatically added placeholder.)
"""
    __slots__ = ('_parent_interferometer', '_a', '_r', '_name')

    def __init__(self, a: u.Quantity, r: u.Quantity, name: str='Unnamed Telescope'):
        """
        Initialize the telescope object.

        Parameters
        ----------
        - a: Area of the telescope
        - r: Relative position to the baseline center
        - name: Name of the telescope
        """
        self._parent_interferometer = None
        self.a = a
        self.r = r
        self.name = name

    def __str__(self) -> str:
        res = f'Telescope "{self.name}"\n'
        res += f'  Area: {self.a:.2e}\n'
        res += f"  Relative position: [{', '.join([f'{i:.2e}' for i in self.r.value])}] {self.r.unit}"
        return res.replace('e+00', '')

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def a(self) -> u.Quantity:
        """"a.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        return self._a

    @a.setter
    def a(self, a: u.Quantity):
        """"a.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        if not isinstance(a, u.Quantity):
            raise TypeError('a must be an astropy Quantity')
        try:
            a = a.to(u.m ** 2)
        except u.UnitConversionError:
            raise ValueError('a must be in a surface area unit')
        self._a = a
        if self.parent_interferometer is not None:
            self.parent_interferometer.parent_ctx.update_photon_flux()

    @property
    def r(self) -> u.Quantity:
        """"r.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        return self._r

    @r.setter
    def r(self, r: u.Quantity):
        """"r.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        if not isinstance(r, u.Quantity):
            raise TypeError('r must be an astropy Quantity')
        try:
            r = r.to(u.m)
        except u.UnitConversionError:
            raise ValueError('r must be in a length unit')
        if r.shape != (2,):
            raise ValueError('r must have a shape of (2,)')
        self._r = r
        if self.parent_interferometer is not None:
            self.parent_interferometer.parent_ctx.project_telescopes_position()

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
    def parent_interferometer(self):
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
    def parent_interferometer(self, parent_interferometer):
        """"parent_interferometer.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        raise ValueError('parent_interferometer is read-only')

def get_VLTI_UTs() -> list[Telescope]:
    """
    Get the relative position of the UTs, in meter.

    Returns
    -------
    - List containing the 4 VLTI UTs as Telescope objects.
    """
    r = np.array([[-70.4048732988764, -24.627602893919807], [-70.40465753243652, -24.627118902835786], [-70.40439460074228, -24.62681028261176], [-70.40384287956437, -24.627033500373024]])
    r -= r[0]
    earth_radius = 6378137 * u.m
    UTs_elevation = 2635 * u.m
    r = np.tan((r * u.deg).to(u.rad)) * (earth_radius + UTs_elevation)
    a = 4 * np.pi * (4 * u.m) ** 2
    return [Telescope(a=a, r=pos, name=f'UT {i + 1}') for (i, pos) in enumerate(r)]

def get_LIFE_telescopes() -> list[Telescope]:
    """
    Get the relative position of the LIFE telescopes, in meter.

    Returns
    -------
    - List containing the 4 LIFE telescopes as Telescope objects.
    """
    r = np.array([[0, 0], [1, 0], [0, 6], [1, 6]]) * 100 * u.m
    a = np.pi * (2 * u.m) ** 2
    return [Telescope(a=a, r=pos, name=f'LIFE telescope {i + 1}') for (i, pos) in enumerate(r)]