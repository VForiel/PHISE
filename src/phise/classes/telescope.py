import astropy.units as u
import numpy as np


class Telescope:
    """Telescope representation used by the interferometer.

    Args:
        a (u.Quantity): Collecting area as an Astropy quantity in a surface
            area unit (e.g., ``m**2``).
        r (u.Quantity): Relative position on the plane as an Astropy quantity
            in a length unit (e.g., ``m``). Must be a 2-vector (shape (2,)).
        name (str): Human-readable name for the telescope.
    Notes:
        Exceptions are raised by the setters if units are incorrect.
    """
    __slots__ = ('_parent_interferometer', '_a', '_a_unit', '_r', '_r_unit', '_name')

    def __init__(self, a: u.Quantity, r: u.Quantity, name: str = 'Unnamed Telescope'):
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
        """Collecting area (u.Quantity in m**2).

        Returns:
            u.Quantity: Mirror collecting area in square meters.
        """
        return (self._a * u.m**2).to(self._a_unit)

    @a.setter
    def a(self, a: u.Quantity):
        if not isinstance(a, u.Quantity):
            raise TypeError('a must be an astropy Quantity')
        try:
            new_a = a.to(u.m ** 2).value
        except u.UnitConversionError:
            raise ValueError('a must be in a surface area unit')
        self._a_unit = a.unit
        self._a = new_a
        if self.parent_interferometer is not None:
            self.parent_interferometer.parent_ctx._update_pf()

    @property
    def r(self) -> u.Quantity:
        """Relative telescope position on the plane (u.Quantity in m, shape (2,)).

        Returns:
            u.Quantity: 2-vector [x, y] position in meters.
        """
        return (self._r * u.m).to(self._r_unit)

    @r.setter
    def r(self, r: u.Quantity):
        if not isinstance(r, u.Quantity):
            raise TypeError('r must be an astropy Quantity')
        try:
            new_r = r.to(u.m).value
        except u.UnitConversionError:
            raise ValueError('r must be in a length unit')
        if r.shape != (2,):
            raise ValueError('r must have a shape of (2,)')
        self._r_unit = r.unit
        self._r = new_r
        if self.parent_interferometer is not None:
            self.parent_interferometer.parent_ctx._update_p()

    @property
    def name(self) -> str:
        """Telescope name.

        Returns:
            str: Readable name.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        if not isinstance(name, str):
            raise TypeError('name must be a string')
        self._name = name

    @property
    def parent_interferometer(self):
        """Reference to the parent interferometer (read-only).

        Returns:
            Any: Parent interferometer reference.
        """
        return self._parent_interferometer

    @parent_interferometer.setter
    def parent_interferometer(self, parent_interferometer):
        raise ValueError('parent_interferometer is read-only')


def get_VLTI_UTs() -> list[Telescope]:
    """Return the relative geometry of VLTI UTs.

    Returns:
        list[Telescope]: Four `Telescope` objects positioned according to the
            standard UT configuration.
    """
    r = np.array([
        [0, 0],
        [24.812, 50.837],
        [54.840, 86.518],
        [113.231, 64.334]
    ]) * u.m
    a = 4 * np.pi * (4 * u.m) ** 2
    return [Telescope(a=a, r=pos, name=f'UT {i + 1}') for (i, pos) in enumerate(r)]


def get_LIFE_telescopes() -> list[Telescope]:
    """Generate a telescope configuration for the LIFE concept.

    Returns:
        list[Telescope]: Four `Telescope` objects.
    """
    r = np.array([[0, 0], [1, 0], [0, 6], [1, 6]]) * 100 * u.m
    a = np.pi * (2 * u.m) ** 2
    return [Telescope(a=a, r=pos, name=f'LIFE telescope {i + 1}') for (i, pos) in enumerate(r)]