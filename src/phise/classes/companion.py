from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from target import Target
from astropy import units as u

class Companion:
    """Point-like astronomical companion.

    Args:
        c (float): Contrast relative to the host star (must be >= 0).
        ρ (u.Quantity): Angular separation (e.g., ``100 * u.mas``).
        θ (u.Quantity): Parallactic angle (e.g., ``0.1 * u.rad``).
        name (str, optional): Readable companion name.

    Raises:
        TypeError: If provided types are not as expected.
        ValueError: If invalid physical values are supplied (e.g., negative contrast).
    """

    __slots__ = ('_parent_target', '_c', '_ρ', '_ρ_unit', '_θ', '_θ_unit', '_name')

    def __init__(self, c: float, ρ: u.Quantity, θ: u.Quantity, name: str = 'Unnamed Companion'):
        self._parent_target = None
        self.ρ = ρ
        self.θ = θ
        self.c = c
        self.name = name

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        res = f'Companion "{self.name}"\n'
        res += f'  Contrast: {self.c:.2f}\n'
        res += f'  Angular separation: {self.ρ:.2f}\n'
        res += f'  Parallactic angle: {self.θ:.2f}'
        return res

    @property
    def c(self) -> float:
        """Companion contrast (dimensionless).

        Returns:
            float: Positive contrast value.
        """
        return self._c

    @c.setter
    def c(self, c: float):
        """Set the companion contrast.

        Args:
            c (float): Contrast (>= 0). Raises ``TypeError`` if not int/float.
        """
        if not isinstance(c, (int, float)):
            raise TypeError('c must be a float')
        if c < 0:
            raise ValueError('c must be positive')
        self._c = float(c)

    @property
    def ρ(self) -> u.Quantity:
        """Angular separation (u.Quantity in mas).

        Returns:
            u.Quantity: Separation in milliarcseconds (mas).
        """
        return (self._ρ * u.mas).to(self._ρ_unit)

    @ρ.setter
    def ρ(self, ρ: u.Quantity):
        """Set the angular separation.

        Args:
            ρ (u.Quantity): Angle quantity (e.g., ``100 * u.mas`` or ``0.1 * u.arcsec``).
        """
        if not isinstance(ρ, u.Quantity):
            raise TypeError('ρ must be an astropy Quantity')
        try:
            new_ρ = ρ.to(u.mas).value
        except u.UnitConversionError:
            raise ValueError('ρ must be an angle')
        self._ρ_unit = ρ.unit
        self._ρ = new_ρ

    @property
    def θ(self) -> u.Quantity:
        """Parallactic angle (u.Quantity in radians).

        Returns:
            u.Quantity: Angle in radians.
        """
        return (self._θ * u.rad).to(self._θ_unit)

    @θ.setter
    def θ(self, θ: u.Quantity):
        """Set the parallactic angle.

        Args:
            θ (u.Quantity): Angle quantity (e.g., ``0.1 * u.rad`` or ``10 * u.deg``).
        """
        if not isinstance(θ, u.Quantity):
            raise TypeError('θ must be an astropy Quantity')
        try:
            new_θ = θ.to(u.rad).value
        except u.UnitConversionError:
            raise ValueError('θ must be an angle')
        self._θ_unit = θ.unit
        self._θ = new_θ

    @property
    def parent_target(self) -> Target:
        """Read-only reference to the parent `Target` object.

        Any direct assignment attempts will raise; the relation is set by the parent.
        """
        return self._parent_target

    @parent_target.setter
    def parent_target(self, target: Target):
        """Setter is disabled; ``parent_target`` is read-only.

        Raises:
            ValueError: Always raised; property is read-only.
        """
        raise ValueError('parent_target is read-only')

    @property
    def name(self) -> str:
        """Readable companion name.

        Returns:
            str: Name of the companion.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Set the companion name.

        Args:
            name (str): Human-readable name.
        """
        if not isinstance(name, str):
            raise TypeError('name must be a string')
        self._name = name