"""Target star and companions model."""
import astropy.units as u
from copy import deepcopy as copy
from .companion import Companion

class Target:
    """Target star with declination, spectral flux, and companions."""
    __slots__ = ('_parent_ctx', '_f', '_δ', '_companions', '_name')

    def __init__(self, f: u.Quantity, δ: u.Quantity, companions: list[Companion], name: str='Unnamed Target'):
        """Create a target star.

        Args:
            f (u.Quantity): Spectral flux of the star.
            δ (u.Quantity): Declination of the star.
            companions (list[Companion]): List of companions.
            name (str, optional): Target name (default: "Unnamed Target").
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
        """Spectral flux (W m^-2 nm^-1).

        Returns:
            u.Quantity: Spectral flux density converted to W/m^2/nm when set.
        """
        return self._f

    @f.setter
    def f(self, f: u.Quantity):
        """Set spectral flux.

        Args:
            f (u.Quantity): Flux in a convertible spectral density unit.

        Raises:
            TypeError: If not an ``astropy.units.Quantity``.
            ValueError: If not convertible to W/m^2/nm.
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
        """Declination (degrees).

        Returns:
            u.Quantity: Declination in degrees.
        """
        return self._δ

    @δ.setter
    def δ(self, δ: u.Quantity):
        """Set declination.

        Args:
            δ (u.Quantity): Declination in a convertible angular unit.

        Raises:
            TypeError: If not an ``astropy.units.Quantity``.
            ValueError: If not convertible to degrees.
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
        """List of companions.

        Returns:
            list[Companion]: Managed list of companion objects.
        """
        return self._companions

    @companions.setter
    def companions(self, companions: list[Companion]):
        """Set companions.

        Args:
            companions (list[Companion]): List of ``Companion`` objects.

        Raises:
            TypeError: If the list contains non-``Companion`` items or is not iterable.
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
        """Target name.

        Returns:
            str: Readable target name.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Set target name.

        Args:
            name (str): Readable name.

        Raises:
            TypeError: If not a string.
        """
        if not isinstance(name, str):
            raise TypeError('name must be a string')
        self._name = name

    @property
    def parent_ctx(self) -> list:
        """Parent observing context (read-only).

        Returns:
            Any: Parent context reference or ``None``.
        """
        return self._parent_ctx

    @parent_ctx.setter
    def parent_ctx(self, parent_ctx):
        """Setter is disabled; ``parent_ctx`` is read-only.

        Raises:
            AttributeError: Always raised.
        """
        raise AttributeError('parent_ctx is read-only')