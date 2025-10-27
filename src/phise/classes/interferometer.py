from astropy import units as u
from copy import deepcopy as copy
from .chip import SuperKN
from .telescope import Telescope
from .camera import Camera

class Interferometer:
    """Interferometer representation.

    Provides global instrument state and utility properties to keep the
    observing context in sync (e.g., recompute projected telescope positions
    or update photon flux when certain parameters change).

    Args:
        l (u.Quantity): Array latitude (deg).
        λ (u.Quantity): Central wavelength (nm).
        Δλ (u.Quantity): Bandwidth (nm).
        fov (u.Quantity): Field of view (mas).
        η (float): Global optical efficiency (0..1).
        telescopes (list[Telescope]): Telescopes defining the geometry.
        kn (SuperKN): Kernel nuller configuration.
        camera (Camera): Associated camera.
        name (str, optional): Instrument name.
    """
    __slots__ = ('_parent_ctx', '_l', '_λ', '_Δλ', '_fov', '_η', '_telescopes', '_kn', '_camera', '_name')

    def __init__(self, l: u.Quantity, λ: u.Quantity, Δλ: u.Quantity, fov: u.Quantity, η: float, telescopes: list[Telescope], kn: SuperKN, camera: Camera, name: str='Unnamed Interferometer'):
        self._parent_ctx = None
        self.l = l
        self.λ = λ
        self.Δλ = Δλ
        self.fov = fov
        self.η = η
        self.telescopes = copy(telescopes)
        for telescope in self.telescopes:
            telescope._parent_interferometer = self
        self.kn = copy(kn)
        self.kn._parent_interferometer = self
        self.camera = copy(camera)
        self.camera._parent_interferometer = self
        self.name = name

    def __str__(self) -> str:
        res = f'Interferometer "{self.name}"\n'
        res += f'  Latitude: {self.l:.2f}\n'
        res += f'  Central wavelength: {self.λ:.2f}\n'
        res += f'  Bandwidth: {self.Δλ:.2f}\n'
        res += f'  Field of view: {self.fov:.2f}\n'
        res += f'  Telescopes:\n'
        lines = []
        for telescope in self.telescopes:
            lines += str(telescope).split('\n')
        res += f'    ' + '\n    '.join(lines) + '\n'
        res += f'  ' + '\n  '.join(str(self.kn).split('\n')) + '\n'
        res += f'  ' + '\n  '.join(str(self.camera).split('\n'))
        return res

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def l(self) -> u.Quantity:
        """Array latitude (degrees).

        Returns:
            u.Quantity: Latitude in degrees. Updating this triggers projected
                telescope positions recomputation when a parent context exists.
        """
        return self._l

    @l.setter
    def l(self, l: u.Quantity):
        """Set array latitude.

        Args:
            l (u.Quantity): Latitude in a convertible angular unit.

        Raises:
            TypeError: If not an ``astropy.units.Quantity``.
            ValueError: If not convertible to degrees.
        """
        if not isinstance(l, u.Quantity):
            raise TypeError('l must be an astropy Quantity')
        try:
            l = l.to(u.deg)
        except u.UnitConversionError:
            raise ValueError('l must be in degrees')
        self._l = l
        if self.parent_ctx is not None:
            self.parent_ctx.project_telescopes_position()

    @property
    def λ(self) -> u.Quantity:
        """Central wavelength (nm).

        Returns:
            u.Quantity: Central wavelength in nm. Updating triggers the
                context photon flux recomputation when a parent context exists.
        """
        return self._λ

    @λ.setter
    def λ(self, λ: u.Quantity):
        """Set central wavelength.

        Args:
            λ (u.Quantity): Wavelength in a convertible unit.

        Raises:
            TypeError: If not an ``astropy.units.Quantity``.
            ValueError: If not convertible to nanometers.
        """
        if not isinstance(λ, u.Quantity):
            raise TypeError('λ must be an astropy Quantity')
        try:
            λ = λ.to(u.nm)
        except u.UnitConversionError:
            raise ValueError('λ must be in nanometers')
        self._λ = λ
        if self.parent_ctx is not None:
            self.parent_ctx.update_photon_flux()

    @property
    def Δλ(self) -> u.Quantity:
        """Bandwidth (nm).

        Returns:
            u.Quantity: Positive bandwidth expressed in nanometers.
        """
        return self._Δλ

    @Δλ.setter
    def Δλ(self, Δλ: u.Quantity):
        """Set bandwidth.

        Args:
            Δλ (u.Quantity): Bandwidth in a convertible unit.

        Raises:
            TypeError: If not an ``astropy.units.Quantity``.
            ValueError: If not convertible to nanometers or non-positive.
        """
        if not isinstance(Δλ, u.Quantity):
            raise TypeError('Δλ must be an astropy Quantity')
        try:
            Δλ = Δλ.to(u.nm)
        except u.UnitConversionError:
            raise ValueError('Δλ must be in nanometers')
        if Δλ <= 0 * u.nm:
            raise ValueError('Δλ must be positive')
        if self.parent_ctx is not None:
            self.parent_ctx.update_photon_flux()
        self._Δλ = Δλ

    @property
    def fov(self) -> u.Quantity:
        """Field of view (typically in mas).

        Returns:
            u.Quantity: Field of view in milliarcseconds.
        """
        return self._fov

    @fov.setter
    def fov(self, fov: u.Quantity):
        """Set field of view.

        Args:
            fov (u.Quantity): FOV in a convertible angular unit.

        Raises:
            TypeError: If not an ``astropy.units.Quantity``.
            ValueError: If not convertible to milliarcseconds.
        """
        if not isinstance(fov, u.Quantity):
            raise TypeError('fov must be an astropy Quantity')
        try:
            fov = fov.to(u.mas)
        except u.UnitConversionError:
            raise ValueError('fov must be in milliarcseconds')
        self._fov = fov

    @property
    def telescopes(self) -> list[Telescope]:
        """List of `Telescope` objects constituting the array.

        Returns:
            list[Telescope]: Managed list of telescopes.
        """
        return self._telescopes

    @telescopes.setter
    def telescopes(self, telescopes: list[Telescope]):
        """Set telescopes.

        Args:
            telescopes (list[Telescope]): List of telescope objects.

        Raises:
            TypeError: If not a list of ``Telescope`` instances.
        """
        if not isinstance(telescopes, list):
            raise TypeError('telescopes must be a list')
        if not all((isinstance(telescope, Telescope) for telescope in telescopes)):
            raise TypeError('telescopes must be a list of Telescope objects')
        self._telescopes = copy(telescopes)
        for telescope in self._telescopes:
            telescope._parent_interferometer = self

    @property
    def kn(self) -> SuperKN:
        """Associated `SuperKN` instance.

        Returns:
            SuperKN: Kernel nuller configuration.
        """
        return self._kn

    @kn.setter
    def kn(self, kn: SuperKN):
        """Set kernel nuller.

        Args:
            kn (SuperKN): Kernel nuller object.

        Raises:
            TypeError: If not a ``SuperKN`` instance.
        """
        if not isinstance(kn, SuperKN):
            raise TypeError('kn must be a SuperKN object')
        self._kn = copy(kn)
        self._kn._parent_interferometer = self

    @property
    def camera(self) -> Camera:
        """Associated `Camera` object.

        Returns:
            Camera: Camera instance.
        """
        return self._camera

    @camera.setter
    def camera(self, camera: Camera):
        """Set camera.

        Args:
            camera (Camera): Camera object.

        Raises:
            TypeError: If not a ``Camera`` instance.
        """
        if not isinstance(camera, Camera):
            raise TypeError('camera must be a Camera object')
        self._camera = copy(camera)
        self._camera._parent_interferometer = self

    @property
    def name(self) -> str:
        """Interferometer name.

        Returns:
            str: Name of the instrument.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Set instrument name.

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
            AttributeError: If attempting to overwrite an existing parent.
        """
        if self._parent_ctx is not None:
            raise AttributeError('parent_ctx is read-only')
        else:
            self._parent_ctx = parent_ctx

    @property
    def η(self) -> u.Quantity:
        """Global optical efficiency.

        Returns:
            float: Efficiency factor in [0, +inf).
        """
        return self._η

    @η.setter
    def η(self, η: float):
        """Set optical efficiency.

        Args:
            η (float): Efficiency factor.

        Raises:
            ValueError: If not convertible to float or negative.
        """
        try:
            η = float(η)
        except (ValueError, TypeError):
            raise ValueError('η must be a float')
        if η < 0:
            raise ValueError('η must be positive')
        self._η = η
        if self.parent_ctx is not None:
            self.parent_ctx.update_photon_flux()