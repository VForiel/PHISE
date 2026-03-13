from astropy import units as u
from copy import deepcopy as copy
from .chip import Chip
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
        chip (Chip): Photonic Chip.
        camera (Camera): Associated camera.
        name (str, optional): Instrument name.
    """
    __slots__ = ('_parent_ctx', '_l', '_l_unit', '_λ', '_λ_unit', '_Δλ', '_Δλ_unit', '_fov', '_fov_unit', '_η', '_telescopes', '_chip', '_camera', '_name')

    def __init__(self, l: u.Quantity, λ: u.Quantity, Δλ: u.Quantity, fov: u.Quantity, η: float, telescopes: list[Telescope], chip: Chip, camera: Camera, name: str='Unnamed Interferometer'):
        self._parent_ctx = None
        self.l = l
        self.λ = λ
        self.Δλ = Δλ
        self.fov = fov
        self.η = η
        self.telescopes = copy(telescopes)
        for telescope in self.telescopes:
            telescope._parent_interferometer = self
        self.chip = copy(chip)
        self.chip._parent_interferometer = self
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
        res += f'  ' + '\n  '.join(str(self.chip).split('\n')) + '\n'
        res += f'  ' + '\n  '.join(str(self.camera).split('\n'))
        return res

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def l(self) -> u.Quantity:
        """Latitude of the center of the telescope array.
        """
        return (self._l * u.deg).to(self._l_unit)

    @l.setter
    def l(self, l: u.Quantity):
        if not isinstance(l, u.Quantity):
            raise TypeError('l must be an astropy Quantity')
        try:
            new_l = l.to(u.deg).value
        except u.UnitConversionError:
            raise ValueError('l must be in degrees')
        self._l = new_l
        self._l_unit = l.unit
        if self.parent_ctx is not None:
            self.parent_ctx._update_p()

    @property
    def λ(self) -> u.Quantity:
        """Central wavelength
        """
        return (self._λ * u.nm).to(self._λ_unit)

    @λ.setter
    def λ(self, λ: u.Quantity):
        if not isinstance(λ, u.Quantity):
            raise TypeError('λ must be an astropy Quantity')
        try:
            new_λ = λ.to(u.nm).value
        except u.UnitConversionError:
            raise ValueError('λ must be in nanometers')
        self._λ = new_λ
        self._λ_unit = λ.unit
        if self.parent_ctx is not None:
            self.parent_ctx._update_pf()

    @property
    def Δλ(self) -> u.Quantity:
        """Bandwidth
        """
        return (self._Δλ * u.nm).to(self._Δλ_unit)

    # Set bandwidth
    @Δλ.setter
    def Δλ(self, Δλ: u.Quantity):

        # Check if it's a quantity
        if not isinstance(Δλ, u.Quantity):
            raise TypeError('Δλ must be an astropy Quantity')

        # Convert to nanometers
        try:
            new_Δλ = Δλ.to(u.nm).value
        except u.UnitConversionError:
            raise ValueError('Δλ must be in nanometers')

        # Ensure it's positive
        if new_Δλ <= 0:
            raise ValueError('Δλ must be positive')

        # Update internal state
        self._Δλ = new_Δλ
        self._Δλ_unit = Δλ.unit

        # Update parent context if it exists
        if self.parent_ctx is not None:
            self.parent_ctx._update_pf()

    @property
    def fov(self) -> u.Quantity:
        """Field of view"""
        return (self._fov * u.mas).to(self._fov_unit)

    @fov.setter
    def fov(self, fov: u.Quantity):
        if not isinstance(fov, u.Quantity):
            raise TypeError('fov must be an astropy Quantity')
        try:
            new_fov = fov.to(u.mas).value
        except u.UnitConversionError:
            raise ValueError('fov must be in milliarcseconds')
        self._fov = new_fov
        self._fov_unit = fov.unit

    @property
    def telescopes(self) -> list[Telescope]:
        """List of `Telescope` objects constituting the array"""
        return self._telescopes

    @telescopes.setter
    def telescopes(self, telescopes: list[Telescope]):
        if not isinstance(telescopes, list):
            raise TypeError('telescopes must be a list')
        if not all((isinstance(telescope, Telescope) for telescope in telescopes)):
            raise TypeError('telescopes must be a list of Telescope objects')
        self._telescopes = copy(telescopes)
        for telescope in self._telescopes:
            telescope._parent_interferometer = self

    @property
    def chip(self) -> Chip:
        """Associated `Chip` instance"""
        return self._chip

    @chip.setter
    def chip(self, chip: Chip):
        if not isinstance(chip, Chip):
            raise TypeError('chip must be a Chip object')
        self._chip = copy(chip)
        self._chip._parent_interferometer = self

    @property
    def camera(self) -> Camera:
        """Associated `Camera` object"""
        return self._camera

    @camera.setter
    def camera(self, camera: Camera):
        if not isinstance(camera, Camera):
            raise TypeError('camera must be a Camera object')
        self._camera = copy(camera)
        self._camera._parent_interferometer = self

    @property
    def name(self) -> str:
        """Interferometer name"""
        return self._name

    @name.setter
    def name(self, name: str):
        if not isinstance(name, str):
            raise TypeError('name must be a string')
        self._name = name

    @property
    def parent_ctx(self) -> list:
        """Parent observing context (read-only)"""
        return self._parent_ctx

    @parent_ctx.setter
    def parent_ctx(self, parent_ctx):
        if self._parent_ctx is not None:
            raise AttributeError('parent_ctx is read-only')
        else:
            self._parent_ctx = parent_ctx

    @property
    def η(self) -> u.Quantity:
        """Global optical efficiency"""
        return self._η

    @η.setter
    def η(self, η: float):
        try:
            η = float(η)
        except (ValueError, TypeError):
            raise ValueError('η must be a float')
        if η < 0:
            raise ValueError('η must be positive')
        self._η = η
        if self.parent_ctx is not None:
            self.parent_ctx._update_pf()