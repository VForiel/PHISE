"""Définitions et utilitaires pour l'interféromètre.

Ce module définit la classe `Interferometer` qui encapsule les
paramètres globaux de l'instrument : latitude du réseau, longueur
d'onde centrale, bande passante, champ de vue, rendement optique,
liste des télescopes, kernel-nuller et caméra associée.
"""
from astropy import units as u
from copy import deepcopy as copy
from .kernel_nuller import KernelNuller
from .telescope import Telescope
from .camera import Camera

class Interferometer:
    """Représentation d'un interféromètre instrument.

    Fournit l'état global de l'instrument et des propriétés utilitaires
    pour synchroniser le contexte (par ex. recalcul des positions projetées
    des télescopes ou mise à jour du flux photonique lorsque certaines
    propriétés changent).
    """
    __slots__ = ('_parent_ctx', '_l', '_λ', '_Δλ', '_fov', '_η', '_telescopes', '_kn', '_camera', '_name')

    def __init__(self, l: u.Quantity, λ: u.Quantity, Δλ: u.Quantity, fov: u.Quantity, η: float, telescopes: list[Telescope], kn: KernelNuller, camera: Camera, name: str='Unnamed Interferometer'):
        """Initialisation de l'interféromètre.

        Paramètres
        ----------
        l : astropy.units.Quantity
            Latitude du centre du réseau (degrés).
        λ : astropy.units.Quantity
            Longueur d'onde centrale (nm).
        Δλ : astropy.units.Quantity
            Bande passante (nm).
        fov : astropy.units.Quantity
            Champ de vue (mas).
        η : float
            Rendement optique global (0..1).
        telescopes : list[Telescope]
            Liste d'objets `Telescope` définissant la géométrie.
        kn : KernelNuller
            Objet `KernelNuller` configurant le nuller.
        camera : Camera
            Objet `Camera` associé.
        name : str, optionnel
            Nom de l'instrument.
        """
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
        """Latitude du réseau (Quantity en degrés).

        Lors de la modification, les positions projetées des télescopes
        sont recalculées si le contexte parent est défini.
        """
        return self._l

    @l.setter
    def l(self, l: u.Quantity):
        """"l.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
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
        """Longueur d'onde centrale (Quantity en nm).

        La mise à jour déclenche le recalcul du flux photonique du contexte.
        """
        return self._λ

    @λ.setter
    def λ(self, λ: u.Quantity):
        """"λ.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
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
        """Bande passante (Quantity en nm).

        Doit être strictement positive.
        """
        return self._Δλ

    @Δλ.setter
    def Δλ(self, Δλ: u.Quantity):
        """"Δλ.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
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
        """Champ de vue (Quantity, généralement en mas)."""
        return self._fov

    @fov.setter
    def fov(self, fov: u.Quantity):
        """"fov.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
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
        """Liste des objets `Telescope` constituant le réseau."""
        return self._telescopes

    @telescopes.setter
    def telescopes(self, telescopes: list[Telescope]):
        """"telescopes.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        if not isinstance(telescopes, list):
            raise TypeError('telescopes must be a list')
        if not all((isinstance(telescope, Telescope) for telescope in telescopes)):
            raise TypeError('telescopes must be a list of Telescope objects')
        self._telescopes = copy(telescopes)
        for telescope in self._telescopes:
            telescope._parent_interferometer = self

    @property
    def kn(self) -> KernelNuller:
        """Instance `KernelNuller` associée à l'interféromètre."""
        return self._kn

    @kn.setter
    def kn(self, kn: KernelNuller):
        """"kn.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        if not isinstance(kn, KernelNuller):
            raise TypeError('kn must be a KernelNuller object')
        self._kn = copy(kn)
        self._kn._parent_interferometer = self

    @property
    def camera(self) -> Camera:
        """Objet `Camera` associé à l'interféromètre."""
        return self._camera

    @camera.setter
    def camera(self, camera: Camera):
        """"camera.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        if not isinstance(camera, Camera):
            raise TypeError('camera must be a Camera object')
        self._camera = copy(camera)
        self._camera._parent_interferometer = self

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
        if self._parent_ctx is not None:
            raise AttributeError('parent_ctx is read-only')
        else:
            self._parent_ctx = parent_ctx

    @property
    def η(self) -> u.Quantity:
        """Rendement optique global (float)."""
        return self._η

    @η.setter
    def η(self, η: float):
        """"η.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
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