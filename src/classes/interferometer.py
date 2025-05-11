# External libs
from astropy import units as u
from copy import deepcopy as copy

# Internal libs
from .kernel_nuller import KernelNuller
from .telescope import Telescope
from .camera import Camera

class Interferometer:
    def __init__(
            self,
            l:u.Quantity,
            λ:u.Quantity,
            Δλ:u.Quantity,
            fov:u.Quantity,
            telescopes:list[Telescope],
            kn:KernelNuller,
            camera:Camera,
            name:str = "Unnamed",
        ):
        """
        Instrument object.

        Parameters
        ----------
        - l: Latitude of baseline center
        - λ: Central wavelength
        - Δλ: Bandwidth
        - fov: Field of view
        - telescopes: List of telescope in the array
        - kn: Kernel-Nuller object
        - name: Name of the instrument
        """
        
        self._parent_ctx = None
        
        self.l = l
        self.λ = λ
        self.Δλ = Δλ
        self.fov = fov
        self.telescopes = copy(telescopes)
        for telescope in self.telescopes:
            telescope._parent_interferometer = self
        self.kn = copy(kn)
        self.kn._parent_interferometer = self
        self.camera = copy(camera)
        self.camera._parent_interferometer = self
        self.name = name


    # l property --------------------------------------------------------------

    @property
    def l(self) -> u.Quantity:
        return self._l
    
    @l.setter
    def l(self, l:u.Quantity):
        if not isinstance(l, u.Quantity):
            raise TypeError("l must be an astropy Quantity")
        try:
            l = l.to(u.deg)
        except u.UnitConversionError:
            raise ValueError("l must be in degrees")
        self._l = l

        # If latitude is set, project the telescopes position
        if self.parent_ctx is not None:
            self.parent_ctx.project_telescopes_position()


    # λ property --------------------------------------------------------------

    @property
    def λ(self) -> u.Quantity:
        return self._λ
    
    @λ.setter
    def λ(self, λ:u.Quantity):
        if not isinstance(λ, u.Quantity):
            raise TypeError("λ must be an astropy Quantity")
        try:
            λ = λ.to(u.m)
        except u.UnitConversionError:
            raise ValueError("λ must be in meters")
        self._λ = λ

    # Δλ property -------------------------------------------------------------

    @property
    def Δλ(self) -> u.Quantity:
        return self._Δλ
    
    @Δλ.setter
    def Δλ(self, Δλ:u.Quantity):
        if not isinstance(Δλ, u.Quantity):
            raise TypeError("Δλ must be an astropy Quantity")
        try:
            Δλ = Δλ.to(u.m)
        except u.UnitConversionError:
            raise ValueError("Δλ must be in meters")
        self._Δλ = Δλ

    # fov property ------------------------------------------------------------

    @property
    def fov(self) -> u.Quantity:
        return self._fov
    
    @fov.setter
    def fov(self, fov:u.Quantity):
        if not isinstance(fov, u.Quantity):
            raise TypeError("fov must be an astropy Quantity")
        try:
            fov = fov.to(u.mas)
        except u.UnitConversionError:
            raise ValueError("fov must be in milliarcseconds")
        self._fov = fov

    # telescopes property -----------------------------------------------------

    @property
    def telescopes(self) -> list[Telescope]:
        return self._telescopes
    
    @telescopes.setter
    def telescopes(self, telescopes:list[Telescope]):
        if not isinstance(telescopes, list):
            raise TypeError("telescopes must be a list")
        if not all(isinstance(telescope, Telescope) for telescope in telescopes):
            raise TypeError("telescopes must be a list of Telescope objects")
        self._telescopes = copy(telescopes)
        for telescope in self._telescopes:
            telescope._parent_interferometer = self

    # kn property -------------------------------------------------------------

    @property
    def kn(self) -> KernelNuller:
        return self._kn
    
    @kn.setter
    def kn(self, kn:KernelNuller):
        if not isinstance(kn, KernelNuller):
            raise TypeError("kn must be a KernelNuller object")
        self._kn = copy(kn)
        self._kn._parent_interferometer = self

    # camera property ---------------------------------------------------------

    @property
    def camera(self) -> Camera:
        return self._camera
    
    @camera.setter
    def camera(self, camera:Camera):
        if not isinstance(camera, Camera):
            raise TypeError("camera must be a Camera object")
        self._camera = copy(camera)
        self._camera._parent_interferometer = self

    # name property -----------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, name:str):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        self._name = name

    # parent_ctx property -----------------------------------------------------

    @property
    def parent_ctx(self) -> list:
        return self._parent_ctx
    
    @parent_ctx.setter
    def parent_ctx(self, parent_ctx):
        if self._parent_ctx is not None:
            raise AttributeError("parent_ctx is read-only")
        else:
            self._parent_ctx = parent_ctx
    
