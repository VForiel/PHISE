# External libs
import astropy.units as u

# Internal libs
from .companion import Companion

#==============================================================================
# Target class
#==============================================================================

class Target():

    def __init__(self, m:u.Quantity, δ:u.Quantity, companions:list[Companion], name:str = "Unnamed"):
        """
        A target star with a given magnitude and declination, and a list of companions.

        Parameters
        ----------
        m : `astropy.units.Quantity`
            Magnitude of the star.
        δ : `astropy.units.Quantity`
            Declination of the star.
        companions : list of `Companion`
            List of Companion objects.
        name : str, optional
            Name of the scene (default is "Unnamed").
        """
        
        self._ctx = None

        self.m = m
        self.δ = δ
        self.companions = companions
        self.name = name


    # m property --------------------------------------------------------------

    
    @property
    def m(self) -> u.Quantity:
        return self._m
    
    @m.setter
    def m(self, m:u.Quantity):
        if not isinstance(m, u.Quantity):
            raise TypeError("m must be an astropy Quantity")
        try:
            m = m.to(u.mag)
        except u.UnitConversionError:
            raise ValueError("m must be in magnitudes")
        self._m = m
    
    # δ property --------------------------------------------------------------

    @property
    def δ(self) -> u.Quantity:
        return self._δ
    
    @δ.setter
    def δ(self, δ:u.Quantity):
        if not isinstance(δ, u.Quantity):
            raise TypeError("δ must be an astropy Quantity")
        try:
            δ = δ.to(u.deg)
        except u.UnitConversionError:
            raise ValueError("δ must be in degrees")
        self._δ = δ

        # If declination is set, project the telescopes position
        if self.ctx is not None:
            self.ctx.project_telescopes_position()

    # companions property -----------------------------------------------------
    
    @property
    def companions(self) -> list[Companion]:
        return self._companions
    
    @companions.setter
    def companions(self, companions: list[Companion]):
        if not all(isinstance(companion, Companion) for companion in companions):
            raise TypeError("`companions` must be a list of Companion objects.")
        try:
            companions = list(companions)
        except TypeError:
            raise TypeError("companions must be a list of Companion objects")
        self._companions = companions

    # name property -----------------------------------------------------------

    
    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, name: str):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        self._name = name

    # ctx property ------------------------------------------------------------

    @property
    def ctx(self) -> list:
        return self._ctx
    
    @ctx.setter
    def ctx(self, ctx):
        raise AttributeError("ctx is read-only")

