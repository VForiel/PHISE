# Trick to import Target but avoiding circular import
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from target import Target

# External libs
from astropy import units as u

class Companion():
    def __init__(self, c:float, θ:u.Quantity, α:u.Quantity, name:str = "Unnamed Companion"):
        """Any light source in the sky (unresolved).

        Parameters
        ----------
        - c: Contrast with respect to the host star
        - θ: Angular separation
        - α: Parallactic angle
        """

        self._parent_target = None

        self.θ = θ
        self.α = α
        self.c = c
        self.name = name
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        res = f'Companion "{self.name}"\n'
        res += f'  Contrast: {self.c:.2f}\n'
        res += f'  Angular separation: {self.θ:.2f}\n'
        res += f'  Parallactic angle: {self.α:.2f}'
        return res

    # c property --------------------------------------------------------------

    @property
    def c(self) -> float:
        return self._c
    
    @c.setter
    def c(self, c:float):
        if not isinstance(c, (int, float)):
            raise TypeError("c must be a float")
        if c < 0:
            raise ValueError("c must be positive")
        self._c = float(c)

    # θ property --------------------------------------------------------------

    @property
    def θ(self) -> u.Quantity:
        return self._θ
    
    @θ.setter
    def θ(self, θ:u.Quantity):
        if not isinstance(θ, u.Quantity):
            raise TypeError("θ must be an astropy Quantity")
        try:
            θ = θ.to(u.mas)
        except u.UnitConversionError:
            raise ValueError("θ must be an angle")
        self._θ = θ

    # α property --------------------------------------------------------------

    @property
    def α(self) -> u.Quantity:
        return self._α
    
    @α.setter
    def α(self, α:u.Quantity):
        if not isinstance(α, u.Quantity):
            raise TypeError("α must be an astropy Quantity")
        try:
            α = α.to(u.rad)
        except u.UnitConversionError:
            raise ValueError("α must be an angle")
        self._α = α

    # parent_target property --------------------------------------------------

    @property
    def parent_target(self) -> Target:
        return self._parent_target
    
    @parent_target.setter
    def parent_target(self, target:Target):
        raise ValueError("parent_target is read-only")