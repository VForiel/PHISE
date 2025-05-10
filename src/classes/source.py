from astropy import units as u
from copy import deepcopy as copy

class Source():
    def __init__(self, name:str, c:float, θ:u.Quantity, α:u.Quantity):
        """Any light source in the sky (unresolved).

        Parameters
        ----------
        - c: Contrast with respect to the star
        - θ: Angular separation
        - α: Parallactic angle
        """

        self.name = name
        self.c = c
        self.θ = θ.to(u.mas)
        self.α = α.to(u.rad)

    def copy(self,
            name:str = None,
            c:float = None,
            θ:u.Quantity = None,
            α:u.Quantity = None,
            **kwargs
        ) -> "Source":
        """
        Return a copy of the Source object with some parameters changed.

        Parameters
        ----------
        - c: Contrast with respect to the star
        - θ: Angular separation
        - α: Parallactic angle
        """
        return Source(
            name = copy(name) if name is not None else copy(self.name),
            c = copy(c) if c is not None else copy(self.c),
            θ = copy(θ) if θ is not None else copy(self.θ),
            α = copy(α) if α is not None else copy(self.α),
        )
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        return f'Source "{self.name}": contrast = {self.c:.2e}, θ = {self.θ}, α = {self.α}'