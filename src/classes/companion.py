# External libs
from astropy import units as u

class Companion():
    def __init__(self, c:float, θ:u.Quantity, α:u.Quantity, name:str = "Unnamed"):
        """Any light source in the sky (unresolved).

        Parameters
        ----------
        - c: Contrast with respect to the host star
        - θ: Angular separation
        - α: Parallactic angle
        """

        self.θ = θ.to(u.mas)
        self.α = α.to(u.rad)
        self.c = float(c)
        self.name = str(name)
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        return f'Companion "{self.name}": contrast = {self.c:.2e}, θ = {self.θ}, α = {self.α}'