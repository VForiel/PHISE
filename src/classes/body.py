from astropy import units as u

class Body():
    def __init__(self, name:str, c:float, θ:u.Quantity, α:u.Quantity):
        """Any object in the sky (unresolved).

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