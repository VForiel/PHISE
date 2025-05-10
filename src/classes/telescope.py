import astropy.units as u
from copy import deepcopy as copy

class Telescope():
    def __init__(self, a:u.Quantity, r:u.Quantity, name:str = "Unnamed"):
        """
        Initialize the telescope object.

        Parameters
        ----------
        - a: Area of the telescope
        - r: Relative position to the baseline center
        - name: Name of the telescope
        """
        self.a = a
        self.r = r
        self.name = name

    def copy(
            self,
            a:u.Quantity = None,
            r:u.Quantity = None,
            name:str = None,
        ) -> "Telescope":
        """
        Return a copy of the Telescope object with some parameters changed.

        Parameters
        ----------
        - a: Area of the telescope
        - r: Relative position to the baseline center
        - name: Name of the telescope

        Returns
        -------
        - Telescope object with the same parameters as the original one, but with some parameters changed.
        """
        return Telescope(
            a=copy(a) if a is not None else copy(self.a),
            r=copy(r) if r is not None else copy(self.r),
            name=name if name is not None else self.name,
        )