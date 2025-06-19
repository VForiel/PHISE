# External libs
import astropy.units as u
import numpy as np

class Telescope():
    def __init__(self, a:u.Quantity, r:u.Quantity, name:str = "Unnamed Telescope"):
        """
        Initialize the telescope object.

        Parameters
        ----------
        - a: Area of the telescope
        - r: Relative position to the baseline center
        - name: Name of the telescope
        """

        self._parent_interferometer = None

        self.a = a
        self.r = r
        self.name = name

    # To string ---------------------------------------------------------------

    def __str__(self) -> str:
        res = f'Telescope "{self.name}"\n'
        res += f'  Area: {self.a:.2e}\n'
        res += f'  Relative position: [{", ".join([f"{i:.2e}" for i in self.r.value])}] {self.r.unit}'
        return res.replace("e+00", "")
    
    def __repr__(self) -> str:
        return self.__str__()

    # a property --------------------------------------------------------------

    @property
    def a(self) -> u.Quantity:
        return self._a
    
    @a.setter
    def a(self, a:u.Quantity):
        if not isinstance(a, u.Quantity):
            raise TypeError("a must be an astropy Quantity")
        try:
            a = a.to(u.m**2)
        except u.UnitConversionError:
            raise ValueError("a must be in a surface area unit")
        self._a = a

        if self.parent_interferometer is not None:
            self.parent_interferometer.parent_ctx.update_photon_flux()

    # r property --------------------------------------------------------------

    @property
    def r(self) -> u.Quantity:
        return self._r
    
    @r.setter
    def r(self, r:u.Quantity):
        if not isinstance(r, u.Quantity):
            raise TypeError("r must be an astropy Quantity")
        try:
            r = r.to(u.m)
        except u.UnitConversionError:
            raise ValueError("r must be in a length unit")
        if r.shape != (2,):
            raise ValueError("r must have a shape of (2,)")
        self._r = r

        if self.parent_interferometer is not None:
            self.parent_interferometer.parent_ctx.project_telescopes_position()

    # name property -----------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, name: str):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        self._name = name

    # parent_interferometer property ------------------------------------------

    @property
    def parent_interferometer(self):
        return self._parent_interferometer
    
    @parent_interferometer.setter
    def parent_interferometer(self, parent_interferometer):
        raise ValueError("parent_interferometer is read-only")

#==============================================================================
# Other functions
#==============================================================================

# Get VLTI UTs ----------------------------------------------------------------

def get_VLTI_UTs() -> list[Telescope]:
    """
    Get the relative position of the UTs, in meter.

    Returns
    -------
    - List containing the 4 VLTI UTs as Telescope objects.
    """

    # UT coordinates... obtained on Google map 😅 <- TODO: update with precise positions
    r = np.array([
        [-70.4048732988764, -24.627602893919807],
        [-70.40465753243652, -24.627118902835786],
        [-70.40439460074228, -24.62681028261176],
        [-70.40384287956437, -24.627033500373024]
    ]) # ⚠️ Expressed in [longitude, latitude] to easily convert to [x, y]
    
    # We are only interested in the relative positions of the UTs
    # The first UT is the reference
    r -= r[0]

    # Altitude of the UTs
    earth_radius = 6_378_137 * u.m
    UTs_elevation = 2_635 * u.m

    # Angle to distance conversion
    r = np.tan((r * u.deg).to(u.rad)) * (earth_radius + UTs_elevation)

    # Area of the collectors (4m radius)
    a = 4 * np.pi * (4*u.m)**2

    return [Telescope(a=a, r=pos, name=f"UT {i + 1}") for i, pos in enumerate(r)]