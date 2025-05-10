# External libs
import numpy as np
import astropy.units as u
import numba as nb
from copy import deepcopy as copy
import matplotlib.pyplot as plt
from io import BytesIO

# Internal libs
from .interferometer import Interferometer
from .target import Target

class Context:
    def __init__(
            self,
            interferometer:Interferometer,
            target:Target,
            h:u.Quantity,
            Δh: u.Quantity,
            e: u.Quantity,
            Γ: u.Quantity,
            name:str = "Unnamed",
        ):
        """
        Parameters
        ----------
        - instrument: Instrument object
        - h: Central hourangle of the observation
        - Δh: Hourangle range of the observation
        - e: Exposition time
        - Γ: Input cophasing error (rms)
        - name: Name of the scene
        """

        self._initialized = False

        self.interferometer = copy(interferometer)
        self.interferometer._ctx = self
        self.target = copy(target)
        self.target._ctx = self
        self.h = h
        self.Δh = Δh
        self.e= e
        self.Γ = Γ
        self.name = name
        
        self.project_telescopes_position()

        self._initialized = True

    # Interferometer property -------------------------------------------------

    @property
    def interferometer(self) -> Interferometer:
        return self._interferometer
    
    @interferometer.setter
    def interferometer(self, interferometer:Interferometer):
        if not isinstance(interferometer, Interferometer):
            raise TypeError("interferometer must be an Interferometer object")
        self._interferometer = copy(interferometer)
        self.interferometer._ctx = self
        if self._initialized:
            self.project_telescopes_position()

    # Target property ---------------------------------------------------------
    
    @property
    def target(self) -> Target:
        return self._target
    
    @target.setter
    def target(self, target: Target):
        if not isinstance(target, Target):
            raise TypeError("target must be a Target object")
        self._target = copy(target)
        self.target._ctx = self
        if self._initialized:
            self.project_telescopes_position()

    # h property --------------------------------------------------------------

    @property
    def h(self) -> u.Quantity:
        return self._h
    
    @h.setter
    def h(self, h: u.Quantity):
        if type(h) != u.Quantity:
            raise TypeError("h must be a Quantity")
        try:
            h = h.to(u.hourangle)
        except u.UnitConversionError:
            raise ValueError("h must be in a hourangle unit")
        self._h = h
        if self._initialized:
            self.project_telescopes_position()

    # Δh property -------------------------------------------------------------

    @property
    def Δh(self) -> u.Quantity:
        return self._Δh
    
    @Δh.setter
    def Δh(self, Δh: u.Quantity):
        if type(Δh) != u.Quantity:
            raise TypeError("Δh must be a Quantity")
        try:
            Δh = Δh.to(u.hourangle)
        except u.UnitConversionError:
            raise ValueError("Δh must be in a hourangle unit")
        self._Δh = Δh

    # e property --------------------------------------------------------------

    @property
    def e(self) -> u.Quantity:
        return self._e
    
    @e.setter
    def e(self, e: u.Quantity):
        if type(e) != u.Quantity:
            raise TypeError("e must be a Quantity")
        try:
            e = e.to(u.s)
        except u.UnitConversionError:
            raise ValueError("e must be in a time unit")
        self._e = e

    # Γ property --------------------------------------------------------------

    @property
    def Γ(self) -> u.Quantity:
        return self._Γ
    
    @Γ.setter
    def Γ(self, Γ: u.Quantity):
        if type(Γ) != u.Quantity:
            raise TypeError("Γ must be a Quantity")
        try:
            Γ = Γ.to(u.m)
        except u.UnitConversionError:
            raise ValueError("Γ must be in a distance unit")
        self._Γ = Γ

    # p property --------------------------------------------------------------
    
    @property
    def p(self) -> u.Quantity:
        return self._p
        
    @p.setter
    def p(self, p: u.Quantity):
        raise ValueError("p is a read-only property. Use project_telescopes_position() to set it accordingly to the other parameters in this context.")

    # Projected position ------------------------------------------------------

    def project_telescopes_position(self):
        """
        Project the telescopes position in a plane perpendicular to the line of sight.
        """
        h = self.h.to(u.rad).value
        l = self.interferometer.l.to(u.rad).value
        δ = self.target.δ.to(u.rad).value

        r = np.array([i.r.to(u.m).value for i in self.interferometer.telescopes])
        
        self._p = project_position_njit(r, h, l, δ) * u.m

        return self.p
    
    # Plot projected positions over the time ----------------------------------

    def plot_projected_positions(
            self,
            N:int = 11,
            return_image = False,
        ):
        """
        Plot the telescope positions over the time.

        Parameters
        ----------
        - N: Number of positions to plot
        - return_image: Return the image buffer instead of displaying it

        Returns
        -------
        - None | Image buffer if return_image is True
        """
        _, ax = plt.subplots()

        h_range = np.linspace(self.h - self.Δh/2, self.h + self.Δh/2, N, endpoint=True)

        # Plot UT trajectory
        for i, h in enumerate(h_range):
            ctx = copy(self)
            ctx.h = h
            for j, (x, y) in enumerate(ctx.p):
                ax.scatter(x, y, label=f"Telescope {j+1}" if i==len(h_range)-1 else None, color=f"C{j}", s=1+14*i/len(h_range))

        print(self.interferometer.l)
        for (x, y) in self.p:
            ax.scatter(x, y, color="black", marker="+")

        ax.set_aspect("equal")
        ax.set_xlabel(f"x [{self.p.unit}]")
        ax.set_ylabel(f"y [{self.p.unit}]")
        ax.set_title(f"Projected telescope positions over the time (8h long)")
        plt.legend()

        if return_image:
            buffer = BytesIO()
            plt.savefig(buffer,format='png')
            plt.close()
            return buffer.getvalue()
        plt.show()
    
#==============================================================================
# Number functions
#==============================================================================

# Projected position ----------------------------------------------------------

@nb.njit()
def project_position_njit(
        r: np.ndarray[float],
        h: float,
        l: float,
        δ: float,
    ) -> np.ndarray[float]:
    """
    Project the telescope position in a plane perpendicular to the line of sight.

    Parameters
    ----------
    - r: Array of telescope positions (in meters)
    - h: Hour angle (in radian)
    - l: Latitude (in radian)
    - δ: Declination (in radian)

    Returns
    -------
    - Array of projected telescope positions (same shape and unit as p)
    """

    M = np.array([
        [ -np.sin(l)*np.sin(h),                                np.cos(h)          ],
        [ np.sin(l)*np.cos(h)*np.sin(δ) + np.cos(l)*np.cos(δ), np.sin(h)*np.sin(δ)],
    ])

    p = np.empty_like(r)
    for i, (x,y) in enumerate(r):
        p[i] = M @ np.array([y, x])

    return p