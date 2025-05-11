# External libs
import numpy as np
import astropy.units as u
import astropy.constants as const
import numba as nb
from copy import deepcopy as copy
import matplotlib.pyplot as plt
from io import BytesIO

# Internal libs
from .interferometer import Interferometer
from .target import Target
from . import kernel_nuller

from ..modules import coordinates
from ..modules import signals

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
        
        self.project_telescopes_position() # define self.p
        self.update_photon_flux() # define self.pf

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
        self.interferometer._parent_ctx = self
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
        self.target._parent_ctx = self
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
        if Δh < (e := self.interferometer.camera.e.to(u.hour) * u.hourangle):
            Δh = e
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
    

    # Photon flux -------------------------------------------------------------

    @property
    def pf(self) -> u.Quantity:
        """
        Photon flux in the context
        """
        if not hasattr(self, "_ph"):
            raise AttributeError("pf is not defined. Call update_photon_flux() first.")
        return self._ph
    
    @pf.setter
    def pf(self, pf: u.Quantity):
        """
        Set the photon flux in the context
        """
        raise ValueError("pf is a read-only property. Use update_photon_flux() to set it accordingly to the other parameters in this context.")

    def update_photon_flux(self):
        """
        Update the photon flux in the context.
        ❗ Assumption: the spectral flux is constant over the bandwidth.
        """

        f = self.target.f.to(u.W / u.m**2 / u.nm)
        λ = self.interferometer.λ.to(u.m)
        Δλ = self.interferometer.Δλ.to(u.nm)
        a = np.array([i.a.to(u.m**2).value for i in self.interferometer.telescopes]) * u.m**2
        h = const.h
        c = const.c

        p = f * a * Δλ # Optical power [W]

        self._ph = p * λ / (h*c) # Photon flux [photons/s]

    # Projected position ------------------------------------------------------

    @property
    def p(self) -> u.Quantity:
        """
        Projected position of the telescopes in a plane perpendicular to the line of sight.
        """
        if not hasattr(self, "_p"):
            raise AttributeError("p is not defined. Call project_telescopes_position() first.")
        return self._p
    
    @p.setter
    def p(self, p: u.Quantity):
        """
        Set the projected position of the telescopes in a plane perpendicular to the line of sight.
        """
        raise ValueError("p is a read-only property. Use project_telescopes_position() to set it accordingly to the other parameters in this context.")

    def project_telescopes_position(self):
        """
        Project the telescopes position in a plane perpendicular to the line of sight.
        """
        h = self.h.to(u.rad).value
        l = self.interferometer.l.to(u.rad).value
        δ = self.target.δ.to(u.rad).value

        r = np.array([i.r.to(u.m).value for i in self.interferometer.telescopes])
        
        self._p = project_position_njit(r, h, l, δ) * u.m
    
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

    # Transmission maps -------------------------------------------------------

    def get_transmission_maps(self, N:int) -> np.ndarray[float]:
        """
        Generate all the kernel-nuller transmission maps for a given resolution

        Parameters
        ----------
        - N: Resolution of the map

        Returns
        -------
        - Null outputs map (3 x resolution x resolution)
        - Dark outputs map (6 x resolution x resolution)
        - Kernel outputs map (3 x resolution x resolution)
        """

        N=N
        φ=self.interferometer.kn.φ.to(u.m).value
        σ=self.interferometer.kn.σ.to(u.m).value
        p=self.p.value
        λ=self.interferometer.λ.to(u.m).value
        fov=self.interferometer.fov
        output_order=self.interferometer.kn.output_order
        
        return get_transmission_map_njit(N=N, φ=φ, σ=σ, p=p, λ=λ, fov=fov, output_order=output_order)
    
    def plot_transmission_maps(self, N:int, return_plot:bool = False) -> None:
        
        # Get transmission maps
        n_maps, d_maps, k_maps = self.get_transmission_maps(N=N)

        # Get companions position to plot them
        companions_pos = []
        for c in self.target.companions:
            x, y = coordinates.αθ_to_xy(α=c.α, θ=c.θ, fov=self.fov)
            companions_pos.append((x*self.fov, y*self.fov))

        _, axs = plt.subplots(2, 6, figsize=(35, 10))

        fov = self.interferometer.fov
        extent = (-fov.value, fov.value, -fov.value, fov.value)

        for i in range(3):
            im = axs[0, i].imshow(n_maps[i], aspect="equal", cmap="hot", extent=extent)
            axs[0, i].set_title(f"Nuller output {i+1}")
            plt.colorbar(im, ax=axs[0, i])

        for i in range(6):
            im = axs[1, i].imshow(d_maps[i], aspect="equal", cmap="hot", extent=extent)
            axs[1, i].set_title(f"Dark output {i+1}")
            axs[1, i].set_aspect("equal")
            plt.colorbar(im, ax=axs[1, i])

        for i in range(3):
            im = axs[0, i + 3].imshow(k_maps[i], aspect="equal", cmap="bwr", extent=extent)
            axs[0, i + 3].set_title(f"Kernel {i+1}")
            plt.colorbar(im, ax=axs[0, i + 3])

        for ax in axs.flatten():
            ax.set_xlabel(r"$\theta_x$" + f" ({fov.unit})")
            ax.set_ylabel(r"$\theta_y$" + f" ({fov.unit})")
            ax.scatter(0, 0, color="yellow", marker="*", edgecolors="black")
            for x, y in companions_pos:
                ax.scatter(x, y, color="blue", edgecolors="black")

        # Display companions positions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        for ax in axs.flatten():
            ax.set_xlabel(r"$\theta_x$" + f" ({fov.unit})")
            ax.set_ylabel(r"$\theta_y$" + f" ({fov.unit})")
            ax.scatter(0, 0, color="yellow", marker="*", edgecolors="black", s=100)
            for x, y in companions_pos:
                ax.scatter(x, y, color="blue", edgecolors="black")

        transmissions = ""
        for i, c in enumerate(self.target.companions):
            α = c.α
            θ = c.θ
            p = self.p
            ψ = signals.get_input_fields(a=1, θ=θ, α=α, λ=self.λ, p=p)

            n, d, b = self.kn.propagate_fields(ψ=ψ, λ=self.λ)

            k = np.array([np.abs(d[2*i])**2 - np.abs(d[2*i+1])**2 for i in range(3)])

            linebreak = '<br>' if return_plot else '\n   '
            transmissions += '<h2>' if return_plot else ''
            transmissions += f"\n{c.name} throughputs:"
            transmissions += '</h1>' if return_plot else '\n----------' + linebreak
            transmissions += f"B: {np.abs(b)**2*100:.2f}%" + linebreak
            transmissions += f"N: {' | '.join([f'{np.abs(i)**2*100:.2f}%' for i in n])}" + f"   Depth: {np.sum(np.abs(d)**2) / np.abs(b)**2:.2e}" + linebreak
            transmissions += f"D: {' | '.join([f'{np.abs(i)**2*100:.2f}%' for i in d])}" + f"   Depth: {np.sum(np.abs(d)**2) / np.abs(b)**2:.2e}" + linebreak
            transmissions += f"K: {' | '.join([f'{i*100:.2f}%' for i in k])}" + f"   Depth: {np.sum(np.abs(k)) / np.abs(b)**2:.2e}" + linebreak

        if return_plot:
            plot = BytesIO()
            plt.savefig(plot, format='png')
            plt.close()
            return plot.getvalue(), transmissions
        plt.show()
        print(transmissions)

    # Input fields ------------------------------------------------------------

    def get_input_fields(self) -> np.ndarray[complex]:
        """
        Get the complexe amplitude of the signals acquired by the telescopes.

        Returns
        -------
        - Array of acquired signals (complex amplitudes)
        """

        raise NotImplementedError("This function is not implemented yet.")

        α = self.target.α.to(u.rad).value
        θ = self.target.θ.to(u.rad).value
        λ = self.interferometer.λ.to(u.m).value
        p = self.p.to(u.m).value
        
        return get_input_fields_njit(a=1, θ=θ, α=α, λ=λ, p=p)

    # Observation -------------------------------------------------------------

    def observe(
            self,
            n:int = 1,
        ) -> np.ndarray[int]:
        """
        Generate a series of observations in this context.

        Parameters
        ----------
        - n: Number of nights (= number of observations for a given hour angle)

        Returns
        -------
        - Array of observations
        """

        raise NotImplementedError("This function is not implemented yet.")

        ψ = self.interferometer.kn.propagate_fields(ψ=self.target.ψ, λ=self.interferometer.λ)
        i = self.interferometer.camera.acquire(ψ=ψ)

    
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

# Transmission maps -----------------------------------------------------------

@nb.njit()
def get_transmission_map_njit(
        N: int,
        φ: np.ndarray[float],
        σ: np.ndarray[float],
        p: np.ndarray[float],
        λ: float,
        fov: float,
        output_order: np.ndarray[int]
    ) -> tuple[np.ndarray[complex], np.ndarray[complex], np.ndarray[float]]:
    """
    Generate the transmission maps of this context with a given resolution

    Parameters
    ----------
    - N: Resolution of the map
    - φ: Array of 14 injected OPD (in meter)
    - σ: Array of 14 intrasic OPD (in meter)
    - p: Projected telescope positions (in meter)
    - λ: Wavelength (in meter)
    - fov: Field of view in mas
    - output_order: Order of the outputs

    Returns
    -------
    - Null outputs map (3 x resolution x resolution)
    - Dark outputs map (6 x resolution x resolution)
    - Kernel outputs map (3 x resolution x resolution)
    """

    # Get the coordinates of the map
    _, _, α_map, θ_map = coordinates.get_maps_njit(N=N, fov=fov)

    # mas to radian
    θ_map = θ_map / 1000 / 3600 / 180 * np.pi

    n_maps = np.zeros((3, N, N), dtype=np.complex128)
    d_maps = np.zeros((6, N, N), dtype=np.complex128)
    k_maps = np.zeros((3, N, N), dtype=float)

    for x in range(N):
        for y in range(N):
            
            α = α_map[x, y]
            θ = θ_map[x, y]

            ψ = signals.get_input_fields_njit(a=1, θ=θ, α=α, λ=λ, p=p)

            n, d, _ = kernel_nuller.propagate_fields_njit(ψ, φ, σ, λ, output_order)

            k = np.array([np.abs(d[2*i])**2 - np.abs(d[2*i+1])**2 for i in range(3)])

            for i in range(3):
                n_maps[i, x, y] = n[i]

            for i in range(6):
                d_maps[i, x, y] = d[i]

            for i in range(3):
                k_maps[i, x, y] = k[i]

    return np.abs(n_maps) ** 2, np.abs(d_maps) ** 2, k_maps

# Input fields ----------------------------------------------------------------

@nb.njit()
def get_input_fields_njit(
    a: float,
    θ: float,
    α: float,
    λ: float,
    p: np.ndarray[float],
) -> np.ndarray[complex]:
    """
    Get the complexe amplitude of the input signals according to the object and telescopes positions.

    Parameters
    ----------
    - a: Amplitude of the signal (prop. to #photons/s)
    - θ: Angular separation (in radian)
    - α: Parallactic angle (in radian)
    - λ: Wavelength (in meter)
    - p: Projected telescope positions (in meter)

    Returns
    -------
    - Array of acquired signals (complex amplitudes).
    """

    # Array of complex signals
    s = np.empty(p.shape[0], dtype=np.complex128)

    for i, t in enumerate(p):

        # Rotate the projected telescope positions by the parallactic angle
        p_rot = t[0] * np.cos(-α) - t[1] * np.sin(-α)

        # Compute the phase delay according to the object position
        Φ = 2 * np.pi * p_rot * np.sin(θ) / λ

        # Build the complex amplitude of the signal
        s[i] = np.exp(1j * Φ)

    return s * np.sqrt(a / p.shape[0])