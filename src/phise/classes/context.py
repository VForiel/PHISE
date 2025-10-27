import numpy as np
import astropy.units as u
import astropy.constants as const
import numba as nb
from copy import deepcopy as copy
import matplotlib.pyplot as plt

from . import telescope
from .camera import Camera
try:
    import matplotlib.pyplot as plt
    try:
        plt.rcParams['image.origin'] = 'lower'
    except Exception:
        pass
except Exception:
    plt = None
from io import BytesIO
from LRFutils import color
from scipy.optimize import curve_fit

# Internal libs
from .interferometer import Interferometer
from .target import Target
from .companion import Companion
from . import kernel_nuller
from .kernel_nuller import KernelNuller

from ..modules import coordinates
from ..modules import signals
from ..modules import phase

class Context:
    """
    Observation context holding instrument, target and acquisition settings.

    Args:
        interferometer (Interferometer): Instrument and geometry.
        target (Target): Target definition (coordinates, flux, companions).
        h (u.Quantity): Local hour angle (central time) of the observation.
        Δh (u.Quantity): Time/Hour-angle span of the observation.
        Γ (u.Quantity): RMS cophasing error (length quantity).
        monochromatic (bool): If ``True``, use monochromatic approximation.
        name (str): Human-readable context name.
    """

    __slots__ = ('_initialized', '_interferometer', '_target', '_h', '_Δh', '_Γ', '_name', '_p', '_ph', '_monochromatic')

    def __init__(
            self,
            interferometer:Interferometer,
            target:Target,
            h:u.Quantity,
            Δh: u.Quantity,
            Γ: u.Quantity,
            monochromatic = False,
            name:str = "Unnamed Context",
        ):

        self._initialized = False

        self.interferometer = copy(interferometer)
        self.interferometer._parent_ctx = self
        self.target = copy(target)
        self.target._parent_ctx = self
        self.h = h
        self.Δh = Δh
        self.Γ = Γ
        self.monochromatic = monochromatic
        self.name = name
        
        self.project_telescopes_position() # define self.p
        self.update_photon_flux() # define self.pf

        self._initialized = True

    # To string ---------------------------------------------------------------

    def __str__(self) -> str:
        res = f'Context "{self.name}"\n'
        res += "  " + "\n  ".join(str(self.interferometer).split("\n")) + "\n"
        res += "  " + "\n  ".join(str(self.target).split("\n")) + "\n"
        res += f'  h: {self.h:.2f}\n'
        res += f'  Δh: {self.Δh:.2f}\n'
        res += f'  Γ: {self.Γ:.2f}'
        return res
    
    def __repr__(self) -> str:
        return self.__str__()

    # Interferometer property -------------------------------------------------

    @property
    def interferometer(self) -> Interferometer:
        """
        Interferometer used in this context.
        """
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
        """
        Target observed in this context.
        """
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
        """
        Local hour angle (central time) of the observation.
        """
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
        """
        Time/Hour-angle span of the observation.
        """
        return self._Δh
    
    @Δh.setter
    def Δh(self, Δh: u.Quantity):
        if type(Δh) != u.Quantity:
            raise TypeError("Δh must be a Quantity")
        try:
            Δh = Δh.to(u.hourangle)
        except u.UnitConversionError:
            raise ValueError("Δh must be in a hourangle unit")
        if Δh < (e := self.interferometer.camera.e.to(u.hour).value * u.hourangle):
            Δh = e
        self._Δh = Δh

    # Γ property --------------------------------------------------------------

    @property
    def Γ(self) -> u.Quantity:
        """
        RMS cophasing error (in length units) of the observation.
        """
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
        """
        (Read-only) Projected telescope positions in a plane perpendicular to the line of sight.
        """
        return self._p
        
    @p.setter
    def p(self, p: u.Quantity):
        raise ValueError("p is a read-only property. Use project_telescopes_position() to set it accordingly to the other parameters in this context.")
    
    def project_telescopes_position(self):
        """
        Project telescope positions into the plane perpendicular to the LOS.

        Sets the ``p`` property (projected positions in meters) from local
        telescope positions and the hour angle ``h``.
        """
        h = self.h.to(u.rad).value
        l = self.interferometer.l.to(u.rad).value
        δ = self.target.δ.to(u.rad).value
        r = np.array([i.r.to(u.m).value for i in self.interferometer.telescopes])
        
        self._p = project_position_jit(r, h, l, δ) * u.m
        return self._p
    
    # monochromatic property --------------------------------------------------

    @property
    def monochromatic(self) -> bool:
        """
        Whether to use the monochromatic approximation.
        """
        return self._monochromatic
    
    @monochromatic.setter
    def monochromatic(self, monochromatic: bool):
        if not isinstance(monochromatic, bool):
            raise TypeError("monochromatic must be a boolean")
        self._monochromatic = monochromatic
    
    # Name property -----------------------------------------------------------

    @property
    def name(self) -> str:
        """
        Human-readable context name.
        """
        return self._name
    
    @name.setter
    def name(self, name: str):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        self._name = name    

    # Photon flux -------------------------------------------------------------

    @property
    def pf(self) -> u.Quantity:
        """
        (Read-only) Photon flux per telescope. Shape: (n_telescopes,)
        """
        if not hasattr(self, "_ph"):
            raise AttributeError("pf is not defined. Call update_photon_flux() first.")
        return self._ph
    
    @pf.setter
    def pf(self, pf: u.Quantity):
        """Set photon flux in the context (read-only property).

        Raises:
            ValueError: Always raised; use ``update_photon_flux()``.
        """
        raise ValueError("pf is a read-only property. Use update_photon_flux() to set it accordingly to the other parameters in this context.")

    def update_photon_flux(self):
        """Compute and store the photon flux received by each telescope.

        Notes:
            - The target spectral flux is assumed constant over the band Δλ.
            - If Δλ == 0 (monochromatic case), the value is normalized by 1 nm.
        """

        f = self.target.f.to(u.W / u.m**2 / u.nm)
        λ = self.interferometer.λ.to(u.m)
        η = self.interferometer.η
        Δλ = self.interferometer.Δλ.to(u.nm)
        a = np.array([i.a.to(u.m**2).value for i in self.interferometer.telescopes]) * u.m**2
        h = const.h
        c = const.c

        # Monochromatic case
        if Δλ == 0:
            Δλ = 1 * u.nm

        p = η * f * a * Δλ # Optical power [W]

        self._ph = p * λ / (h*c) # Photon flux [photons/s] (array of (n_telescopes,))
    
    # Plot projected positions over the time ----------------------------------

    def plot_projected_positions(
            self,
            N:int = 11,
            return_image = False,
        ):
        """Plot telescope positions over time.

        Args:
            N (int): Number of positions to plot.
            return_image (bool): If ``True``, return an image buffer instead of
                displaying it.

        Returns:
            Optional[bytes]: PNG image buffer when ``return_image=True``; otherwise ``None``.
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
        ax.set_title(f"Projected telescope positions over the time ({ctx.Δh.to(u.hourangle).value * u.h} long)")
        plt.legend()

        if return_image:
            buffer = BytesIO()
            plt.savefig(buffer,format='png')
            plt.close()
            return buffer.getvalue()
        plt.show()

    # Transmission maps -------------------------------------------------------

    def get_transmission_maps(self, N:int) -> np.ndarray[float]:
        """Generate all kernel nuller transmission maps at a given resolution.

        Args:
            N (int): Map resolution.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Null outputs map (3 x N x N)
                - Dark outputs map (6 x N x N)
                - Kernel outputs map (3 x N x N)
        """

        N=N
        φ=self.interferometer.kn.φ.to(u.m).value
        σ=self.interferometer.kn.σ.to(u.m).value
        p=self.p.value
        λ=self.interferometer.λ.to(u.m).value
        λ0=self.interferometer.kn.λ0.to(u.m).value
        fov=self.interferometer.fov
        output_order=self.interferometer.kn.output_order

        return get_transmission_map_jit(N=N, φ=φ, σ=σ, p=p, λ=λ, λ0=λ0, fov=fov, output_order=output_order)

    def plot_transmission_maps(self, N:int, return_plot:bool = False) -> None:
        
        # Get transmission maps
        n_maps, d_maps, k_maps = self.get_transmission_maps(N=N)

        # Get companions position to plot them
        companions_pos = []
        for c in self.target.companions:
            x, y = coordinates.ρθ_to_xy(ρ=c.ρ, θ=c.θ, fov=self.interferometer.fov)
            companions_pos.append((x*self.interferometer.fov/2, y*self.interferometer.fov/2))

        _, axs = plt.subplots(2, 6, figsize=(35, 10))

        fov = self.interferometer.fov
        extent = (-fov.value/2, fov.value/2, -fov.value/2, fov.value/2)

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
        companions = [Companion(name=self.target.name + " Star", c=1, θ=0*u.deg, ρ=0*u.mas)] + self.target.companions
        for i, c in enumerate(companions):
            θ = c.θ.to(u.rad)
            ρ = c.ρ.to(u.rad)
            p = self.p.to(u.m)
            λ = self.interferometer.λ.to(u.m)
            ψ = get_unique_source_input_fields_jit(a=1, ρ=ρ.value, θ=θ.value, λ=λ.value, p=p.value)

            n, d, b = self.interferometer.kn.propagate_fields(ψ=ψ, λ=self.interferometer.λ)

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

    # Get transmission map gradiant norm --------------------------------------

    def get_transmission_map_gradient_norm(self, N:int) -> np.ndarray[float]:
        """Get the gradient norm of the transmission maps.

        Args:
            N (int): Map resolution.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Gradient norm of null outputs (3 x N x N)
                - Gradient norm of dark outputs (6 x N x N)
                - Gradient norm of kernels (3 x N x N)
        """

        n_maps, d_maps, k_maps = self.get_transmission_maps(N=N)

        n_grad = np.empty_like(n_maps)
        d_grad = np.empty_like(d_maps)
        k_grad = np.empty_like(k_maps)

        for i in range(3):
            dnx, dny = np.gradient(n_maps[i])
            n_grad[i] = np.sqrt(dnx**2 + dny**2)

            dkx, dky = np.gradient(k_maps[i])
            k_grad[i] = np.sqrt(dkx**2 + dky**2)

        for i in range(6):
            ddx, ddy = np.gradient(d_maps[i])
            d_grad[i] = np.sqrt(ddx**2 + ddy**2)

        return n_grad, d_grad, k_grad
    
    # Plot transmission map gradient norm -------------------------------------

    def plot_transmission_map_gradient_norm(self, N:int, return_plot:bool = False) -> None:
        """Plot the gradient norm of the transmission maps.

        Args:
            N (int): Map resolution.
            return_plot (bool): If ``True``, return the image buffer instead of displaying it.

        Returns:
            Optional[bytes]: PNG image buffer when ``return_plot=True``; otherwise ``None``.
        """

        n_grad, d_grad, k_grad = self.get_transmission_map_gradient_norm(N=N)

        _, axs = plt.subplots(2, 6, figsize=(35, 10))

        fov = self.interferometer.fov
        extent = (-fov.value/2, fov.value/2, -fov.value/2, fov.value/2)

        companions_pos = []
        for c in self.target.companions:
            x, y = coordinates.ρθ_to_xy(ρ=c.ρ, θ=c.θ, fov=self.interferometer.fov)
            companions_pos.append((x*self.interferometer.fov/2, y*self.interferometer.fov/2))

        for i in range(3):
            im = axs[0, i].imshow(n_grad[i], aspect="equal", cmap="gray", extent=extent)
            axs[0, i].set_title(f"Nuller output {i+1} gradient norm")
            plt.colorbar(im, ax=axs[0, i])

        for i in range(6):
            im = axs[1, i].imshow(d_grad[i], aspect="equal", cmap="gray", extent=extent)
            axs[1, i].set_title(f"Dark output {i+1} gradient norm")
            axs[1, i].set_aspect("equal")
            plt.colorbar(im, ax=axs[1, i])

        for i in range(3):
            im = axs[0, i + 3].imshow(k_grad[i], aspect="equal", cmap="gray", extent=extent)
            axs[0, i + 3].set_title(f"Kernel {i+1} gradient norm")
            plt.colorbar(im, ax=axs[0, i + 3])

        for ax in axs.flatten():
            ax.set_xlabel(r"$\theta_x$" + f" ({fov.unit})")
            ax.set_ylabel(r"$\theta_y$" + f" ({fov.unit})")
            ax.scatter(0, 0, color="yellow", marker="*", edgecolors="black", s=100)
            for x, y in companions_pos:
                ax.scatter(x, y, color="blue", edgecolors="black")

        if return_plot:
            plot = BytesIO()
            plt.savefig(plot, format='png')
            plt.close()
            return plot.getvalue()
        plt.show()

    # Input fields ------------------------------------------------------------

    def get_input_fields(self) -> np.ndarray[complex]:
        """Get complex amplitudes of the signals acquired by the telescopes.

        Returns:
            np.ndarray[complex]: Array of shape (n_companions + 1, n_telescopes).
        """
    
        input_fields = []
        λ = self.interferometer.λ.to(u.m).value
        p = self.p.to(u.m).value # Projected telescope positions
        pf = self.pf.to(1/u.s).value # Photon flux from the star for each telescope
        
        # Star input fields
        input_fields.append(get_unique_source_input_fields_jit(a=pf, ρ=0, θ=0, λ=λ, p=p))

        # Companion input fields
        for c in self.target.companions:
            pf_c = pf * c.c # Photon flux from the companion for each telescope
            input_fields.append(get_unique_source_input_fields_jit(a=pf_c, ρ=c.ρ.to(u.rad).value, θ=c.θ.to(u.rad).value, λ=λ, p=p))
        
        # Error OPD
        γ = np.random.normal(0, self.Γ.to(u.m).value, size=len(self.interferometer.telescopes))

        # OPD to phase difference
        phase = 2 * np.pi * γ / λ

        # Add the OPD error to the input fields
        for i in range(len(input_fields)):
            input_fields[i] = input_fields[i] * np.exp(1j * phase)

        return np.array(input_fields, dtype=np.complex128)
    
    # H range -----------------------------------------------------------------

    def get_h_range(self) -> np.ndarray[float]:
        """Get the hour-angle range of the observation.

        Returns:
            np.ndarray[float]: Hour angle values (radians).
        """
        
        nb_obs_per_night = int(self.Δh.to(u.hourangle).value // self.interferometer.camera.e.to(u.hour).value)

        if nb_obs_per_night < 1:
            nb_obs_per_night = 1
        
        h_range = np.linspace(self.h - self.Δh/2, self.h + self.Δh/2, nb_obs_per_night)
        return h_range

    # Observation -------------------------------------------------------------

    def observe_monochromatic(self):
        """Observe the target with monochromatic approximation.

        Returns:
            tuple[np.ndarray, np.ndarray, float]:
                - Dark data (6,) — photon events
                - Kernel data (3,) — photon events
                - Bright data (1,) — photon events
        """

        ctx = copy(self)
        ctx.interferometer.Δλ = 1 * u.nm
        
        nb_objects = len(ctx.target.companions) + 1

        ds = np.empty((nb_objects, 6), dtype=np.complex128)
        bs = np.empty(nb_objects, dtype=np.complex128)

        for ψ_i, ψ in enumerate(ctx.get_input_fields()):

            _, d, b = ctx.interferometer.kn.propagate_fields(ψ=ψ, λ=ctx.interferometer.λ)
            ds[ψ_i] = d
            bs[ψ_i] = b

        bright = ctx.interferometer.camera.acquire_pixel(bs)

        darks = np.empty(6)
        for i in range(6):
            darks[i] = ctx.interferometer.camera.acquire_pixel(ds[:, i])

        kernels = np.empty(3)
        for i in range(3):
            kernels[i] = darks[2*i] - darks[2*i+1]

        # Multiply by the bandwidth
        darks *= self.interferometer.Δλ.to(u.nm).value
        kernels *= self.interferometer.Δλ.to(u.nm).value
        bright *= self.interferometer.Δλ.to(u.nm).value

        return darks, kernels, bright
    
    def observe(self, spectral_samples=5):
        """Observe the target in this context.

        Args:
            spectral_samples (int): Number of spectral samples to acquire (default: 5).

        Returns:
            tuple[np.ndarray, np.ndarray, float]:
                - Dark data (6,) — photon events
                - Kernel data (3,) — photon events
                - Bright data (1,) — photon events
        """

        if self.monochromatic:
            return self.observe_monochromatic()

        # Sampling bandwidth
        λ_range = np.linspace(self.interferometer.λ - self.interferometer.Δλ/2, self.interferometer.λ + self.interferometer.Δλ/2, spectral_samples)

        darks = np.empty((spectral_samples, 6))
        kernels = np.empty((spectral_samples, 3))
        brights = np.empty(spectral_samples)

        # Monochromatic approximation
        for i, λ in enumerate(λ_range):
            ctx_mono = copy(self)
            ctx_mono.interferometer.λ = λ
            ctx_mono.interferometer.Δλ = 1 * u.nm

            d, k, b = ctx_mono.observe_monochromatic()

            # Store the results for each wavelength
            darks[i] = d
            kernels[i] = k
            brights[i] = b

        # Integrate over the bandwidth
        dark = np.trapz(darks, λ_range.value, axis=0)
        kernel = np.trapz(kernels, λ_range.value, axis=0)
        bright = np.trapz(brights, λ_range.value, axis=0)

        return dark, kernel, bright

    def observation_serie(
            self,
            n:int = 1,
        ) -> np.ndarray[int]:
        """Generate a series of observations in this context.

        Args:
            n (int): Number of nights (observations per given hour angle).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Dark data (n, n_h, 6) — photon events
                - Kernel data (n, n_h, 3) — photon events
                - Bright data (n, n_h) — photon events
        """

        h_range = self.get_h_range()

        brights = np.empty((n, len(h_range)))
        darks = np.empty((n, len(h_range), 6))
        kernels = np.empty((n, len(h_range), 3))

        for h_i, h in enumerate(h_range):
            ctx = copy(self)
            ctx.h = h

            for n_i in range(n):
                
                d, k, b = ctx.observe()

                brights[n_i, h_i] = b
                darks[n_i, h_i] = d
                kernels[n_i, h_i] = k

        return darks, kernels, brights
    
    # Genetic calibration -----------------------------------------------------

    def calibrate_gen(
            self,
            β: float,
            verbose: bool = False,
            plot:bool = False,
            figsize:tuple = (10, 10),
        ) -> dict:
        """Optimize phase shifter offsets to maximize nulling performance.

        Args:
            β (float): Decay factor for the step size (0.5 <= β < 1).
            verbose (bool): If ``True``, print optimization progress.
            plot (bool): If ``True``, plot the optimization process.
            figsize (tuple): Figure size for plots.

        Returns:
            dict: Dictionary with optimization history (depth, shifters).
        """

        self.Δh = self.interferometer.camera.e.to(u.hour).value * u.hourangle

        ψ = np.sqrt(self.pf.to(1/self.interferometer.camera.e.unit).value) * (1 + 0j) # Perfectly cophased inputs
        total_execpted_photons = np.sum(np.abs(ψ)**2)

        ε = 1e-6 * self.interferometer.λ.unit # Minimum shift step size

        # Shifters that contribute to redirecting light to the bright output
        φb = [1, 2, 3, 4, 5, 7]

        # Shifters that contribute to the symmetry of the dark outputs
        φk = [6, 8, 9, 10, 11, 12, 13, 14]

        # History of the optimization
        depth_history = []
        shifters_history = []

        Δφ = self.interferometer.λ / 4
        while Δφ > ε:

            if verbose:
                print(color.black(color.on_red(f"--- New iteration ---")), f"Δφ={Δφ:.2e}")

            for i in φb + φk:
                log = ""

                # Getting observation with different phase shifts
                self.interferometer.kn.φ[i-1] += Δφ
                _, k_pos, b_pos = self.observe()

                self.interferometer.kn.φ[i-1] -= 2*Δφ
                _, k_neg, b_neg = self.observe()

                self.interferometer.kn.φ[i-1] += Δφ
                _, k_old, b_old = self.observe()

                # Computing throughputs
                b_pos = b_pos / total_execpted_photons
                b_neg = b_neg / total_execpted_photons
                b_old = b_old / total_execpted_photons
                k_pos = np.sum(np.abs(k_pos)) / total_execpted_photons
                k_neg = np.sum(np.abs(k_neg)) / total_execpted_photons
                k_old = np.sum(np.abs(k_old)) / total_execpted_photons

                # Save the history
                depth_history.append(np.sum(k_old) / np.sum(b_old))
                shifters_history.append(np.copy(self.interferometer.kn.φ.value))

                # Maximize the bright metric for group 1 shifters
                if i in φb:
                    log += "Shift " + color.black(color.on_lightgrey(f"{i}")) + " Bright: " + color.black(color.on_green(f"{b_neg:.2e} | {b_old:.2e} | {b_pos:.2e}")) + " -> "

                    if b_pos > b_old and b_pos > b_neg:
                        log += color.black(color.on_green(" + "))
                        self.interferometer.kn.φ[i-1] += Δφ
                    elif b_neg > b_old and b_neg > b_pos:
                        log += color.black(color.on_green(" - "))
                        self.interferometer.kn.φ[i-1] -= Δφ
                    else:
                        log += color.black(color.on_green(" = "))

                # Minimize the kernel metric for group 2 shifters
                else:
                    log += "Shift " + color.black(color.on_lightgrey(f"{i}")) + " Kernel: " + color.black(color.on_blue(f"{k_neg:.2e} | {k_old:.2e} | {k_pos:.2e}")) + " -> "

                    if k_pos < k_old and k_pos < k_neg:
                        self.interferometer.kn.φ[i-1] += Δφ
                        log += color.black(color.on_blue(" + "))
                    elif k_neg < k_old and k_neg < k_pos:
                        self.interferometer.kn.φ[i-1] -= Δφ
                        log += color.black(color.on_blue(" - "))
                    else:
                        log += color.black(color.on_blue(" = "))
                
                if verbose:
                    print(log)

            Δφ *= β

        self.interferometer.kn.φ = phase.bound(self.interferometer.kn.φ, self.interferometer.λ)

        if plot:

            shifters_history = np.array(shifters_history)

            _, axs = plt.subplots(2,1, figsize=figsize, constrained_layout=True)

            axs[0].plot(depth_history)
            axs[0].set_xlabel("Iterations")
            axs[0].set_ylabel("Kernel-Null depth")
            axs[0].set_yscale("log")
            axs[0].set_title("Performance of the Kernel-Nuller")

            for i in range(shifters_history.shape[1]):
                axs[1].plot(shifters_history[:,i], label=f"Shifter {i+1}")
            axs[1].set_xlabel("Iterations")
            axs[1].set_ylabel("Phase shift")
            axs[1].set_yscale("linear")
            axs[1].set_title("Convergence of the phase shifters")
            # axs[1].legend(loc='upper right')

            plt.show()

        return {
            "depth": np.array(depth_history),
            "shifters": np.array(shifters_history),
        }
    
    # Obstruction calibration -------------------------------------------------

    def calibrate_obs(
            self,
            n: int = 1_000,
            plot: bool = False,
            figsize:tuple[int] = (30,20),
        ):
        """Optimize calibration via least squares sampling.

        Args:
            n (int): Number of sampling points for least squares.
            plot (bool): If ``True``, plot the optimization process.
            figsize (tuple[int]): Figure size for plots.

        Returns:
            None | Context: New context with optimized kernel nuller (if implemented to return).
        """


        kn = self.interferometer.kn
        input_attenuation_backup = kn.input_attenuation.copy()
        λ = self.interferometer.λ
        e = self.interferometer.camera.e
        total_photons = np.sum(self.pf.to(1/e.unit).value) * e.value

        if plot:
            _, axs = plt.subplots(6, 3, figsize=figsize, constrained_layout=True)
            for i in range(7):
                axs.flatten()[i].set_xlabel("Phase shift")
                axs.flatten()[i].set_ylabel("Throughput")

        def maximize_bright(p, plt_coords=None):

            x = np.linspace(0, λ.value,n)
            y = np.empty(n)

            if isinstance(p,list):
                Δp = kn.φ[p[1]-1] - kn.φ[p[0]-1]

            for i in range(n):

                if isinstance(p,list):
                    kn.φ[p[0]-1] = i * λ / n
                    kn.φ[p[1]-1] = (kn.φ[p[0]-1] + Δp) % λ
                else:
                    kn.φ[p-1] = i * λ / n
            
                _, _, b = self.observe()
                y[i] = b / total_photons
            
            def sin(x, x0):
                return (np.sin((x-x0)/λ.value*2*np.pi)+1)/2 * (np.max(y)-np.min(y)) + np.min(y)
            
            popt, _ = curve_fit(sin, x, y, p0=[0], maxfev = 100_000)

            if isinstance(p,list):
                kn.φ[p[0]-1] = (np.mod(popt[0]+λ.value/4, λ.value) * λ.unit).to(kn.φ.unit)
                kn.φ[p[1]-1] = (kn.φ[p[0]-1] + Δp) % λ
            else:
                kn.φ[p-1] = (np.mod(popt[0]+λ.value/4, λ.value) * λ.unit).to(kn.φ.unit)

            if plot:
                axs[plt_coords].set_title(f"$|B(\phi{p})|$")
                axs[plt_coords].scatter(x, y, label='Data', color='tab:blue')
                axs[plt_coords].plot(x, sin(x, *popt), label='Fit', color='tab:orange')
                axs[plt_coords].axvline(x=np.mod(popt[0]+λ.value/4, λ.value), color='k', linestyle='--', label='Optimal phase shift')
                axs[plt_coords].set_xlabel(f"Phase shift ({λ.unit})")
                axs[plt_coords].set_ylabel("Bright throughput")
                axs[plt_coords].legend()

        def minimize_kernel(p, m, plt_coords=None):

            x = np.linspace(0,λ.value,n)
            y = np.empty(n)

            for i in range(n):
                kn.φ[p-1] = i * λ / n
                _, k, b = self.observe()
                y[i] = k[m-1] / b
            
            def sin(x, x0):
                return (np.sin((x-x0)/λ.value*2*np.pi)+1)/2 * (np.max(y)-np.min(y)) + np.min(y)
            
            popt, _ = curve_fit(sin, x, y, p0=[0], maxfev = 100_000)

            kn.φ[p-1] = (np.mod(popt[0], λ.value) * λ.unit).to(kn.φ.unit)

            if plot:
                axs[plt_coords].set_title(f"$K_{m}(\phi{p})$")
                axs[plt_coords].scatter(x, y, label='Data', color='tab:blue')
                axs[plt_coords].plot(x, sin(x, *popt), label='Fit', color='tab:orange')
                axs[plt_coords].axvline(x=np.mod(popt[0], λ.value), color='k', linestyle='--', label='Optimal phase shift')
                axs[plt_coords].set_xlabel(f"Phase shift ({λ.unit})")
                axs[plt_coords].set_ylabel("Kernel throughput")
                axs[plt_coords].legend()

        def maximize_darks(p, ds, plt_coords=None):

            x = np.linspace(0, λ.value, n)
            y = np.empty(n)

            for i in range(n):
                kn.φ[p-1] = i * λ / n
                d, _, b = self.observe()
                y[i] = np.sum(np.abs(d[np.array(ds)-1])) / b

            def sin(x, x0):
                return (np.sin((x-x0)/λ.value*2*np.pi)+1)/2 * (np.max(y)-np.min(y)) + np.min(y)
            
            popt, _ = curve_fit(sin, x, y, p0=[0], maxfev = 100_000)

            kn.φ[p-1] = (np.mod(popt[0]+λ.value/4, λ.value) * λ.unit).to(kn.φ.unit)

            if plot:
                axs[plt_coords].set_title(f"$|D_{ds[0]}(\phi{p})| + |D_{ds[1]}(\phi{p})|$")
                axs[plt_coords].scatter(x, y, label='Data', color='tab:blue')
                axs[plt_coords].plot(x, sin(x, *popt), label='Fit', color='tab:orange')
                axs[plt_coords].axvline(x=np.mod(popt[0]+λ.value/4, λ.value), color='k', linestyle='--', label='Optimal phase shift')
                axs[plt_coords].set_xlabel(f"Phase shift ({λ.unit})")
                axs[plt_coords].set_ylabel(f"Dark pair {ds} throughput")
                axs[plt_coords].legend()

        # Bright maximization
        self.interferometer.kn.input_attenuation = [1, 1, 0, 0]
        maximize_bright(2, plt_coords=(0,0))
        maximize_bright([1,2], plt_coords=(1,0))

        if plot:
            axs[1,1].axis('off')
            axs[1,2].axis('off')
            plt.show()

        return

        self.interferometer.kn.input_attenuation = [0, 0, 1, 1]
        maximize_bright(4, plt_coords=(0,1))
        maximize_bright([3,4], plt_coords=(1,1))

        self.interferometer.kn.input_attenuation = [1, 0, 1, 0]
        maximize_bright(7, plt_coords=(0,2))
        maximize_bright([5,7], plt_coords=(1,2))

        # Darks maximization
        self.interferometer.kn.input_attenuation = [1, 0, 0, -1]
        maximize_darks(8, [1,2], plt_coords=(1,0))

        # Kernel minimization
        self.interferometer.kn.input_attenuation = [1, 0, 0, 0]
        minimize_kernel(11, 1, plt_coords=(2,0))
        minimize_kernel(13, 2, plt_coords=(2,1))
        minimize_kernel(14, 3, plt_coords=(2,2))

        kn.φ = phase.bound(kn.φ, λ)
        kn.input_attenuation = input_attenuation_backup

        if plot:
            axs[1,1].axis('off')
            axs[1,2].axis('off')
            plt.show()

    #==============================================================================
    # VLTI Context
    #==============================================================================

    def get_VLTI() -> 'Context':
        """Get a default VLTI context for analysis.

        Uses:
            - VLTI with 4 UTs
            - First generation active kernel nuller
            - Vega as target star and a hypothetical 2 mas, 1e-6 contrast companion
        """

        λ = 1.55 * u.um # Central wavelength

        ctx = Context(
            h = 0 * u.hourangle, # Central hour angle
            Δh = 8 * u.hourangle, # Hour angle range
            Γ = 100 * u.nm, # Input cophasing error (RMS)
            monochromatic=False,
            name="Default Context", # Context name
            interferometer = Interferometer(
                l = -24.6275 * u.deg, # Latitude
                λ = λ, # Central wavelength
                Δλ = 1 * u.nm, # Bandwidth
                fov = 10 * u.mas, # Field of view
                η = 0.02, # Optical efficiency
                telescopes = telescope.get_VLTI_UTs(),
                name = "VLTI", # Interferometer name
                kn = KernelNuller(
                    φ = np.zeros(14) * u.um, # Injected phase shifts
                    σ = np.abs(np.random.normal(0, 1, 14)) * u.um, # Manufacturing OPD errors
                    λ0 = λ,
                    name = "First Generation Kernel-Nuller", # Kernel nuller name
                ),
                camera = Camera(
                    e = 5 * u.min, # Exposure time
                    name = "Default Camera", # Camera name
                ),
            ),
            target=Target(
                f = (1050 * u.Jy * 2 * np.pi * const.c / λ**2).to(u.W / u.m**2 / u.nm), # Target flux
                δ = -64.71 * u.deg, # Target declination
                name = "Vega", # Target name
                companions = [
                    Companion(
                        c = 1e-6, # Companion contrast
                        ρ = 4 * u.mas, # Companion angular separation
                        θ = 0 * u.deg, # Companion position angle
                        name = "Hypothetical Companion", # Companion name
                    ),
                ],
            ),
        )

        return ctx

    #==============================================================================
    # LIFE Context
    #==============================================================================

    def get_LIFE() -> 'Context':
        """Get a default LIFE context for analysis.

        Uses:
            - 4 telescopes of LIFE
            - First generation active kernel nuller
            - Vega as target star and a hypothetical 2 mas, 1e-6 contrast companion
        """

        λ = 4 * u.um # Central wavelength

        ctx = Context(
            interferometer = Interferometer(
                l = -90 * u.deg, # Latitude
                λ = λ, # Central wavelength
                Δλ = 1 * u.nm, # Bandwidth
                fov = 10 * u.mas, # Field of view
                η = 0.02, # Optical efficiency
                telescopes = telescope.get_VLTI_UTs(),
                name = "LIFE", # Interferometer name
                kn = KernelNuller(
                    φ = np.zeros(14) * u.um, # Injected phase shifts
                    σ = np.abs(np.random.normal(0, 1, 14)) * u.um, # Manufacturing OPD errors
                    λ0 = λ,
                    name = "First Generation Kernel-Nuller", # Kernel nuller name
                ),
                camera = Camera(
                    e = 5 * u.min, # Exposure time
                    name = "Default Camera", # Camera name
                ),
            ),
            target=Target(
                f = (1050 * u.Jy * 2 * np.pi * const.c / λ**2).to(u.W / u.m**2 / u.nm), # Target flux
                δ = -90 * u.deg, # Target declination
                name = "Vega", # Target name
                companions = [
                    Companion(
                        c = 1e-6, # Companion contrast
                        ρ = 4 * u.mas, # Companion angular separation
                        θ = 0 * u.deg, # Companion position angle
                        name = "Hypothetical Companion", # Companion name
                    ),
                ],
            ),
            h = 0 * u.hourangle, # Central hour angle
            Δh = 24 * u.hourangle, # Hour angle range
            Γ = 1 * u.nm, # Input cophasing error (RMS)
            name="Default Context", # Context name
        )

        return ctx
            
#==============================================================================
# Number functions
#==============================================================================

# Projected position ----------------------------------------------------------

@nb.njit()
def project_position_jit(
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
def get_transmission_map_jit(
        N: int,
        φ: np.ndarray[float],
        σ: np.ndarray[float],
        p: np.ndarray[float],
        λ: float,
        λ0: float,
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
    - λ0: Reference wavelength (in meter)
    - fov: Field of view in mas
    - output_order: Order of the outputs

    Returns
    -------
    - Null outputs map (3 x resolution x resolution)
    - Dark outputs map (6 x resolution x resolution)
    - Kernel outputs map (3 x resolution x resolution)
    """

    # Get the coordinates of the map
    _, _, θ_map, ρ_map = coordinates.get_maps_jit(N=N, fov=fov)

    # mas to radian
    ρ_map = ρ_map / 1000 / 3600 / 180 * np.pi

    n_maps = np.zeros((3, N, N), dtype=np.complex128)
    d_maps = np.zeros((6, N, N), dtype=np.complex128)
    k_maps = np.zeros((3, N, N), dtype=float)

    for x in range(N):
        for y in range(N):
            
            θ = θ_map[x, y]
            ρ = ρ_map[x, y]

            ψ = get_unique_source_input_fields_jit(a=1, ρ=ρ, θ=θ, λ=λ, p=p)

            n, d, _ = kernel_nuller.propagate_fields_jit(ψ, φ, σ, λ, λ0, output_order)

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
def get_unique_source_input_fields_jit(
    a: float,
    ρ: float,
    θ: float,
    λ: float,
    p: np.ndarray[float],
) -> np.ndarray[complex]:
    """
    Get the complexe amplitude of the input signals according to the object and telescopes positions.

    Parameters
    ----------
    - a: Intensity of the signal (prop. to #photons/s)
    - ρ: Angular separation (in radian)
    - θ: Parallactic angle (in radian)
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
        p_rot = t[0] * np.cos(-θ) - t[1] * np.sin(-θ)

        # Compute the phase delay according to the object position
        Φ = 2 * np.pi * p_rot * np.sin(ρ) / λ

        # Build the complex amplitude of the signal
        s[i] = np.exp(1j * Φ)

    return s * np.sqrt(a / p.shape[0])
