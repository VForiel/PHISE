import numpy as np
import math
import astropy.units as u
import astropy.constants as const
import numba as nb
from copy import deepcopy as copy
import matplotlib.pyplot as plt

from . import telescope
from .camera import Camera
try:
    import matplotlib.pyplot as plt
    from phise.modules import utils
except ImportError:
    pass
    try:
        plt.rcParams['image.origin'] = 'lower'
    except Exception:
        pass
except Exception:
    plt = None
from io import BytesIO
from scipy.optimize import curve_fit
import scipy

# Internal libs
from .interferometer import Interferometer
from .target import Target
from .companion import Companion
from ..modules import coordinates
from ..modules import signals
from ..modules import phase
from ..modules.calibration import calibrate_gen as _calibrate_gen
from ..modules.calibration import calibrate_obs as _calibrate_obs
from ..modules.transmission_map import (
    get_unique_source_input_fields_jit,
    get_transmission_maps as _get_transmission_maps,
    get_transmission_map_gradient_norm as _get_transmission_map_gradient_norm,
    plot_transmission_maps as _plot_transmission_maps,
    plot_analytical_transmission_maps as _plot_analytical_transmission_maps,
)

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

    __slots__ = ('_initialized', '_interferometer', '_target', '_h', '_h_unit', '_Δh', '_Δh_unit', '_Γ', '_Γ_unit', '_name', '_p', '_pf', '_monochromatic')

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
        
        self._update_p() # define self.p
        self._update_pf() # define self.pf

        self._initialized = True

    # To string ---------------------------------------------------------------

    def __str__(self) -> str:
        res = f'Context "{self.name}"\n'
        res += "  " + "\n  ".join(str(self.interferometer).split("\n")) + "\n"
        res += "  " + "\n  ".join(str(self.target).split("\n")) + "\n"
        res += f'  h: {self.h:.2g}\n'
        res += f'  Δh: {self.Δh:.2g}\n'
        res += f'  Γ: {self.Γ:.2g}'
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
            self._update_p()

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
            self._update_p()

    # Chip shortcut -----------------------------------------------------------

    @property
    def chip(self):
        return self.interferometer.chip

    @chip.setter
    def chip(self, chip):
        self.interferometer.chip = chip

    # Camera shortcut ---------------------------------------------------------

    @property
    def camera(self):
        return self.interferometer.camera

    @camera.setter
    def camera(self, camera):
        self.interferometer.camera = camera

    # h property --------------------------------------------------------------

    @property
    def h(self) -> u.Quantity:
        """
        Local hour angle (central time) of the observation.
        """
        return (self._h * u.hourangle).to(self._h_unit)
    
    @h.setter
    def h(self, h: u.Quantity):
        if type(h) != u.Quantity:
            raise TypeError("h must be a Quantity")
        try:
            new_h = h.to(u.hourangle).value
        except u.UnitConversionError:
            raise ValueError("h must be in a hourangle unit")
        self._h_unit = h.unit
        self._h = new_h
        if self._initialized:
            self._update_p()

    # Δh property -------------------------------------------------------------

    @property
    def Δh(self) -> u.Quantity:
        """
        Time/Hour-angle span of the observation.
        """
        return (self._Δh * u.hourangle).to(self._Δh_unit)
    
    @Δh.setter
    def Δh(self, Δh: u.Quantity):
        if type(Δh) != u.Quantity:
            raise TypeError("Δh must be a Quantity")
        try:
            new_Δh = Δh.to(u.hourangle).value
        except u.UnitConversionError:
            raise ValueError("Δh must be in a hourangle unit")
        if new_Δh < (e := self.interferometer.camera.e.to(u.hour).value):
            raise ValueError(f"Δh must be upper or equal to the exposure time {e}")
        self._Δh = new_Δh
        self._Δh_unit = Δh.unit

    # Γ property --------------------------------------------------------------

    @property
    def Γ(self) -> u.Quantity:
        """
        Upstream piston RMS (in length units) during the observation.
        """
        return (self._Γ * u.m).to(self._Γ_unit)
    
    @Γ.setter
    def Γ(self, Γ: u.Quantity):
        if type(Γ) != u.Quantity:
            raise TypeError("Γ must be a Quantity")
        try:
            new_Γ = Γ.to(u.m).value
        except u.UnitConversionError:
            raise ValueError("Γ must be in a distance unit")
        self._Γ = new_Γ
        self._Γ_unit = Γ.unit

    # p property --------------------------------------------------------------
    
    @property
    def p(self) -> u.Quantity:
        """
        (Read-only) Projected telescope positions in a plane perpendicular to the line of sight.
        """
        return self._p * u.m
        
    @p.setter
    def p(self, _):
        raise ValueError("p is a read-only property. Use _update_p() to set it accordingly to the other parameters in this context.")
    
    def _update_p(self):
        h = self.h.to(u.rad).value
        l = self.interferometer.l.to(u.rad).value
        δ = self.target.δ.to(u.rad).value
        r = np.array([i.r.to(u.m).value for i in self.interferometer.telescopes])
        
        self._p = project_position_jit(r, h, l, δ)
    
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
        if not hasattr(self, "_pf"):
            raise AttributeError("pf is not defined.")
        return self._pf
    
    @pf.setter
    def pf(self, pf: u.Quantity):
        raise ValueError("pf is a read-only property.")

    def _update_pf(self):
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

        self._pf = p * λ / (h*c) # Photon flux [photons/s] (array of (n_telescopes,))
        self._pf = self._pf.to(1/u.s)

    def get_reference_exposure_time(
            self,
            target_adu_fraction: float = 1,
            include_dark_current: bool = True,
        ) -> u.Quantity:
        """Compute a reference exposure time for stellar acquisition.

        This helper estimates an exposure time that places a detector pixel
        at a chosen fraction of full-well capacity, assuming **perfect
        coupling** of all stellar photons into a single output (ideal fiber
        coupling scenario). This avoids dependence on chip imperfections or
        interferometer coherence.

        The stellar electron rate on the **central pixel** is computed as:
            electron_rate = (total_photon_flux) × (central_pixel_fraction)
                            × (quantum_efficiency) [+ dark_current]

        where total_photon_flux = sum(pf) is the combined flux from all telescopes,
        and central_pixel_fraction is the integral of the 2D Gaussian spot over
        the central pixel area, with ``sigma = camera.spot_size`` in pixel units.

        Args:
            target_adu_fraction (float): Target fraction of pixel full-well
                capacity, in ``(0, 1]``.
            include_dark_current (bool): If ``True``, include the mean dark
                current contribution per pixel in the electron budget.

        Returns:
            u.Quantity: Reference exposure time in seconds.

        Raises:
            ValueError: If ``target_adu_fraction`` is outside ``(0, 1]`` or
                if the resulting electron rate is not strictly positive.
            TypeError: If ``target_adu_fraction`` is not numeric.

        Notes:
            This calculation assumes 100% coupling efficiency (all stellar
            photons reach one output). In practice, chip efficiency and
            interferometric visibility will be lower.

            If exposure time exceeds dark saturation threshold (≈ fwc/dc),
            a warning is issued.
        """

        if not isinstance(target_adu_fraction, (float, int)):
            raise TypeError("target_adu_fraction must be a float")

        target_adu_fraction = float(target_adu_fraction)
        if target_adu_fraction <= 0.0:
            raise ValueError("target_adu_fraction must be >= 0")
        # Target ADU can be > 1 to allow going above camera saturation

        # Total photon flux from all telescopes (ideal coupling: all photons
        # reach one output).
        total_photon_flux = float(np.sum(self._pf).to(1/u.s).value)

        # Fraction of the Gaussian spot falling into the central pixel.
        # For a centered isotropic 2D Gaussian with standard deviation sigma,
        # integrated over [-0.5, 0.5] x [-0.5, 0.5] pixel:
        # f_c = erf(0.5 / (sqrt(2) * sigma))^2
        sigma_px = float(self.camera.spot_size)
        central_pixel_fraction = math.erf(0.5 / (math.sqrt(2.0) * sigma_px)) ** 2

        # Electron rate on central pixel = photons/s in central pixel × QE
        # [+ dark current].
        electron_rate = total_photon_flux * central_pixel_fraction * self.camera.qe

        if include_dark_current:
            electron_rate += self.camera.dc

        if electron_rate <= 0:
            raise ValueError("Computed electron rate must be strictly positive")

        # Exposure time to reach target fraction of full-well capacity.
        target_electrons = target_adu_fraction * self.camera.fwc
        exposure_seconds = float(target_electrons / electron_rate)

        # Warn if exposure approaches dark saturation threshold.
        dark_sat_threshold = float(self.camera.fwc / self.camera.dc)
        if exposure_seconds > dark_sat_threshold:
            import warnings
            warnings.warn(
                f"Exposure time {exposure_seconds:.4e}s exceeds dark saturation "
                f"threshold ({dark_sat_threshold:.4e}s). Image will be dominated by "
                f"dark current.",
                UserWarning
            )

        return exposure_seconds * u.s
    
    # Plot projected positions over the time ----------------------------------

    def plot_projected_positions(
            self,
            N:int = 11,
            return_image = False,
            save_as:str = None,
        ):
        """Plot telescope positions over time.

        Args:
            N (int): Number of positions to plot.
            return_image (bool): If ``True``, return an image buffer instead of
                displaying it.
            save_as (str): Path to save the plot.

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

    def get_transmission_maps(self, N: int):
        """See :func:`phise.modules.transmission_map.get_transmission_maps`."""
        return _get_transmission_maps(self, N=N)

    def get_transmission_map_gradient_norm(self, N: int):
        """See :func:`phise.modules.transmission_map.get_transmission_map_gradient_norm`."""
        return _get_transmission_map_gradient_norm(self, N=N)

    def plot_transmission_maps(self, N: int = 100, return_plot: bool = False, grad=False, save_as=None):
        """See :func:`phise.modules.transmission_map.plot_transmission_maps`."""
        return _plot_transmission_maps(self, N=N, return_plot=return_plot, grad=grad, save_as=save_as)

    def plot_analytical_transmission_maps(self, N: int, return_plot: bool = False, save_as=None):
        """See :func:`phise.modules.transmission_map.plot_analytical_transmission_maps`."""
        return _plot_analytical_transmission_maps(self, N=N, return_plot=return_plot, save_as=save_as)

    # Distributions -----------------------------------------------------------

    def plot_distributions(self, samples:int=1000, ctx_h0=None) -> None:

        assert ctx_h0.chip._raw_output_labels == self.chip._raw_output_labels, "ctx_h0 must have the same interferometer chip configuration as this context"
        assert ctx_h0.chip._processed_output_labels == self.chip._processed_output_labels, "ctx_h0 must have the same interferometer chip configuration as this context"

        # Keeping only the mid observation
        ctx_h1 = copy(self)
        ctx_h1.Δh = ctx_h1.camera.e.to(u.hour).value * u.hourangle
        if ctx_h0 is not None:
            ctx_h0 = copy(ctx_h0)
            ctx_h0.Δh = ctx_h0.camera.e.to(u.hour).value * u.hourangle

        nb_raw = self.interferometer.chip.nb_raw_outputs
        nb_proc = self.interferometer.chip.nb_processed_outputs

        data_h1 = np.empty((samples, nb_raw + nb_proc))
        if ctx_h0 is not None:
            data_h0 = np.empty((samples, nb_raw + nb_proc))

        outs = ctx_h1.observation_serie(n=samples)[:,0,:]
        data_h1[:, :nb_raw] = outs
        for i in range(samples):
            data_h1[i, nb_raw:] = ctx_h1.chip.process_outputs(outs[i])
        if ctx_h0 is not None:
            outs = ctx_h0.observation_serie(n=samples)[:,0,:]
            data_h0[:, :nb_raw] = outs
            for i in range(samples):
                data_h0[i, nb_raw:] = ctx_h0.chip.process_outputs(outs[i])

        fig, axs = plt.subplots(1, nb_raw+nb_proc, figsize=((nb_raw+nb_proc)*5, 5))
        for o in range(nb_raw+nb_proc):
            ax = axs[o]
            if ctx_h0 is not None:
                ax.hist(data_h0[:, o], bins=30, alpha=0.5, label=f"{ctx_h0.name} - {ctx_h0.chip._raw_output_labels[o] if o < nb_raw else ctx_h0.chip._processed_output_labels[o-nb_raw]}")
            ax.hist(data_h1[:, o], bins=30, alpha=0.5, label=f"{ctx_h1.name} - {ctx_h1.chip._raw_output_labels[o] if o < nb_raw else ctx_h1.chip._processed_output_labels[o-nb_raw]}")
            ax.set_title(f"Distribution of {ctx_h1.chip._raw_output_labels[o] if o < nb_raw else ctx_h1.chip._processed_output_labels[o-nb_raw]}")
            ax.set_xlabel("Flux")
            ax.set_ylabel("Count")
            ax.legend()


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

        return np.array(input_fields, dtype=np.complex128)
    
    # H range -----------------------------------------------------------------

    def get_h_range(self) -> np.ndarray[float]:
        """Get the hour-angle range of the observation.

        Returns:
            np.ndarray[float]: Hour angle values.
        """
        
        nb_obs_per_night = int(self.Δh.to(u.hourangle).value // self.interferometer.camera.e.to(u.hour).value)

        if nb_obs_per_night < 1:
            nb_obs_per_night = 1
        
        h_range = np.linspace(self.h - self.Δh/2, self.h + self.Δh/2, nb_obs_per_night)
        return h_range

    # Observation -------------------------------------------------------------

    def observe_monochromatic(self, upstream_pistons:u.Quantity=None, mode:str="flux"):
        """Observe the target with monochromatic approximation.

        Args:
            upstream_pistons (Optional[u.Quantity]): If provided, use this static
                OPD error instead of random atmospheric piston. Shape: (n_telescopes,)
            mode (str): Output mode. Supported values are ``'flux'`` (total
                photon counts per output), ``'image'`` (camera images per
                output with shape ``(nb_outputs, resolution, resolution)``),
                and ``'demo'`` (matplotlib Figure showing output images).

        Returns:
            np.ndarray[float] | np.ndarray[int] | matplotlib.figure.Figure:
                Depends on ``mode``.
        """

        nb_outs = self.interferometer.chip.nb_raw_outputs
        nb_objects = len(self.target.companions) + 1

        # Get ideall input fields
        ψi = self.get_input_fields()

        # Get input OPD (atmospheric piston) for each telescope
        if upstream_pistons is None:
            Δφ = np.random.normal(0, self.Γ.to(u.nm).value, size=len(self.interferometer.telescopes))
        else:
            Δφ = upstream_pistons.to(u.nm).value
        λ = self.interferometer.λ.to(u.nm).value

        # Add the OPD error to the input fields
        for i in range(len(ψi)):
            ψi[i] = phase.shift_jit(ψi[i], Δφ, λ)

        # Get output fields for all companions & star
        out_fields = np.empty((nb_objects, nb_outs), dtype=np.complex128)
       
        for companion, ψc in enumerate(ψi):
            out_fields[companion] = self.interferometer.chip.get_output_fields(ψ=ψc, λ=self.interferometer.λ)

        if mode == "flux":
            # Acquire total photon counts for each output
            outs = np.empty(nb_outs)
            for o in range(nb_outs):
                outs[o] = self.interferometer.camera.get_flux(out_fields[:, o])
            return outs

        elif mode in ("image", "demo"):
            # Acquire a 2D detector image for each output
            images = np.array([
                self.interferometer.camera.get_image(out_fields[:, o])
                for o in range(nb_outs)
            ])

            if mode == "image":
                return images

            # mode == "demo": build a figure with one subplot per output
            labels = self.interferometer.chip._raw_output_labels
            ncols = min(nb_outs, 5)
            nrows = (nb_outs + ncols - 1) // ncols
            fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
            axs = np.array(axs).reshape(-1)
            for o in range(nb_outs):
                im = axs[o].imshow(images[o], origin="lower", cmap="hot")
                axs[o].set_title(labels[o] if o < len(labels) else f"Output {o}")
                axs[o].axis("off")
                fig.colorbar(im, ax=axs[o], fraction=0.046, pad=0.04)
            for o in range(nb_outs, len(axs)):
                axs[o].axis("off")
            fig.tight_layout()
            return fig

        else:
            raise ValueError(f"Unknown mode '{mode}'. Expected 'flux', 'image', or 'demo'.")
    
    def observe(self, spectral_samples=5, upstream_pistons:u.Quantity=None, mode:str="flux"):
        """Observe the target in this context.

        Args:
            spectral_samples (int): Number of spectral samples to acquire (default: 5).
            upstream_pistons (Optional[u.Quantity]): If provided, use this static
                OPD error instead of random atmospheric piston. Shape: (n_telescopes,)
            mode (str): Output mode. Supported values are ``'flux'`` (total
                photon counts per output), ``'image'`` (camera images per
                output integrated over bandwidth, shape
                ``(nb_outputs, resolution, resolution)``), and ``'demo'``
                (matplotlib Figure showing output images).

        Returns:
            np.ndarray[float] | np.ndarray[int] | matplotlib.figure.Figure:
                Depends on ``mode``.
        """

        # If this context use monochromatic approximation
        if self.monochromatic:
            return self.observe_monochromatic(upstream_pistons=upstream_pistons, mode=mode)

        # Sampling bandwidth
        λ_range = np.linspace(self.interferometer.λ - self.interferometer.Δλ/2, self.interferometer.λ + self.interferometer.Δλ/2, spectral_samples)

        nb_outs = self.interferometer.chip.nb_raw_outputs

        # Atmospheric piston for each telescope (drawn once, shared across sub-bands)
        if upstream_pistons is None:
            Δφ = np.random.normal(0, self.Γ.value, size=len(self.interferometer.telescopes)) * self.Γ.unit
        else:
            Δφ = upstream_pistons

        if mode == "flux":
            # Initialize output array for flux integration
            outs = np.empty((spectral_samples, nb_outs))

            # Monochromatic approximation for each sub-band
            for i, λ in enumerate(λ_range):
                ctx_mono = copy(self)
                ctx_mono.interferometer.λ = λ
                ctx_mono.interferometer.Δλ = 1 * u.nm
                ctx_mono._update_pf()
                outs[i] = ctx_mono.observe_monochromatic(upstream_pistons=Δφ, mode="flux")

            # Integrate over the bandwidth
            # outs contains counts for a 1 nm bandwidth (ctx_mono.interferometer.Δλ)
            # We integrate the spectral density (outs / 1 nm) over the wavelength range (in nm)
            return np.trapezoid(outs, λ_range.to(u.nm).value, axis=0) / 1.0

        elif mode in ("image", "demo"):
            # Collect per-sub-band images and integrate over bandwidth
            sample_images = None
            for i, λ in enumerate(λ_range):
                ctx_mono = copy(self)
                ctx_mono.interferometer.λ = λ
                ctx_mono.interferometer.Δλ = 1 * u.nm
                ctx_mono._update_pf()
                imgs = ctx_mono.observe_monochromatic(upstream_pistons=Δφ, mode="image")
                # imgs shape: (nb_outs, resolution, resolution)
                if sample_images is None:
                    sample_images = np.empty((spectral_samples, *imgs.shape))
                sample_images[i] = imgs

            # Integrate spectral density over wavelength range (axis=0 is spectral axis)
            integrated = np.trapezoid(sample_images, λ_range.to(u.nm).value, axis=0) / 1.0

            if mode == "image":
                return integrated

            # mode == "demo": build a figure with one subplot per output
            labels = self.interferometer.chip._raw_output_labels
            ncols = min(nb_outs, 5)
            nrows = (nb_outs + ncols - 1) // ncols
            fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
            axs = np.array(axs).reshape(-1)
            for o in range(nb_outs):
                im = axs[o].imshow(integrated[o], origin="lower", cmap="hot")
                axs[o].set_title(labels[o] if o < len(labels) else f"Output {o}")
                axs[o].axis("off")
                fig.colorbar(im, ax=axs[o], fraction=0.046, pad=0.04)
            for o in range(nb_outs, len(axs)):
                axs[o].axis("off")
            fig.tight_layout()
            return fig

        else:
            raise ValueError(f"Unknown mode '{mode}'. Expected 'flux', 'image', or 'demo'.")

    def observation_serie(
            self,
            n:int = 1,
        ) -> np.ndarray[int]:
        """Generate a series of observations in this context.

        Args:
            n (int): Number of nights (observations per given hour angle).

        Returns:
            np.ndarray[int]: Array of shape (n, nb_hour_angles, nb_outputs)
        """

        # Get hour angle range
        h_range = self.get_h_range()

        # Initialize output array
        nb_outs = self.interferometer.chip.nb_raw_outputs
        outs = np.empty((n, len(h_range), nb_outs))

        # Observe for each hour angle
        for h_i, h in enumerate(h_range):
            ctx = copy(self)
            ctx.h = h

            # Observe n nights at this hour angle
            for n_i in range(n):
                outs[n_i, h_i,:] = ctx.observe()

        return outs
    
    # Genetic calibration -----------------------------------------------------

    def calibrate_gen(
            self,
            β: float,
            verbose: bool = False,
            plot:bool = False,
            figsize:tuple = (10, 10),
            save_as = None,
        ) -> dict:
        """Optimize phase shifter offsets to maximize nulling performance.

        .. deprecated:: 1.0
            Use :func:`phise.modules.calibration.calibrate_gen` directly instead.

        Args:
            β (float): Decay factor for the step size (0.5 <= β < 1).
            verbose (bool): If ``True``, print optimization progress.
            plot (bool): If ``True``, plot the optimization process.
            figsize (tuple): Figure size for plots.
            save_as (str): Path to save the plot if plot is True.

        Returns:
            dict: Dictionary with optimization history (depth, shifters).
        """
        return _calibrate_gen(self, β, verbose=verbose, plot=plot, figsize=figsize, save_as=save_as)
    
    # Obstruction calibration -------------------------------------------------

    def calibrate_obs(
            self,
            n: int = 1_000,
            plot: bool = False,
            figsize:tuple[int] = (30,20),
            save_as = None,
        ):
        """Optimize calibration via least squares sampling.

        .. deprecated:: 1.0
            Use :func:`phise.modules.calibration.calibrate_obs` directly instead.

        Args:
            n (int): Number of sampling points for least squares.
            plot (bool): If ``True``, plot the optimization process.
            figsize (tuple[int]): Figure size for plots.
            save_as (str): Path to save the plot if plot is True.

        Returns:
            None | Context: New context with optimized kernel nuller (if implemented to return).
        """
        return _calibrate_obs(self, n=n, plot=plot, figsize=figsize, save_as=save_as)

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
# Moved to phise.modules.transmission_map — kept here only as re-exports for
# any legacy code that imported them directly from this module.
from ..modules.transmission_map import (  # noqa: F401  (re-export)
    get_transmission_map_jit,
    get_analytical_transmission_map_jit,
    get_unique_source_input_fields_jit,
)
