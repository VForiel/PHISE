import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from astropy import units as u
import ipywidgets as widgets
from IPython.display import display
from io import BytesIO
from copy import deepcopy as copy

from src.modules import signals
from src.modules import telescopes
from src.modules import coordinates
from src.modules import signals
from src.classes import kernel_nuller
from src.classes.kernel_nuller import KernelNuller
from src.classes.source import Source
from src.classes.telescope import Telescope

class Interferometer:
    def __init__(
            self,
            l:u.Quantity,
            λ:u.Quantity,
            Δλ:u.Quantity,
            telescopes:list[Telescope],
            kn:KernelNuller,
            name:str = "Unnamed",
        ):
        """
        Instrument object.

        Parameters
        ----------
        - l: Latitude of baseline center
        - λ: Central wavelength
        - Δλ: Bandwidth
        - telescopes: List of telescope in the array
        - kn: Kernel-Nuller object
        - name: Name of the instrument
        """
        
        self.l = l
        self.λ = λ
        self.Δλ = Δλ
        self.telescopes = telescopes
        self.kn = kn
        self.name = name

    def copy(
            self,
            l:u.Quantity=None,
            λ:u.Quantity=None,
            Δλ:u.Quantity=None,
            telescopes:list[Telescope]=None,
            kn:KernelNuller=None,
            **kwargs,
        ) -> "Interferometer":
        """
        Create a copy of the interferometer with some parameters changed.

        Parameters
        ----------
        - l: Latitude of baseline center
        - λ: Central wavelength
        - Δλ: Bandwidth
        - telescopes: List of telescopes in the array
        - kn: Kernel-Nuller object
        - **kwargs: Parameters to change on the kernel-nuller object (if no new kn is provided)
        """
        return Interferometer(
            l = copy(l) if l is not None else copy(self.l),
            λ = copy(λ) if λ is not None else copy(self.λ),
            Δλ = copy(Δλ) if Δλ is not None else copy(self.Δλ),
            telescopes = copy(telescopes) if telescopes is not None else [t.copy() for t in self.telescopes],
            kn = kn.copy(**kwargs) if kn is not None else self.kn.copy(**kwargs),
            name = self.name,
        )

    # Observation -------------------------------------------------------------

    def observe(
            self,
            sources:list[Source],
            δ:u.Quantity,
            h:u.Quantity,
            f:float,
            Δt:u.Quantity,
            input_ce_rms:u.Quantity,
        ) -> np.ndarray[float]:
        """
        Simulate a 4 telescope Kernel-Nuller observation

        Parameters
        ----------
        - sources: List of the sources to observe
        - δ: Declination
        - h: Hour angle
        - f: Flux of the star (photons/s)
        - Δt: Integration time
        - input_ce_rms: RMS of the input complex electric field

        Returns
        -------
        - np.ndarray: Dark outputs (6 float values)
        - np.ndarray: Kernel outputs (3 float values)
        - float: Bright output
        """

        # Project telescope positions
        p = telescopes.project_position(r=self.r, h=h, l=self.l, δ=δ)

        # Get input fields
        d_tot = np.zeros(6)
        k_tot = np.zeros(3)
        b_tot = 0

        input_perturbation = np.random.normal(0, input_ce_rms.to(self.λ.unit).value, 4) * self.λ.unit

        for s in sources:
            ψ = signals.get_input_fields(a=f*s.c, θ=s.θ, α=s.α, λ=self.λ, p=p)
            ψ *= np.exp(1j * input_perturbation.value / self.λ.value * 2 * np.pi)

            d, k, b = self.kn.observe(ψ, self.λ, f, Δt)
            d_tot += d
            k_tot += k
            b_tot += b

        return d_tot, k_tot, b_tot

    # Transmission maps -------------------------------------------------------
    
    def get_transmission_maps(
        self,
        N:int,
        h:u.Quantity,
        δ:u.Quantity,
    ) -> tuple[np.ndarray[complex], np.ndarray[complex], np.ndarray[float]]:
        """
        Generate all the kernel-nuller transmission maps for a given resolution

        Parameters
        ----------
        - N: Resolution of the map
        - h: Hour angle
        - δ: Declination

        Returns
        -------
        - Null outputs map (3 x resolution x resolution)
        - Dark outputs map (6 x resolution x resolution)
        - Kernel outputs map (3 x resolution x resolution)
        """
        φ = self.kn.φ.to(u.m).value
        σ = self.kn.σ.to(u.m).value
        fov = self.fov.to(u.mas).value
        p = telescopes.project_position(r=self.r, h=h, l=self.l, δ=δ).to(u.m).value
        λ = self.λ.to(u.m).value

        return get_transmission_map_njit(N=N, φ=φ, σ=σ, p=p, λ=λ, fov=fov, output_order=self.kn.output_order)
    
    # Plotting ----------------------------------------------------------------

    def plot_transmission_maps(
            self,
            N: int,
            h: float,
            δ: float,
            sources: list[Source],
            return_plot=False
        ) -> None:
        
        # Get transmission maps
        n_maps, d_maps, k_maps = self.get_transmission_maps(N=N, h=h, δ=δ)

        # Get sources position to plot them
        sources_pos = []
        for c in sources:
            x, y = coordinates.αθ_to_xy(α=c.α, θ=c.θ, fov=self.fov)
            sources_pos.append((x*self.fov, y*self.fov))

        _, axs = plt.subplots(2, 6, figsize=(35, 10))

        for i in range(3):
            im = axs[0, i].imshow(n_maps[i], aspect="equal", cmap="hot", extent=(-self.fov.value, self.fov.value, -self.fov.value, self.fov.value))
            axs[0, i].set_title(f"Nuller output {i+1}")
            plt.colorbar(im, ax=axs[0, i])

        for i in range(6):
            im = axs[1, i].imshow(d_maps[i], aspect="equal", cmap="hot", extent=(-self.fov.value, self.fov.value, -self.fov.value, self.fov.value))
            axs[1, i].set_title(f"Dark output {i+1}")
            axs[1, i].set_aspect("equal")
            plt.colorbar(im, ax=axs[1, i])

        for i in range(3):
            im = axs[0, i + 3].imshow(k_maps[i], aspect="equal", cmap="bwr", extent=(-self.fov.value, self.fov.value, -self.fov.value, self.fov.value))
            axs[0, i + 3].set_title(f"Kernel {i+1}")
            plt.colorbar(im, ax=axs[0, i + 3])

        for ax in axs.flatten():
            ax.set_xlabel(r"$\theta_x$" + f" ({self.fov.unit})")
            ax.set_ylabel(r"$\theta_y$" + f" ({self.fov.unit})")
            ax.scatter(0, 0, color="yellow", marker="*", edgecolors="black")
            for x, y in sources_pos:
                ax.scatter(x, y, color="blue", edgecolors="black")

        transmissions = ""
        for i, c in enumerate(sources):
            α = c.α
            θ = c.θ
            p = telescopes.project_position(r=self.r, h=h, l=self.l, δ=δ)
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

    # Interactive plot ------------------------------------------------------------
        
    def iplot_transmission_maps(
            self,
            N: int,
            δ: u.Quantity,
            h: u.Quantity,
            Δh: u.Quantity,
            sources: list[Source],
        ):

        # UI elements
        h_slider = widgets.FloatSlider(value=0, min=(h-Δh/2).value, max=(h+Δh/2).value, step=0.01, description='Hour angle:')
        l_slider = widgets.FloatSlider(value=self.l.to(u.deg).value, min=-90, max=90, step=0.01, description='Latitude:')
        δ_slider = widgets.FloatSlider(value=δ.to(u.deg).value, min=-90, max=90, step=0.01, description='Declination:')
        reset = widgets.Button(description="Reset values")
        run = widgets.Button(description="Run")
        plot = widgets.Image()
        transmission = widgets.HTML()

        inst = copy(self)

        def update_plot(*args):
            run.button_style = "warning"

            inst.l = l_slider.value*u.deg
            
            img, txt = inst.plot_transmission_maps(
                N=N,
                h=h_slider.value*u.deg,
                δ=δ_slider.value*u.deg,
                sources=sources,
                return_plot=True,
            )
            plot.value = img
            transmission.value = txt
            
            run.button_style = ""

        def reset_values(*args):
            h_slider.value = 0
            l_slider.value = self.l.to(u.deg).value
            δ_slider.value = δ.to(u.deg).value
            run.color = "blue"
            enable_run()

        def enable_run(*args):
            run.button_style = "success"
        
        reset.on_click(reset_values)
        h_slider.observe(enable_run)
        l_slider.observe(enable_run)
        δ_slider.observe(enable_run)
        run.on_click(update_plot)
        display(widgets.VBox([h_slider, l_slider, δ_slider, widgets.HBox([reset, run]), plot, transmission]))
        update_plot()

    def __repr(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        return f'Interferometer "{self.name}":\n' + \
            " | " + f"λ = {self.λ}, Δλ = {self.Δλ}, l = {self.l}, fov = {self.fov}" + \
            "\n | " + f"r = {[str(i) for i in self.r]}" + \
            "\n | " + "\n | ".join(str(self.kn).split('\n'))

#==============================================================================
# Numba functions
#==============================================================================

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
    Generate all the kernel-nuller transmission maps for a given resolution

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