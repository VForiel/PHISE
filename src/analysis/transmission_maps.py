# External libs
from copy import deepcopy as copy
import astropy.units as u
import numpy as np
import ipywidgets as widgets
from IPython.display import display

# Internal libs
from .. import *
from . import default_context


def gui(
        N:int = 100,
        r:np.ndarray[float] = None,
        δ:u.Quantity = None,
        h:u.Quantity = None,
        l:u.Quantity = None,
        fov:u.Quantity = None,
        λ:u.Quantity = None,
        φ:u.Quantity = None,
        σ:u.Quantity = None,
        companions:list[Companion] = None,
    ):
    """
    GUI to visualize the transmission maps of the VLTI.
    Parameters
    ----------
    N : int, optional
        Resolution of the maps. The default is 100 (for 100x100).
    r : np.ndarray[float], optional
        Positions of the telescopes in the array. If None, the default positions
        are used.
    δ : `astropy.units.Quantity`, optional
        Declination of the target. If None, the default value is used.
    h : `astropy.units.Quantity`, optional
        Hour angle of the target. If None, the default value is used.
    l : `astropy.units.Quantity`, optional
        Latitude of the target. If None, the default value is used.
    fov : `astropy.units.Quantity`, optional
        Field of view of the instrument. If None, the default value is used.
    λ : `astropy.units.Quantity`, optional
        Wavelength of the instrument. If None, the default value is used.
    φ : `astropy.units.Quantity`, optional
        Phase of the kernel nuller. If None, the default value is used.
    σ : `astropy.units.Quantity`, optional
        Standard deviation of the kernel nuller. If None, the default value is used.
    companions : list[Companion], optional
        List of companions to add to the target. If None, the default value is used.
    """

    # Set default values ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ref_ctx = default_context.get()

    if r is not None:
        ref_ctx.interferometer.telescopes = [Telescope(a=1*u.m**2, r=pos) for pos in r]
    else:
        # Setting the area of all telescopes to 1 m^2
        for telescope in ref_ctx.interferometer.telescopes:
            telescope.a = 1*u.m**2
    if h is not None:
        ref_ctx.h = h
    if l is not None:
        ref_ctx.interferometer.l = l
    if δ is not None:
        ref_ctx.target.δ = δ
    if fov is not None:
        ref_ctx.interferometer.fov = fov
    if λ is not None:
        ref_ctx.interferometer.λ = λ
    if φ is not None:
        ref_ctx.interferometer.kn.φ = φ
    else:
        # Setting the phase of all telescopes to 0
        ref_ctx.interferometer.kn.φ *= 0
    if σ is not None:
        ref_ctx.interferometer.kn.σ = σ
    else:
        # Setting the standard deviation of all telescopes to 0
        ref_ctx.interferometer.kn.σ *= 0
    if companions is not None:
        ref_ctx.target.companions = companions

    # UI elements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    h_slider = widgets.FloatSlider(value=0, min=(ref_ctx.h-ref_ctx.Δh/2).value, max=(ref_ctx.h+ref_ctx.Δh/2).value, step=0.01, description='Hour angle:')
    l_slider = widgets.FloatSlider(value=ref_ctx.interferometer.l.to(u.deg).value, min=-90, max=90, step=0.01, description='Latitude:')
    δ_slider = widgets.FloatSlider(value=ref_ctx.target.δ.to(u.deg).value, min=-90, max=90, step=0.01, description='Declination:')
    reset = widgets.Button(description="Reset values")
    run = widgets.Button(description="Run")
    plot = widgets.Image()
    transmission = widgets.HTML()

    # Callbacks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def update_plot(*args):
        run.button_style = "warning"

        ctx = copy(ref_ctx)

        ctx.interferometer.l = l_slider.value*u.deg
        ctx.target.δ = δ_slider.value*u.deg
        ctx.h = h_slider.value*u.deg
        
        img, txt = ctx.plot_transmission_maps(
            N=N,
            return_plot=True,
        )
        plot.value = img
        transmission.value = txt
        
        run.button_style = ""

    def reset_values(*args):
        l_slider.value = ref_ctx.interferometer.l.to(u.deg).value
        δ_slider.value = ref_ctx.target.δ.to(u.deg).value
        h_slider.value = ref_ctx.h.to(u.deg).value
        run.color = "blue"
        enable_run()

    def enable_run(*args):
        run.button_style = "success"

    # Triggers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    reset.on_click(reset_values)
    h_slider.observe(enable_run)
    l_slider.observe(enable_run)
    δ_slider.observe(enable_run)
    run.on_click(update_plot)

    # Display ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    display(widgets.VBox([h_slider, l_slider, δ_slider, widgets.HBox([reset, run]), plot, transmission]))
    update_plot()