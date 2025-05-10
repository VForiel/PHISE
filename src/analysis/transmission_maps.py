# External libs
from copy import deepcopy as copy
import astropy.units as u
import numpy as np
import ipywidgets as widgets
from IPython.display import display

# Internal libs
from .. import *


def gui(
        N:int = 100,
        r:u.Quantity = None,
        δ:u.Quantity = None,
        h:u.Quantity = None,
        l:u.Quantity = None,
        fov:u.Quantity = None,
        λ:u.Quantity = None,
        companions:list[Companion] = None,
    ):

    # Set default values ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if r is None:
        telescopes = telescope.get_VLTI_UTs()
        r = np.array([t.r for t in telescopes]) * u.m
    if h is None:
        h = 0 * u.hourangle
    if l is None:
        l = -24.6275 * u.deg # Cerro Paranal
    if δ is None:
        δ = -64.71 * u.deg # Vega
    if fov is None:
        fov = 10 * u.mas
    if λ is None:
        λ = 1.65 * u.um
    if companions is None:
        companions = []

    # Context ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ref_ctx = Context(
        interferometer=Interferometer(
            l=l,
            λ=λ,
            Δλ=0 * u.m, # Unused
            fov=fov,
            telescopes=telescope.get_VLTI_UTs(), # Unused
            kn=KernelNuller(
                φ=np.zeros(14) * u.m, # Unused
                σ=np.zeros(14) * u.m, # Unused
            ),
        ),
        target=Target(
            m=0 * u.mag, # Unused
            δ=δ,
            companions=companions,
        ),
        h=h,
        Δh=24 * u.hourangle,
        e=0*u.s, # Unused
        Γ=0*u.nm, # Unused
    )

    # UI elements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    h_slider = widgets.FloatSlider(value=0, min=(h-ref_ctx.Δh/2).value, max=(h+ref_ctx.Δh/2).value, step=0.01, description='Hour angle:')
    l_slider = widgets.FloatSlider(value=ref_ctx.interferometer.l.to(u.deg).value, min=-90, max=90, step=0.01, description='Latitude:')
    δ_slider = widgets.FloatSlider(value=δ.to(u.deg).value, min=-90, max=90, step=0.01, description='Declination:')
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