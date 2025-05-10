# External libs
import numpy as np
import astropy.units as u
from ipywidgets import widgets
from IPython.display import display
from io import BytesIO
from copy import deepcopy as copy

# Internal libs
from ..classes import context
from ..classes.interferometer import Interferometer
from ..classes.kernel_nuller import KernelNuller
from ..classes.target import Target
from ..classes import telescope

def gui(
        r:u.Quantity = None,
        h:u.Quantity = None,
        Δh:u.Quantity = None,
        l:u.Quantity = None,
        δ:u.Quantity = None,
    ):

    # Set default values ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if r is None:
        telescopes = telescope.get_VLTI_UTs()
        r = np.array([t.r for t in telescopes]) * u.m
    if h is None:
        h = 0 * u.hourangle
    if Δh is None:
        Δh = 8 * u.hourangle
    if l is None:
        l = -24.6275 * u.deg # Cerro Paranal
    if δ is None:
        δ = -64.71 * u.deg # Vega

    # Context ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ref_ctx = context.Context(
        interferometer=Interferometer(
            l=l,
            λ=0 * u.m, # Unused
            Δλ=0 * u.m, # Unused
            telescopes=telescope.get_VLTI_UTs(), # Unused
            kn=KernelNuller(
                φ=np.zeros(14) * u.m, # Unused
                σ=np.zeros(14) * u.m, # Unused
            ),
        ),
        target=Target(
            m=0 * u.mag, # Unused
            δ=δ,
            companions=[], # Unused
        ),
        h=h,
        Δh=Δh,
        e=0*u.s, # Unused
        Γ=0*u.nm, # Unused
    )
    
    # GUI elements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Latitude
    l_slider = widgets.FloatSlider(
        value=l.to(u.deg).value,
        min=-90,
        max=90,
        step=0.01,
        description='Latitude (deg):',
    )

    # Declination
    δ_slider = widgets.FloatSlider(
        value=δ.to(u.deg).value,
        min=-90,
        max=90,
        step=0.01,
        description='Declination (deg):',
    )

    reset = widgets.Button(description="Reset to default")
    plot = widgets.Image(width=500,height=500)

    # Callbacks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def update_plot(*_):

        ctx = copy(ref_ctx)

        ctx.interferometer.l = l_slider.value * u.deg
        ctx.target.δ = δ_slider.value * u.deg

        plot.value = ctx.plot_projected_positions(N=11, return_image=True)

    def reset_values(*_):
        l_slider.value = l.to(u.deg).value
        δ_slider.value = δ.to(u.deg).value

    # Triggers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    reset.on_click(reset_values)
    l_slider.observe(update_plot, 'value')
    δ_slider.observe(update_plot, 'value')

    # Display ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    display(widgets.VBox([l_slider, δ_slider, reset, plot]))
    update_plot()

    return