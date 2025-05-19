# External libs
import numpy as np
import astropy.units as u
from ipywidgets import widgets
from IPython.display import display
from io import BytesIO
from copy import deepcopy as copy

# Internal libs
from .. import *
from . import default_context

def gui(
        r:u.Quantity = None,
        h:u.Quantity = None,
        Δh:u.Quantity = None,
        l:u.Quantity = None,
        δ:u.Quantity = None,
    ) -> None:
    """
    GUI to visualize the projected positions of the telescopes in the array.

    Parameters
    ----------
    r : `astropy.units.Quantity`, optional
        Positions of the telescopes in the array. If None, the default positions
        are used.
    h : `astropy.units.Quantity`, optional
        Hour angle of the target. If None, the default value is used.
    Δh : `astropy.units.Quantity`, optional
        Range of hour angles to plot. If None, the default value is used.
    l : `astropy.units.Quantity`, optional
        Latitude of the target. If None, the default value is used.
    δ : `astropy.units.Quantity`, optional
        Declination of the target. If None, the default value is used.
    """

    # Set default values ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ref_ctx = default_context.get()

    if r is not None:
        ref_ctx.interferometer.telescopes = [Telescope(a=1*u.m**2, r=pos) for pos in r]
    if h is not None:
        ref_ctx.h = h
    if Δh is not None:
        ref_ctx.Δh = Δh
    if l is not None:
        ref_ctx.interferometer.l = l
    if δ is not None:
        ref_ctx.target.δ = δ
    
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