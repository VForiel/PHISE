# External libs
import numpy as np
import astropy.units as u
from ipywidgets import widgets
from IPython.display import display
from io import BytesIO
from copy import deepcopy as copy

# Internal libs
from src import Telescope
from src import Context
from . import contexts

def gui(
        ctx:Context=None,
        n = 10
    ) -> None:
    """
    GUI to visualize the projected positions of the telescopes in the array.

    Parameters
    ----------
    ctx: Context
        Context object containing the interferometer and target information.
        If None, a default context is used.
    n: int
        Number of telescopes in the interferometer. Default is 10.
    """

    # Set default values ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    
    if ctx is None:
        ref_ctx = contexts.get_VLTI()
    else:
        ref_ctx = copy(ctx)


    # GUI elements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Latitude
    l_slider = widgets.FloatSlider(
        value=ref_ctx.interferometer.l.to(u.deg).value,
        min=-90,
        max=90,
        step=0.01,
        description='Latitude (deg):',
    )

    # Declination
    δ_slider = widgets.FloatSlider(
        value=ref_ctx.target.δ.to(u.deg).value,
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

        plot.value = ctx.plot_projected_positions(N=n, return_image=True)

    def reset_values(*_):
        l_slider.value = ref_ctx.interferometer.l.to(u.deg).value
        δ_slider.value = ref_ctx.target.δ.to(u.deg).value

    # Triggers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    reset.on_click(reset_values)
    l_slider.observe(update_plot, 'value')
    δ_slider.observe(update_plot, 'value')

    # Display ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    display(widgets.VBox([l_slider, δ_slider, reset, plot]))
    update_plot()

    return