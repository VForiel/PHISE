"""Analyse temporelle et outils pour la réponse horaire du nuller.

Interfaces graphiques et fonctions d'ajustement pour étudier la
variation des kernels en fonction de l'angle horaire et des
compagnes artificielles.
"""
from io import BytesIO
import numpy as np
from astropy import units as u
from phise import Context
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from copy import deepcopy as copy
import ipywidgets as widgets
from IPython.display import display
from phise.classes import Companion

def gui(ctx: Context=None):
    """
    GUI for the temporal response analysis.

    Parameters
    ----------
    ctx : Context
        The context to use for the analysis.
    """
    if ctx is None:
        ctx = Context.get_VLTI()
        ctx.Δh = 24 * u.hourangle
        ctx.interferometer.kn.σ = np.zeros(14) * u.nm
        ctx.interferometer.kn.φ = np.zeros(14) * u.um
    else:
        ctx = copy(ctx)
    if len(ctx.target.companions) > 3:
        print('Limiting the number of companions to 3 for simplicity.')
        ctx.target.companions = ctx.target.companions[:3]
    ctx.Γ = 0 * u.nm
    nb_companion_selector = widgets.Dropdown(options=['1', '2', '3'], value='1', description='Companions:', disabled=False)
    companion_parameters_sliders = []
    for i in range(3):
        θ_slider = widgets.FloatSlider(value=0.0, min=-180.0, max=180.0, step=0.1, description=f'θ{i + 1} (deg):', continuous_update=False)
        ρ_slider = widgets.FloatSlider(value=2.0, min=0.0, max=10.0, step=0.01, description=f'ρ{i + 1} (mas):', continuous_update=False)
        c_slider = widgets.FloatSlider(value=-6, min=-12, max=0, step=1, description=f'c{i + 1} (10^x):', continuous_update=False)
        companion_parameters_sliders.append((θ_slider, ρ_slider, c_slider))
    transmission_plot = widgets.Image()
    temporal_response_plot = widgets.Image()
    reset_button = widgets.Button(description='Reset Values', button_style='danger', tooltip='Reset all values to default')
    status_label = widgets.Label(value='Running... ⌛')

    def update_plot(*args):
        """"update_plot.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        status_label.value = 'Running... ⌛'
        tmp_ctx = copy(ctx)
        nb_companions = int(nb_companion_selector.value)
        tmp_ctx.target.companions = []
        for (i, (θ_slider, ρ_slider, c_slider)) in enumerate(companion_parameters_sliders[:nb_companions]):
            θ = θ_slider.value * u.deg
            ρ = ρ_slider.value * u.mas
            c = 10 ** c_slider.value
            tmp_ctx.target.companions.append(Companion(c=c, ρ=ρ, θ=θ, name=f'Companion {i + 1}'))
        (img, txt) = tmp_ctx.plot_transmission_maps(N=100, return_plot=True)
        transmission_plot.value = img
        (_, axs) = plt.subplots(3, 1, figsize=(10, 10))
        tmp_ctx.interferometer.camera.e = ctx.Δh.to(u.hourangle).value * u.hour / 100
        for i in range(nb_companions + 1):
            if nb_companions == 1 and i == 1:
                continue
            tmp2_ctx = copy(tmp_ctx)
            if i < nb_companions:
                tmp2_ctx.target.companions = [tmp_ctx.target.companions[i]]
            else:
                tmp2_ctx.target.companions = tmp_ctx.target.companions
            (d, k, b) = tmp2_ctx.observation_serie(n=1)
            k = k[0, :, :]
            b = b[0, :]
            h_range = tmp_ctx.get_h_range()
            for kernel in range(3):
                k[:, kernel] /= b
                if i < nb_companions:
                    axs[kernel].plot(h_range, k[:, kernel], label=f'Companion {i + 1}', alpha=0.5)
                else:
                    axs[kernel].plot(h_range, k[:, kernel], label='Total Response', alpha=0.5, linestyle='--', color='k')
            for ax in axs:
                ax.set_xlabel('Hour Angle (h)')
                ax.set_ylabel('Kernel Value')
                ax.legend()
        plot = BytesIO()
        plt.savefig(plot, format='png')
        plt.close()
        temporal_response_plot.value = plot.getvalue()
        status_label.value = 'Done ✅'

    def reset_values():
        """"reset_values.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        nb_companion_selector.value = len(ctx.target.companions)
        for (i, (θ_slider, ρ_slider, c_slider)) in enumerate(companion_parameters_sliders):
            if i >= len(ctx.target.companions):
                θ_slider.value = 0.0
                ρ_slider.value = 2.0
                c_slider.value = -6
            else:
                θ_slider.value = ctx.target.companions[i].θ.to(u.deg).value
                ρ_slider.value = ctx.target.companions[i].ρ.to(u.mas).value
                c_slider.value = ctx.target.companions[i].c
        update_plot()
    nb_companion_selector.observe(update_plot)
    for (θ_slider, ρ_slider, c_slider) in companion_parameters_sliders:
        θ_slider.observe(update_plot)
        ρ_slider.observe(update_plot)
        c_slider.observe(update_plot)
    reset_button.on_click(lambda x: reset_values())
    display(widgets.VBox([widgets.Label('Select the number of companions:'), nb_companion_selector, *[widgets.HBox([θ_slider, ρ_slider, c_slider]) for (θ_slider, ρ_slider, c_slider) in companion_parameters_sliders], widgets.HBox([reset_button, status_label]), widgets.Label('Transmission Maps (at h=0):'), transmission_plot, widgets.Label('Temporal Response:'), temporal_response_plot]))
    update_plot()

def fit(ctx: Context, θ_guess: u.Quantity=0 * u.rad, ρ_guess: u.Quantity=2 * u.mas, c_guess: float=1e-06):
    """"fit.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    ideal_ctx = copy(ctx)
    ideal_ctx.interferometer.kn.σ = np.zeros(14) * u.nm
    ideal_ctx.interferometer.kn.φ = np.zeros(14) * u.um
    ideal_ctx.Γ = 0 * u.nm
    ideal_ctx.target.name = 'Ideal Target'
    selected_kernel = 0

    def model(params):
        """"model.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        (θ, ρ) = params
        ideal_ctx.target.companions = [Companion(c=c_guess, ρ=ρ * u.mas, θ=θ * u.deg, name='Companion')]
        (_, k, _) = ideal_ctx.observation_serie(n=1)
        return k[0, :, selected_kernel]

    def cauchy_loss(params, x, y):
        """"cauchy_loss.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        γ = np.median(np.abs(y - np.median(y)))
        residuals = y - model(params)
        return np.sum(np.log(1 + (residuals / γ) ** 2))
    x = ctx.get_h_range()
    (d, k, b) = ctx.observation_serie(n=1)
    y = k[0, :, selected_kernel]
    c_guess = ideal_ctx.target.companions[0].c
    params = np.array([θ_guess.to(u.deg).value, ρ_guess.to(u.mas).value])
    pop = minimize(cauchy_loss, params, args=(x.to(u.hourangle).value, y)).x
    print(x.shape, y.shape)
    plt.plot(x, model(pop), label='Fit', color='red')
    plt.plot(x, ideal_ctx.observation_serie(n=1)[1][0, :, selected_kernel], label='Ideal', color='k', linestyle='--')
    plt.xlabel('Hour Angle')
    plt.ylabel('Kernel Value')
    plt.legend()
    print('Optimized parameters:', pop)
    print(ctx.target)