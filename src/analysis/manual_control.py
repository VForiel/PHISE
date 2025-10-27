"""Interface interactive pour contrôle manuel et visualisation.

Ce module expose des interfaces Jupyter/IPython (widgets) pour tester
manuellement des configurations d'entrée, observer les sorties du
nuller et visualiser phases, intensités et cartes.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
try:
    plt.rcParams['image.origin'] = 'lower'
except Exception:
    pass
import ipywidgets as widgets
import astropy.units as u
from copy import deepcopy as copy
from phise.classes import Context
from phise.classes import Companion, Target, Telescope, SuperKN, Interferometer, Camera
from phise.modules import *

def gui(λ: u.Quantity=None, φ: u.Quantity=None, σ: u.Quantity=None):
    """"gui.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    ref_ctx = Context.get_VLTI()
    if λ is not None:
        ref_ctx.interferometer.λ = λ
    if φ is not None:
        ref_ctx.interferometer.kn.φ = φ
    if σ is not None:
        ref_ctx.interferometer.kn.σ = σ
    step = 1e-20
    IA_sliders = [widgets.FloatSlider(value=0.5, min=0, max=0.5, step=step, description=f'I{i + 1}', continuous_update=False) for i in range(4)]
    IP_sliders = [widgets.FloatSlider(value=0, min=0, max=λ.value, step=step, description=f'I{i + 1}', continuous_update=False) for i in range(4)]
    P_sliders = [widgets.FloatSlider(value=0, min=0, max=λ.value, step=step, description=f'P{i + 1}', continuous_update=False) for i in range(14)]
    for i in range(14):
        P_sliders[i].value = ref_ctx.interferometer.kn.φ[i].to(λ.unit).value

    def beam_repr(beam: complex) -> str:
        """"beam_repr.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        return f'<b>{np.abs(beam):.2e}</b> * exp(<b>{np.angle(beam) / np.pi:.2f}</b> pi i)'
    inputs = [widgets.HTML(value=f' ') for _ in range(4)]
    null_outputs = [widgets.HTML(value=f' ') for _ in range(4)]
    dark_outputs = [widgets.HTML(value=f' ') for _ in range(6)]
    kernel_outputs = [widgets.HTML(value=f' ') for _ in range(3)]

    def update_gui(*args):
        """"update_gui.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
        ctx = copy(ref_ctx)
        ψ = np.array([IA_sliders[i].value * np.exp(1j * IP_sliders[i].value / λ.value * 2 * np.pi) for i in range(4)])
        for i in range(14):
            ctx.interferometer.kn.φ[i] = P_sliders[i].value * λ.unit
        (n, d, b) = ctx.interferometer.kn.get_output_fields(ψ=ψ, λ=λ)
        k = np.array([np.abs(d[2 * i]) ** 2 - np.abs(d[2 * i + 1]) ** 2 for i in range(3)])
        for (i, beam) in enumerate(ψ):
            inputs[i].value = f'<b>Input {i + 1} -</b> Amplitude: <code>{beam_repr(beam)}</code> Intensity: <code><b>{np.abs(beam) ** 2 * 100:.1f}%</b></code>'
        null_outputs[0].value = f'<b>N3a -</b> Amplitude: <code>{beam_repr(b)}</code> Intensity: <code><b>{np.abs(b) ** 2 * 100:.1f}%</b></code> <b><- Bright channel</b>'
        for (i, beam) in enumerate(n):
            null_outputs[i + 1].value = f"<b>N{(i - 1) // 2 + 4}{['a', 'b'][(i + 1) % 2]} -</b> Amplitude: <code>{beam_repr(beam)}</code> Intensity: <code><b>{np.abs(beam) ** 2 * 100:.1f}%</b></code>"
        for (i, beam) in enumerate(d):
            dark_outputs[i].value = f'<b>Dark {i + 1} -</b> Amplitude: <code>{beam_repr(beam)}</code> Intensity: <code><b>{np.abs(beam) ** 2 * 100:.1f}%</b></code>'
        for (i, beam) in enumerate(k):
            kernel_outputs[i].value = f'<b>Kernel {i + 1} -</b> Value: <code>{beam:.2e}</code>  KN depth: <code>{beam / np.abs(b) ** 2:.2e}</code>'
        phases.value = ctx.interferometer.kn.plot_output_phase(λ=λ, plot=False, ψ=ψ)
        for i in range(len(ψ)):
            plt.imshow([[np.abs(ψ[i]) ** 2]], cmap='hot', vmin=0, vmax=np.sum(np.abs(ψ) ** 2))
            plt.savefig(fname=f'docs/img/tmp.png', format='png')
            plt.close()
            with open('docs/img/tmp.png', 'rb') as file:
                image = file.read()
                photometric_cameras[i].value = image
        for i in range(len(n) + 1):
            if i == 0:
                plt.imshow([[np.abs(b) ** 2]], cmap='hot', vmin=0, vmax=np.sum(np.abs(n) ** 2) + np.abs(b) ** 2)
            else:
                plt.imshow([[np.abs(n[i - 1]) ** 2]], cmap='hot', vmin=0, vmax=np.sum(np.abs(n) ** 2) + np.abs(b) ** 2)
            plt.savefig(fname=f'docs/img/tmp.png', format='png')
            plt.close()
            with open('docs/img/tmp.png', 'rb') as file:
                image = file.read()
                null_cameras[i].value = image
        for i in range(len(d)):
            plt.imshow([[np.abs(d[i]) ** 2]], cmap='hot', vmin=0, vmax=np.sum(np.abs(d) ** 2))
            plt.savefig(fname=f'docs/img/tmp.png', format='png')
            plt.close()
            with open('docs/img/tmp.png', 'rb') as file:
                image = file.read()
                dark_cameras[i].value = image
        for i in range(len(k)):
            plt.imshow([[k[i]]], cmap='bwr', vmin=-np.max(np.abs(k)), vmax=np.max(np.abs(k)))
            plt.savefig(fname=f'docs/img/tmp.png', format='png')
            plt.close()
            with open('docs/img/tmp.png', 'rb') as file:
                image = file.read()
                kernel_cameras[i].value = image
        os.remove('docs/img/tmp.png')
        return (b, d)
    photometric_cameras = [widgets.Image(width=50, height=50) for _ in range(4)]
    null_cameras = [widgets.Image(width=50, height=50) for _ in range(4)]
    dark_cameras = [widgets.Image(width=50, height=50) for _ in range(6)]
    kernel_cameras = [widgets.Image(width=50, height=50) for _ in range(3)]
    phases = widgets.Image()
    vbox = widgets.VBox([widgets.HTML('<h1>Inputs</h1>'), widgets.HTML('Amplitude:'), widgets.HBox(IA_sliders[:4]), widgets.HTML('Phase:'), widgets.HBox(IP_sliders[:4]), *[widgets.HBox([photometric_cameras[i], x]) for (i, x) in enumerate(inputs)], widgets.HTML('<h1>Phases</h1>'), phases, widgets.HTML('<h1>Nuller</h1>'), widgets.HBox(P_sliders[:4]), widgets.HBox(P_sliders[4:8]), *[widgets.HBox([null_cameras[i], x]) for (i, x) in enumerate(null_outputs)], widgets.HTML('<h1>Recombiner</h1>'), widgets.HBox(P_sliders[8:11]), widgets.HBox(P_sliders[11:14]), *[widgets.HBox([dark_cameras[i], x]) for (i, x) in enumerate(dark_outputs)], widgets.HTML('<h1>Kernels</h1>'), *[widgets.HBox([kernel_cameras[i], x]) for (i, x) in enumerate(kernel_outputs)]])
    for widget in P_sliders:
        widget.observe(update_gui, 'value')
    for widget in IA_sliders:
        widget.observe(update_gui, 'value')
    for widget in IP_sliders:
        widget.observe(update_gui, 'value')
    update_gui()
    return vbox