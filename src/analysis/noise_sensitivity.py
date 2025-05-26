import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.origin'] = 'lower'
from copy import deepcopy as copy

from src.classes.context import Context
from . import default_context

def plot(ctx:Context=None, β=0.5, n=1000):
    """
    Plot the sensitivity to input noise

    Parameters
    ----------
    ctx : Context
        The context to use for the plot.
    β : float
        The beta parameter for the genetic calibration approach.
    n : int
        The number of observations for the obstruction calibration approach.

    Returns
    -------
    - None
    """

    # Studied contexts --------------------------------------------------------

    # Perturbated context (with manufacturing defects)
    if ctx is None:
        ctx_perturbated = default_context.get()
    else:
        ctx_perturbated = copy(ctx)
    ctx_perturbated.name = "Perturbated"

    # Instant serie -> observation windows = exposure time
    ctx_perturbated.Δh = ctx_perturbated.interferometer.camera.e.to(u.hour).value * u.hourangle
    # No companions
    ctx_perturbated.target.companions = []

    # Ideal context (perfect kernel nuller)
    ctx_ideal = copy(ctx_perturbated)
    ctx_ideal.name = "Ideal"
    ctx_ideal.interferometer.kn.σ = np.zeros(14) * u.nm
    ctx_ideal.interferometer.kn.φ = np.zeros(14) * u.nm

    # KN calibrated using genetic approach
    ctx_gen = copy(ctx_perturbated)
    ctx_gen.name = "Genetic"
    ctx_gen.calibrate_gen(β=β)

    # KN calibrated using obstruction approach
    ctx_obs = copy(ctx_perturbated)
    ctx_obs.name = "Obstruction"
    ctx_obs.calibrate_obs(n=n)



    # ⚠️🧹 A supprimer .......................................................
    # ctx_perturbated.interferometer.kn.σ = np.zeros(14) * u.nm
    # ctx_perturbated.interferometer.kn.φ = np.zeros(14) * u.nm
    #..........................................................................



    contexts = [ctx_ideal, ctx_gen, ctx_obs]#, ctx_perturbated]
    colors = ['tab:green', 'tab:blue', 'tab:orange']#, 'tab:red']

    # Input perturbation ------------------------------------------------------

    Γ_range, step = np.linspace(0, ctx_perturbated.Γ.to(u.nm).value, 10, retstep=True)
    Γ_range *= u.nm
    step *= u.nm
    stds = []

    _, ax = plt.subplots(1, 1, figsize=(15, 5))

    for i, Γ in enumerate(Γ_range):


        for c, ctx in enumerate(contexts):
            ctx.Γ = Γ

            _, k_data, b_data = ctx.observation_serie(n=1000)

            data = np.empty_like(k_data)

            for n in range(k_data.shape[0]):
                for h in range(k_data.shape[1]):
                    data[n, h] = k_data[n, h, :] / b_data[n, h] # Kernel output to kernel depth

            data = data.flatten()

            stds.append(np.std(data))

            x_dispersion = np.random.normal(
                Γ.value + (c - 1.5)*step.value/5,
                step.value/20,
                len(data)
            )

            ax.scatter(
                x_dispersion,
                data,
                color=colors[c],
                s=5 if i == 0 else 0.1,
                alpha=1 if i==0 else 1,
                label=ctx.name if i == 0 else None
            )

            ax.boxplot(
                data,
                vert=True,
                positions=[Γ.value + (c - 1.5)*step.value/5],
                widths=step.value/5,
                showfliers=False,
                manage_ticks=False
            )

    ax.set_ylim(-max(stds), max(stds))
    ax.set_xlabel(f"Input OPD RMS ({Γ_range.unit})")
    ax.set_ylabel("Kernel-Null depth")
    ax.set_title("Sensitivity to noise")
    ax.legend()
