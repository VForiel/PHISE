import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from copy import deepcopy as copy

#==============================================================================
# Work In Progress
#==============================================================================

def plot_detection_capability(ctx, resolution=1000, samples=100, c_min=1e-8, c_max=1e-1, r_min=1*u.mas, r_max=10*u.mas, stat_raw=np.median, stat_processed=np.median):
    """Plot the detection capability curve for a given context.

    Args:
        ctx: PHISE context object containing the instrument and observation parameters.
        resolution (int): Number of points to sample in the contrast-separation space.
        samples (int): Number of samples to draw for each point in the contrast-separation space.
        c_min (float): Minimum contrast to consider (default: 1e-8).
        c_max (float): Maximum contrast to consider (default: 1e-1).
        r_min (u.Quantity): Minimum angular separation to consider (default: 1 mas).
        r_max (u.Quantity): Maximum angular separation to consider (default: 10 mas).
        stat_raw: Statistic function to apply to raw chip outputs distributions (default: np.median).
        stat_processed: Statistic function to apply to processed chip outputs distributions (default: np.median).
    """

    ctx_h0 = copy(ctx)
    ctx_h1 = copy(ctx)

    # Forcing context to have only one companion
    ctx_h0.target.companions = []
    ctx_h1.target.companions = [
        phise.Companion(c=c_min, ρ=r_min, θ=0*u.deg, name='Test Companion')
    ]
    
    # Generate contrast and separation axis
    contrast_ramp = np.logspace(np.log10(c_min), np.log10(c_max), resolution)
    separation_ramp = np.linspace(r_min.to(u.mas).value, r_max.to(u.mas).value, resolution) * u.mas

    # Maps declaration (one for each raw output, one for each processed output, and one for the total (best of each output))
    n_maps = ctx.chip.nb_raw_outputs + ctx.chip.nb_processed_outputs + 1
    maps = np.empty((n_maps, resolution, resolution))
    fig, axs = plt.subplots(1, n_maps, figsize=(5*n_maps, 5))

    for i, ρ in enumerate(separation_ramp):
        for j, c in enumerate(contrast_ramp):
            ctx_h1.target.companions[0].c = c
            ctx_h1.target.companions[0].ρ = ρ

            raw_outputs = np.empty((samples, ctx.chip.nb_raw_outputs))
            processed_outputs = np.empty((samples, ctx.chip.nb_processed_outputs))

            for k in range(samples):
                ctx_h1.target.companions[0].c = c
                raw_outputs[k] = ctx_h1.observe().chip.raw_outputs
                processed_outputs[k] = ctx_h1.observe().chip.processed_outputs

            # Fill maps with the chosen statistic of the distributions
            maps[0, i, j] = stat_raw(raw_outputs)
            maps[1:1+ctx.chip.nb_raw_outputs, i, j] = stat_raw(raw_outputs, axis=0)
            maps[1+ctx.chip.nb_raw_outputs:, i, j] = stat_processed(processed_outputs, axis=0)