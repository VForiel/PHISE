import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

from src.classes.scene import Scene

def plot(scene_ideal:Scene, scene_perturbed:Scene, scene_obs:Scene, scene_gen:Scene, limit:u.Quantity):
    """
    Plot the sensitivity to input noise

    Parameters
    ----------
    - scene_ideal: Scene without shift perturbation
    - scene_perturbed: Scene with shift perturbation
    - scene_obs: Scene with calibrated shifters using obstruction algorithm
    - scene_gen: Scene with calibrated shifters using genetic algorithm
    - limit: Maximum input OPD RMS

    Returns
    -------
    - None
    """

    input_ce_rms_range, step = np.linspace(0, limit.to(u.nm).value, 25, retstep=True)
    input_ce_rms_range *= u.nm
    step *= u.nm
    stds = []

    _, ax = plt.subplots(1, 1, figsize=(15, 5))

    scene_ideal = scene_ideal.copy(sources = [scene_ideal.sources[0].copy()])
    scene_perturbed = scene_perturbed.copy(sources = [scene_perturbed.sources[0].copy()])
    scene_obs = scene_obs.copy(sources = [scene_obs.sources[0].copy()])
    scene_gen = scene_gen.copy(sources = [scene_gen.sources[0].copy()])

    for i, input_ce_rms in enumerate(input_ce_rms_range):

        scene_ideal.input_ce_rms = input_ce_rms
        scene_perturbed.input_ce_rms = input_ce_rms
        scene_obs.input_ce_rms = input_ce_rms
        scene_gen.input_ce_rms = input_ce_rms

        dists_ideal = scene_ideal.instant_serie_observation(1000)
        dists_perturbated = scene_perturbed.instant_serie_observation(1000)
        dists_obs = scene_obs.instant_serie_observation(1000)
        dists_gen = scene_gen.instant_serie_observation(1000)

        kernel_dist_ideal = np.concatenate([*dists_ideal['kernels']]) / (scene_ideal.f * scene_ideal.Δt)
        kernel_dist_perturbated = np.concatenate([*dists_perturbated['kernels']]) / (scene_perturbed.f * scene_perturbed.Δt)
        kernel_dist_obs = np.concatenate([*dists_obs['kernels']]) / (scene_obs.f * scene_obs.Δt)
        kernel_dist_gen = np.concatenate([*dists_obs['kernels']]) / (scene_gen.f * scene_gen.Δt)

        stds.append(np.std(kernel_dist_ideal))
        stds.append(np.std(kernel_dist_perturbated))
        stds.append(np.std(kernel_dist_obs))
        stds.append(np.std(kernel_dist_gen))

        ax.scatter(np.random.normal(input_ce_rms.value - 1.5*step.value/5, step.value/20, len(kernel_dist_ideal)), kernel_dist_ideal, color='tab:green', s=5 if i == 0 else 0.1, alpha=1 if i==0 else 1, label="Ideal" if i == 0 else None)
        ax.scatter(np.random.normal(input_ce_rms.value - 0.5*step.value/5, step.value/20, len(kernel_dist_perturbated)), kernel_dist_perturbated, color='tab:red', s=5 if i == 0 else 0.1, alpha=1 if i==0 else 1, label="Perturbated" if i == 0 else None)
        ax.scatter(np.random.normal(input_ce_rms.value + 0.5*step.value/5, step.value/20, len(kernel_dist_obs)), kernel_dist_obs, color='tab:blue', s=5 if i == 0 else 0.1, alpha=1 if i==0 else 1, label="Obs. calibration" if i == 0 else None)
        ax.scatter(np.random.normal(input_ce_rms.value + 1.5*step.value/5, step.value/20, len(kernel_dist_gen)), kernel_dist_gen, color='tab:orange', s=5 if i == 0 else 0.1, alpha=1 if i==0 else 1, label="Gen. calibration" if i == 0 else None)
        
        ax.boxplot(kernel_dist_ideal, vert=True,       positions=[input_ce_rms.value - 1.5*step.value/5],widths=step.value/5, showfliers=False, manage_ticks=False)
        ax.boxplot(kernel_dist_perturbated, vert=True, positions=[input_ce_rms.value - 0.5*step.value/5],widths=step.value/5, showfliers=False, manage_ticks=False)
        ax.boxplot(kernel_dist_obs, vert=True,         positions=[input_ce_rms.value + 0.5*step.value/5],widths=step.value/5, showfliers=False, manage_ticks=False)
        ax.boxplot(kernel_dist_gen, vert=True,         positions=[input_ce_rms.value + 1.5*step.value/5],widths=step.value/5, showfliers=False, manage_ticks=False)

    ax.set_ylim(-max(stds), max(stds))
    ax.set_xlabel(f"Input OPD RMS ({input_ce_rms_range.unit})")
    ax.set_ylabel("Kernel intensity (photons)")
    ax.set_title("Sensitivity to noise")
    ax.legend()
