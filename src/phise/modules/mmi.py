"""
Multi-Mode Interferometer (MMI) simulation module.

This module provides comprehensive tools for simulating light propagation through
multi-mode interference couplers using eigenmode expansion with finite-difference
step-index modes.

Core Functions
--------------
- simulate() : Full MMI simulation for given input amplitudes
- calibrate_input_phases_genetic() : Optimize input phases for nulling
- plot_mmi_interactive() : Generate comprehensive visualization

The module uses mode decomposition to track field evolution through the MMI,
with mode-dependent effective indices for accurate phase evolution.

References
----------
1. Soref, R., et al. (1991). "Silicon waveguide FIR filters." Journal of Lightwave Technology, 9(4), 571-576.
2. Snyder, A. W., & Love, J. D. (2012). "Optical Waveguide Theory." Springer Science+Business Media.
3. Marcuse, D. (1977). "Loss analysis of single-mode fiber splices." The Bell System Technical Journal, 56(5), 703-718.

Legacy Functions (PHISE v0.1)
-----------------------------
- nuller_2x2() : Ideal 2x2 Hadamard matrix
- cross_recombiner_2x2() : Ideal 2x2 cross recombiner with quadrature phase

Imported from HELIOS (github.com/VForiel/HELIOS), adapted for PHISE.
"""

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import tempfile
import os


# Legacy ideal matrix functions for backward compatibility
@nb.njit()
def nuller_2x2(beams: np.ndarray) -> np.ndarray:
    """Ideal 2x2 nuller (Hadamard) applied to two complex beams.

    Args:
        beams: Array of shape (2,) or (2, M) of input complex amplitudes.

    Returns:
        Complex array of shape (2,) or (2, M):
        - index 0: bright (constructive) output,
        - index 1: dark (destructive) output.
    """
    N = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    return N @ beams


@nb.njit()
def cross_recombiner_2x2(beams: np.ndarray) -> np.ndarray:
    """2x2 cross recombiner (MMI) with ideal quadrature phase.

    Args:
        beams: Array of shape (2,) or (2, M) of input complex amplitudes.

    Returns:
        Complex array of shape (2,) or (2, M) for MMI outputs.
    """
    θ: float = np.pi / 2
    S = 1 / np.sqrt(2) * np.array([[np.exp(1j * θ / 2), np.exp(-1j * θ / 2)], [np.exp(-1j * θ / 2), np.exp(1j * θ / 2)]])
    return S @ beams


# New comprehensive MMI simulation functions

def _wrap_phase_radians(phases_rad):
    """Wrap phases to [0, 2π)."""
    phases = np.asarray(phases_rad, dtype=float)
    return np.mod(phases, 2 * np.pi)


def _calibrate_phases_genetic_like(
    evaluate_metric,
    n_phases,
    beta=0.8,
    initial_step=np.pi / 2,
    epsilon=1e-4,
    initial_phases=None,
    fixed_indices=None,
    max_outer_iterations=200,
    verbose=False,
):
    """Calibrate phase shifters using genetic-like coordinate descent."""
    if not (0.5 <= beta < 1.0):
        raise ValueError("beta must be in the range [0.5, 1[")
    if n_phases <= 0:
        raise ValueError(f"n_phases must be positive, got {n_phases}.")
    if initial_step <= 0:
        raise ValueError(f"initial_step must be > 0, got {initial_step}.")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}.")
    if max_outer_iterations <= 0:
        raise ValueError(f"max_outer_iterations must be > 0, got {max_outer_iterations}.")

    if initial_phases is None:
        phases = np.zeros(n_phases, dtype=float)
    else:
        phases = np.asarray(initial_phases, dtype=float).copy()
        if phases.shape != (n_phases,):
            raise ValueError(f"initial_phases must have shape ({n_phases},), got {phases.shape}.")

    phases = _wrap_phase_radians(phases)

    fixed = set(fixed_indices or [])
    variable_indices = [i for i in range(n_phases) if i not in fixed]

    metric_history = []
    phases_history = []

    best_metric = float("inf")
    best_phases = phases.copy()

    step = float(initial_step)
    outer_it = 0

    while (step > epsilon) and (outer_it < max_outer_iterations):
        if verbose:
            print(f"--- Iteration {outer_it} --- Δφ={step:.3e} rad")

        for i in variable_indices:
            phases_pos = phases.copy()
            phases_neg = phases.copy()
            phases_pos[i] = (phases_pos[i] + step) % (2 * np.pi)
            phases_neg[i] = (phases_neg[i] - step) % (2 * np.pi)

            m_old = float(evaluate_metric(phases))
            m_pos = float(evaluate_metric(phases_pos))
            m_neg = float(evaluate_metric(phases_neg))

            metric_history.append(m_old)
            phases_history.append(phases.copy())

            if m_old < best_metric:
                best_metric = m_old
                best_phases = phases.copy()
            if m_pos < best_metric:
                best_metric = m_pos
                best_phases = phases_pos.copy()
            if m_neg < best_metric:
                best_metric = m_neg
                best_phases = phases_neg.copy()

            if verbose:
                print(f"Phase {i}: {m_neg:.3e} | {m_old:.3e} | {m_pos:.3e}")

            # Minimize metric
            if (m_pos < m_old) and (m_pos < m_neg):
                phases = phases_pos
            elif (m_neg < m_old) and (m_neg < m_pos):
                phases = phases_neg

            for j in fixed:
                phases[j] = float(_wrap_phase_radians(phases[j]))

        step *= beta
        outer_it += 1

    # Record final state
    metric_history.append(float(evaluate_metric(phases)))
    phases_history.append(phases.copy())
    if metric_history[-1] < best_metric:
        best_metric = metric_history[-1]
        best_phases = phases.copy()

    return {
        "metric": np.asarray(metric_history, dtype=float),
        "phases": np.asarray(phases_history, dtype=float),
        "best_metric": float(best_metric),
        "best_phases": _wrap_phase_radians(best_phases),
    }


def calibrate_input_phases_genetic(
    N=4,
    M=4,
    L=None,
    W=10.0e-6,
    n_core=2.0458,
    delta_n=0.0958,
    wavelength=1.55e-6,
    input_amplitudes=None,
    bright_output_idx=0,
    num_modes=50,
    num_z_steps=None,
    z_resolution=None,
    Din=None,
    Dout=None,
    Sin=None,
    Sout=None,
    beta=0.8,
    initial_step=np.pi / 2,
    epsilon=1e-4,
    verbose=False,
):
    """Calibrate input phases to redirect flux to a bright output.
    
    The objective is to minimize: metric = sum(null_outputs) / bright_output
    """
    n_clad = n_core - delta_n
    
    if not (0 <= bright_output_idx < M):
        raise ValueError(f"bright_output_idx must be in [0, {M-1}], got {bright_output_idx}.")

    n_eff = 0.7 * n_core + 0.3 * n_clad
    
    if L is None:
        L_pi = 4 * n_eff * W**2 / (3 * wavelength)
        L = L_pi / 2

    if input_amplitudes is None:
        input_amplitudes = [1.0 / np.sqrt(N)] * N
    if len(input_amplitudes) != N:
        raise ValueError(f"Length of input_amplitudes ({len(input_amplitudes)}) must match N ({N})")

    input_amplitudes = np.asarray(input_amplitudes, dtype=complex)
    magnitudes = np.abs(input_amplitudes)
    start_phases = _wrap_phase_radians(np.angle(input_amplitudes))

    def evaluate_metric(phases_rad):
        phases_rad = _wrap_phase_radians(phases_rad)
        amps = magnitudes * np.exp(1j * phases_rad)

        out = simulate(
            N=N, M=M, L=L, W=W, n_core=n_core, delta_n=delta_n,
            wavelength=wavelength, input_amplitudes=amps, num_modes=num_modes,
            num_z_steps=num_z_steps, z_resolution=z_resolution, output_file=None,
            verbose=False, Din=Din, Dout=Dout, Sin=Sin, Sout=Sout,
        )

        intensities = np.abs(out) ** 2
        bright = float(intensities[bright_output_idx])
        null_sum = float(np.sum(intensities) - bright)

        if bright <= 0:
            return float("inf")
        return null_sum / bright

    result = _calibrate_phases_genetic_like(
        evaluate_metric=evaluate_metric, n_phases=N, beta=beta,
        initial_step=initial_step, epsilon=epsilon, initial_phases=start_phases,
        fixed_indices=None, verbose=verbose,
    )
    result["bright_output_idx"] = int(bright_output_idx)
    return result


def calibrate_n_core_and_phases(
    N=4,
    M=4,
    L=None,
    W=10.0e-6,
    n_core_min=None,
    n_core_max=None,
    n_core_initial=2.0458,
    delta_n=0.0958,
    wavelength=1.55e-6,
    input_amplitudes=None,
    bright_output_idx=0,
    num_modes=50,
    num_z_steps=None,
    z_resolution=None,
    Din=None,
    Dout=None,
    Sin=None,
    Sout=None,
    n_core_steps_coarse=20,
    gradient_convergence_threshold=1e-3,
    gradient_initial_step=0.01,
    beta=0.8,
    initial_step=np.pi / 2,
    epsilon=1e-4,
    verbose=False,
    progress_callback_coarse=None,
    progress_callback_gradient=None,
):
    """Jointly optimize core index n_core and input phases to minimize null depth.

    Strategy: two-stage search identical to HELIOS implementation.
    1) Coarse scan over n_core, running phase calibration at each sample
    2) Gradient descent on n_core, refining around the best coarse value

    Parameters are identical in meaning to calibrate_input_phases_genetic, with
    additional options for the n_core search.

    Returns a dict with coarse and gradient histories, best_n_core, best_metric,
    and best_phases.
    """

    # Defaults for coarse range
    if n_core_min is None:
        n_core_min = 1.0
    if n_core_max is None:
        n_core_max = 2.0 * n_core_initial

    if n_core_min >= n_core_max:
        raise ValueError(f"n_core_min ({n_core_min}) must be < n_core_max ({n_core_max})")
    if n_core_steps_coarse < 2:
        raise ValueError(f"n_core_steps_coarse must be >= 2, got {n_core_steps_coarse}")

    # -----------------------------
    # Stage 1: Coarse scan
    # -----------------------------
    n_core_values_coarse = np.linspace(n_core_min, n_core_max, int(n_core_steps_coarse))
    metrics_coarse = []
    all_phases_coarse = []

    for i, n_core_test in enumerate(n_core_values_coarse):
        if verbose:
            print(f"[Coarse {i+1}/{n_core_steps_coarse}] n_core = {n_core_test:.4f}")
        result_phase = calibrate_input_phases_genetic(
            N=N,
            M=M,
            L=L,
            W=W,
            n_core=n_core_test,
            delta_n=delta_n,
            wavelength=wavelength,
            input_amplitudes=input_amplitudes,
            bright_output_idx=bright_output_idx,
            num_modes=num_modes,
            num_z_steps=num_z_steps,
            z_resolution=z_resolution,
            Din=Din,
            Dout=Dout,
            Sin=Sin,
            Sout=Sout,
            beta=beta,
            initial_step=initial_step,
            epsilon=epsilon,
            verbose=False,
        )

        metrics_coarse.append(float(result_phase["best_metric"]))
        all_phases_coarse.append(np.asarray(result_phase["best_phases"], dtype=float))

        if progress_callback_coarse is not None:
            progress_callback_coarse(i + 1, int(n_core_steps_coarse))

    metrics_coarse = np.asarray(metrics_coarse, dtype=float)
    best_coarse_idx = int(np.argmin(metrics_coarse))
    best_coarse_n_core = float(n_core_values_coarse[best_coarse_idx])
    best_coarse_metric = float(metrics_coarse[best_coarse_idx])

    # -----------------------------
    # Stage 2: Gradient descent
    # -----------------------------
    def _evaluate_n_core(nc_val: float):
        res = calibrate_input_phases_genetic(
            N=N,
            M=M,
            L=L,
            W=W,
            n_core=nc_val,
            delta_n=delta_n,
            wavelength=wavelength,
            input_amplitudes=input_amplitudes,
            bright_output_idx=bright_output_idx,
            num_modes=num_modes,
            num_z_steps=num_z_steps,
            z_resolution=z_resolution,
            Din=Din,
            Dout=Dout,
            Sin=Sin,
            Sout=Sout,
            beta=beta,
            initial_step=initial_step,
            epsilon=epsilon,
            verbose=False,
        )
        return float(res["best_metric"]), np.asarray(res["best_phases"], dtype=float)

    n_core_current = best_coarse_n_core
    metric_current, phases_current = best_coarse_metric, all_phases_coarse[best_coarse_idx]
    step_size = float(gradient_initial_step)

    n_core_values_gradient = [n_core_current]
    metrics_gradient = [metric_current]
    all_phases_gradient = [phases_current]

    iteration = 0
    max_iterations = 100

    while iteration < max_iterations:
        iteration += 1

        n_plus = float(np.clip(n_core_current + step_size, n_core_min, n_core_max))
        n_minus = float(np.clip(n_core_current - step_size, n_core_min, n_core_max))

        if verbose:
            print(f"[Grad {iteration}] current={n_core_current:.4f}, step={step_size:.4f}")

        m_plus, ph_plus = _evaluate_n_core(n_plus)
        m_minus, ph_minus = _evaluate_n_core(n_minus)

        # Choose best move
        if (m_plus < metric_current) and (m_plus < m_minus):
            n_new, m_new, ph_new, direction = n_plus, m_plus, ph_plus, "+"
        elif (m_minus < metric_current) and (m_minus < m_plus):
            n_new, m_new, ph_new, direction = n_minus, m_minus, ph_minus, "-"
        else:
            step_size *= 0.5
            if progress_callback_gradient is not None:
                progress_callback_gradient(iteration, 0.0)
            if step_size < gradient_convergence_threshold:
                break
            continue

        delta = abs(n_new - n_core_current)
        n_core_current, metric_current, phases_current = n_new, m_new, ph_new
        n_core_values_gradient.append(n_core_current)
        metrics_gradient.append(metric_current)
        all_phases_gradient.append(phases_current)

        if progress_callback_gradient is not None:
            progress_callback_gradient(iteration, float(delta))

        if delta < gradient_convergence_threshold:
            break

    return {
        "n_core_values_coarse": n_core_values_coarse,
        "metrics_coarse": metrics_coarse,
        "n_core_values_gradient": np.asarray(n_core_values_gradient, dtype=float),
        "metrics_gradient": np.asarray(metrics_gradient, dtype=float),
        "best_n_core": float(n_core_current),
        "best_metric": float(metric_current),
        "best_phases": np.asarray(phases_current, dtype=float),
        "bright_output_idx": int(bright_output_idx),
    }


def _compute_mode_profile(x_grid, center, width):
    """Compute normalized Gaussian mode profile."""
    if width <= 0:
        raise ValueError(f"Mode width must be > 0, got {width}.")
    
    sigma = width / 2.0
    profile = np.exp(-((x_grid - center)**2) / (sigma**2))
    
    dx = x_grid[1] - x_grid[0] if len(x_grid) > 1 else 1.0
    norm_factor = np.sqrt(np.sum(np.abs(profile)**2) * dx)
    
    if norm_factor <= 0:
        raise ValueError("Mode profile normalization failed.")
    
    return profile / norm_factor


def _compute_symmetric_port_positions(num_ports, W, spacing, name):
    """Compute symmetric port positions centered at x=0."""
    if num_ports <= 0:
        raise ValueError(f"{name} ports must be a positive integer, got {num_ports}.")
    if W <= 0:
        raise ValueError(f"MMI width W must be positive, got {W}.")

    if spacing is None:
        spacing = W / num_ports
    if spacing <= 0:
        raise ValueError(f"{name} spacing must be > 0, got {spacing}.")

    center = 0.0
    offsets = (np.arange(num_ports, dtype=float) - 0.5 * (num_ports - 1)) * spacing
    positions = center + offsets

    eps = max(1e-15, 1e-15 * abs(W))
    min_pos = float(np.min(positions))
    max_pos = float(np.max(positions))
    if (min_pos < -W/2 - eps) or (max_pos > W/2 + eps):
        raise ValueError(
            f"{name} spacing {spacing} m is too large for W={W} m: "
            f"positions would span [{min_pos}, {max_pos}] m outside [-{W/2}, {W/2}] m."
        )

    positions = np.clip(positions, -W/2, W/2)
    return positions.tolist()


def _compute_mmi_field(N, M, L, W, n_core, delta_n, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, verbose=False, Din=None, Dout=None, Sin=None, Sout=None):
    """Core MMI field calculation."""
    n_clad = n_core - delta_n
    k0 = 2 * np.pi / wavelength
    
    if num_z_steps is None:
        if z_resolution is None:
            z_resolution = wavelength / 30.0
            if verbose:
                print(f"Using default z-resolution: lambda/30 = {z_resolution*1e6:.3f} um")
    
        num_z_steps = int(np.ceil(L / z_resolution)) + 1
        if verbose:
            print(f"Calculated num_z_steps = {num_z_steps}")

    input_positions = _compute_symmetric_port_positions(N, W, Din, name="input")
    
    x_grid = np.linspace(-W, W, 500)
    dx = x_grid[1] - x_grid[0]
    
    betas = []
    n_eff_modes = []
    modes = []

    for m in range(num_modes):
        mode_num = m + 1
        kx_m = mode_num * np.pi / W
        
        sq_term = (k0 * n_core) ** 2 - kx_m ** 2
        
        if sq_term < 0:
            betas.append(0.0)
            n_eff_modes.append(n_clad)
            modes.append(np.zeros_like(x_grid, dtype=float))
            continue
        
        beta_m = np.sqrt(sq_term)
        betas.append(beta_m)
        n_eff_m = beta_m / k0
        n_eff_modes.append(n_eff_m)
        
        phi_m = np.zeros_like(x_grid, dtype=float)
        mask_inside = (x_grid >= -W/2) & (x_grid <= W/2)
        x_shifted = x_grid[mask_inside] + W/2
        phi_m[mask_inside] = np.sqrt(2 / W) * np.sin(kx_m * x_shifted)
        
        sq_decay = kx_m ** 2 - (k0 * n_clad) ** 2
        if sq_decay > 0:
            kappa_m = np.sqrt(sq_decay)
            
            mask_left = x_grid < -W/2
            phi_m[mask_left] = np.sqrt(2 / W) * np.exp(kappa_m * (x_grid[mask_left] + W/2))
            
            mask_right = x_grid > W/2
            phi_m[mask_right] = np.sqrt(2 / W) * np.exp(-kappa_m * (x_grid[mask_right] - W/2))
        
        modes.append(phi_m)
    
    betas = np.array(betas)
    n_eff_modes = np.array(n_eff_modes)
    modes = np.array(modes)

    # Input field
    input_field = np.zeros_like(x_grid, dtype=complex)
    
    if Sin is None:
        Sin = (W / N) / 4
    
    for idx, amp in enumerate(input_amplitudes):
        if amp == 0:
            continue
        center_x = input_positions[idx]
        gauss = _compute_mode_profile(x_grid, center_x, Sin)
        input_field += amp * gauss
    
    beam_waist = Sin

    z_grid = np.linspace(0, L, num_z_steps)

    # Mode decomposition
    coeffs = np.array([
        np.trapz(input_field * np.conj(modes[m]), x_grid) for m in range(num_modes)
    ])
    
    # Propagation
    field_evolution = np.zeros((num_z_steps, len(x_grid)), dtype=complex)
    
    iterator = enumerate(z_grid)
    if verbose:
        iterator = enumerate(tqdm(z_grid, desc="Propagating", unit="step"))
        
    for iz, z in iterator:
        phase_term = np.exp(-1j * betas * z)
        weights = coeffs * phase_term
        E_z = np.dot(weights, modes)
        field_evolution[iz, :] = E_z

    output_positions = _compute_symmetric_port_positions(M, W, Dout, name="output")
    
    return z_grid, x_grid, field_evolution, output_positions, input_positions, beam_waist, dx


def _compute_input_intensity_normalization(field_z0, x_grid, W, dx):
    """Compute input power from MMI core region."""
    intensity_z0 = np.abs(field_z0)**2
    
    mask = (x_grid >= -W/2) & (x_grid <= W/2)
    input_power_core = np.sum(intensity_z0[mask]) * dx
    
    if input_power_core < 1e-15:
        return 1.0
    
    return input_power_core


def simulate(N=2, M=2, L=None, W=10.0e-6, n_core=2.0458, delta_n=0.0958, wavelength=1.55e-6, input_amplitudes=None, num_modes=50, num_z_steps=None, z_resolution=None, output_file=None, verbose=False, Din=None, Dout=None, Sin=None, Sout=None):
    """Simulates light propagation in an NxM MMI using eigenmode expansion.
    
    Parameters
    ----------
    N : int, default=2
        Number of input ports
    M : int, default=2
        Number of output ports
    L : float, optional
        Length of MMI [m]. If None, auto-calculated from paired interference
    W : float, default=10.0e-6
        Width of MMI [m]
    n_core : float, default=2.0458
        Core refractive index
    delta_n : float, default=0.0958
        Index contrast (n_core - n_clad)
    wavelength : float, default=1.55e-6
        Wavelength [m]
    input_amplitudes : array-like, optional
        Complex amplitudes for N inputs. Defaults to uniform 1/sqrt(N)
    num_modes : int, default=50
        Number of modes for decomposition
    num_z_steps : int, optional
        Number of z-propagation steps
    z_resolution : float, optional
        Z step size [m]. Defaults to wavelength/30
    output_file : str, optional
        Path to save animation (e.g., 'mmi.mp4')
    verbose : bool, default=False
        Print progress
    Din, Dout, Sin, Sout : float, optional
        Input/output spacing and waveguide widths
    
    Returns
    -------
    np.ndarray
        Complex amplitudes at the M output ports
    """
    
    n_clad = n_core - delta_n
        
    if L is None:
        n_eff = 0.7 * n_core + 0.3 * n_clad
        L_pi = 4 * n_eff * W**2 / (3 * wavelength)
        L = L_pi / 2
        if verbose:
            print(f"Auto-calculated L = {L*1e6:.2f} um")
            
    if input_amplitudes is None:
        val = 1.0 / np.sqrt(N)
        input_amplitudes = [val] * N
    
    if len(input_amplitudes) != N:
        raise ValueError(f"Length of input_amplitudes ({len(input_amplitudes)}) must match N ({N})")

    if Sout is None:
        Sout = Sin
    
    z_grid, x_grid, field_evolution, output_positions, input_positions, beam_waist, dx = _compute_mmi_field(
        N, M, L, W, n_core, delta_n, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, verbose, Din=Din, Dout=Dout, Sin=Sin, Sout=Sout
    )
    
    num_z_steps = len(z_grid)
    intensity_evolution = np.abs(field_evolution)**2
    
    input_power = _compute_input_intensity_normalization(field_evolution[0, :], x_grid, W, dx)
    if input_power > 0:
        intensity_evolution = intensity_evolution / input_power

    # Calculate output amplitudes
    output_amplitudes = []
    final_field = field_evolution[-1, :]
    
    Sout_use = Sout if Sout is not None else (Sin if Sin is not None else (W / N) / 4)
    
    if verbose:
        print(f"Output waveguide width (Sout) = {Sout_use*1e6:.3f} µm")
    
    for j in range(M):
        center_x_out = output_positions[j]
        psi_out = _compute_mode_profile(x_grid, center_x_out, Sout_use)
        overlap = np.sum(final_field * np.conj(psi_out)) * dx
        output_amplitudes.append(overlap)
        
    output_amplitudes = np.array(output_amplitudes)

    if verbose:
        print(f"Output amplitudes: {output_amplitudes}")
        print(f"Output intensities: {np.abs(output_amplitudes)**2}")

    return output_amplitudes


def _compute_single_field_wrapper(i, N, M, L, W, n_core, delta_n, wavelength, input_amplitudes, num_modes, num_z_steps, z_resolution, Din, Dout, Sin, Sout):
    """Wrapper to compute field for a single input (parallel helper)."""
    single_input = np.zeros(N, dtype=complex)
    single_input[i] = input_amplitudes[i]
    ret = _compute_mmi_field(
        N,
        M,
        L,
        W,
        n_core,
        delta_n,
        wavelength,
        single_input,
        num_modes,
        num_z_steps,
        z_resolution,
        verbose=False,
        Din=Din,
        Dout=Dout,
        Sin=Sin,
        Sout=Sout,
    )
    return ret[2]


def compute_contributions(
    N,
    M,
    L,
    W,
    n_core,
    delta_n,
    wavelength,
    input_amplitudes,
    num_modes,
    num_z_steps,
    z_resolution,
    verbose=False,
    Din=None,
    Dout=None,
    Sin=None,
    Sout=None,
):
    """Calculate MMI fields and contributions, returning raw data for custom plotting."""
    n_clad = n_core - delta_n

    if L is None:
        n_eff = 0.7 * n_core + 0.3 * n_clad
        L_pi = 4 * n_eff * W**2 / (3 * wavelength)
        L = L_pi / 2
        if verbose:
            print(f"Auto-calculated L = {L*1e6:.2f} um")
    if input_amplitudes is None:
        val = 1.0 / np.sqrt(N)
        input_amplitudes = [val] * N
    if len(input_amplitudes) != N:
        raise ValueError("Length mismatch")

    if Sout is None:
        Sout = Sin

    z_grid, x_grid, field_total_evol, output_positions, input_positions, beam_waist, dx = _compute_mmi_field(
        N,
        M,
        L,
        W,
        n_core,
        delta_n,
        wavelength,
        input_amplitudes,
        num_modes,
        num_z_steps,
        z_resolution,
        verbose,
        Din=Din,
        Dout=Dout,
        Sin=Sin,
        Sout=Sout,
    )

    num_z_steps = len(z_grid)
    intensity_total_evol = np.abs(field_total_evol) ** 2

    input_power = _compute_input_intensity_normalization(field_total_evol[0, :], x_grid, W, dx)
    if input_power > 0:
        intensity_total_evol = intensity_total_evol / input_power

    if verbose:
        print("Computing separate field contributions (Parallel)...")

    contributions_fields = Parallel(n_jobs=-1)(
        delayed(_compute_single_field_wrapper)(
            i,
            N,
            M,
            L,
            W,
            n_core,
            delta_n,
            wavelength,
            input_amplitudes,
            num_modes,
            num_z_steps,
            z_resolution,
            Din,
            Dout,
            Sin,
            Sout,
        )
        for i in range(N)
    )

    Sout_use = Sout if Sout is not None else (Sin if Sin is not None else (W / N) / 4)
    output_modes = []
    for j in range(M):
        center_x_out = output_positions[j]
        psi = _compute_mode_profile(x_grid, center_x_out, Sout_use)
        output_modes.append(psi)

    phasors = np.zeros((num_z_steps, M, N), dtype=complex)
    for iz in range(num_z_steps):
        for j in range(M):
            psi_out = output_modes[j]
            for i in range(N):
                E_i_z = contributions_fields[i][iz, :]
                coupling = np.sum(E_i_z * np.conj(psi_out)) * dx
                phasors[iz, j, i] = coupling

    Sin_final = Sin if Sin is not None else (W / N) / 4
    Sout_final = Sout_use

    return {
        "z_grid": z_grid,
        "x_grid": x_grid,
        "intensity_total_evol": intensity_total_evol,
        "phasors": phasors,
        "input_positions": input_positions,
        "output_positions": output_positions,
        "L": L,
        "W": W,
        "N": N,
        "M": M,
        "num_z_steps": num_z_steps,
        "Sin": Sin_final,
        "Sout": Sout_final,
    }


def plot_mmi_interactive(
    N,
    M,
    L,
    W,
    n_core,
    delta_n,
    wavelength,
    input_amplitudes,
    num_modes,
    num_z_steps,
    z_resolution,
    Din,
    Dout,
    Sin,
    Sout,
    verbose=False,
):
    """Generate interactive MMI visualization matching HELIOS demo."""
    data = compute_contributions(
        N=N,
        M=M,
        L=L,
        W=W,
        n_core=n_core,
        delta_n=delta_n,
        wavelength=wavelength,
        input_amplitudes=input_amplitudes,
        num_modes=num_modes,
        num_z_steps=num_z_steps,
        z_resolution=z_resolution,
        Din=Din,
        Dout=Dout,
        Sin=Sin,
        Sout=Sout,
        verbose=verbose,
    )

    z_grid = data["z_grid"]
    x_grid = data["x_grid"]
    intensity_map = data["intensity_total_evol"]
    phasors = data["phasors"]
    input_pos = data["input_positions"]
    output_pos = data["output_positions"]
    L_sim = data["L"]
    W = data["W"]
    Sin_computed = data["Sin"]
    Sout_computed = data["Sout"]

    fig = plt.figure(figsize=(12, 24))
    gs = fig.add_gridspec(6, M, height_ratios=[1.5, 1, 1, 1, 1, 1])

    ax_map = fig.add_subplot(gs[0, :])
    x_min = x_grid[0]
    x_max = x_grid[-1]
    extent = [0, L_sim * 1e6, x_min * 1e6, x_max * 1e6]

    im = ax_map.imshow(intensity_map.T, origin="lower", aspect="auto", extent=extent, cmap="inferno")
    ax_map.set_xlabel("z [um]")
    ax_map.set_ylabel("x [um] (centered at 0)")
    ax_map.axhline(y=-W / 2 * 1e6, color="white", linestyle="--", linewidth=1.5, alpha=0.7, label="MMI Core Boundary")
    ax_map.axhline(y=W / 2 * 1e6, color="white", linestyle="--", linewidth=1.5, alpha=0.7)

    sin_str = f"{Sin*1e6:.2f}" if Sin else "auto"
    sout_str = f"{Sout*1e6:.2f}" if Sout else "auto"
    n_clad_calc = n_core - delta_n
    n_eff_calc = 0.7 * n_core + 0.3 * n_clad_calc
    ax_map.set_title(
        f"Intensity Map - Centered Coords (λ={wavelength*1e6:.2f} µm, n_core={n_core:.4f}, Δn={delta_n:.4f}, n_eff={n_eff_calc:.4f})\n"
        f"Sin={sin_str} µm, Sout={sout_str} µm | White lines: MMI core [-W/2, W/2], Cyan: x=0",
        fontsize=9,
    )

    ax_map.scatter([0] * N, [p * 1e6 for p in input_pos], color="cyan", s=10, marker="o", label="Inputs")
    ax_map.scatter([L_sim * 1e6] * M, [p * 1e6 for p in output_pos], color="lime", s=10, marker="s", label="Outputs")
    ax_map.legend(loc="upper right", fontsize=8)

    ax_prof_in = fig.add_subplot(gs[1, :])
    x_display = x_grid * 1e6
    ax_prof_in.plot(x_display, intensity_map[0, :], "b-", lw=2)
    input_colors = plt.cm.get_cmap("Set3", N)
    for i, p in enumerate(input_pos):
        ax_prof_in.axvspan((p - Sin_computed / 2) * 1e6, (p + Sin_computed / 2) * 1e6, alpha=0.15, color=input_colors(i), label=f"Input {i+1}" if i < 3 else "")
    ax_prof_in.axvline(x=-W / 2 * 1e6, color="red", linestyle="--", alpha=0.5, label="MMI Core [-W/2, W/2]")
    ax_prof_in.axvline(x=W / 2 * 1e6, color="red", linestyle="--", alpha=0.5)
    ax_prof_in.axvline(x=0, color="cyan", linestyle="-", lw=2, alpha=0.7, label="Center (x=0)")
    for p in input_pos:
        ax_prof_in.axvline(x=p * 1e6, color="k", linestyle=":", alpha=0.5)
    ax_prof_in.set_xlim(-W * 1e6, W * 1e6)
    ax_prof_in.set_xlabel("x [um]")
    ax_prof_in.set_ylabel("Intensity")
    ax_prof_in.set_title("Input Profile (x) at z=0")
    ax_prof_in.legend(loc="upper right", fontsize=8)

    ax_prof = fig.add_subplot(gs[2, :])
    ax_prof.plot(x_display, intensity_map[-1, :], "b-", lw=2)
    output_colors = plt.cm.get_cmap("Set2", M)
    for j, p in enumerate(output_pos):
        ax_prof.axvspan((p - Sout_computed / 2) * 1e6, (p + Sout_computed / 2) * 1e6, alpha=0.15, color=output_colors(j), label=f"Output {j+1}" if j < 3 else "")
    ax_prof.axvline(x=-W / 2 * 1e6, color="red", linestyle="--", alpha=0.5, label="MMI Core [-W/2, W/2]")
    ax_prof.axvline(x=W / 2 * 1e6, color="red", linestyle="--", alpha=0.5)
    ax_prof.axvline(x=0, color="cyan", linestyle="-", lw=2, alpha=0.7, label="Center (x=0)")
    for p in output_pos:
        ax_prof.axvline(x=p * 1e6, color="k", linestyle=":", alpha=0.5)
    ax_prof.set_xlim(-W * 1e6, W * 1e6)
    ax_prof.set_xlabel("x [um]")
    ax_prof.set_ylabel("Intensity")
    ax_prof.set_title("Output Profile (x) at z=L")
    ax_prof.legend(loc="upper right", fontsize=8)

    colors = plt.cm.get_cmap("hsv", N + 1)
    max_val = np.max(np.abs(phasors[-1, :, :]))
    limit = max_val * 1.1 if max_val > 1e-6 else 1.0
    for j in range(M):
        ax_p = fig.add_subplot(gs[3, j], projection="polar")
        ax_p.set_title(f"Out {j+1}")
        ax_p.set_ylim(0, limit)
        for i in range(N):
            val = phasors[-1, j, i]
            ax_p.plot([0, np.angle(val)], [0, np.abs(val)], color=colors(i), lw=2, label=f"In {i+1}")
        tot = np.sum(phasors[-1, j, :])
        ax_p.plot([0, np.angle(tot)], [0, np.abs(tot)], "k--", lw=2, label="Total")
        if j == M - 1:
            ax_p.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)

    ax_z = fig.add_subplot(gs[4, :])
    ax_z.set_title("Z-Profile All Outputs")
    z_colors = plt.cm.get_cmap("tab10", M)
    max_int_z = np.max(intensity_map) * 1.1
    for j in range(M):
        x_out = output_pos[j]
        ix = np.argmin(np.abs(x_grid - x_out))
        I_z = intensity_map[:, ix]
        ax_z.plot(z_grid * 1e6, I_z, color=z_colors(j), lw=1.5, label=f"Out {j+1}")
    ax_z.axvline(x=L_sim * 1e6, color="r", linestyle="--", lw=1.0)
    ax_z.set_xlabel("z [um]")
    ax_z.set_xlim(0, L_sim * 1e6)
    ax_z.set_ylim(0, max_int_z)
    ax_z.legend(loc="upper right", fontsize=8)

    ax_power = fig.add_subplot(gs[5, :])
    ax_power.set_title("Integrated Power in Core [-W/2, W/2] vs. Propagation")
    mask_core = (x_grid >= -W / 2) & (x_grid <= W / 2)
    dx = x_grid[1] - x_grid[0]
    power_in_core_z = np.array([np.sum(intensity_map[iz, mask_core]) * dx for iz in range(len(z_grid))])
    ax_power.plot(z_grid * 1e6, power_in_core_z, "darkblue", lw=2.5, label="Power in Core")
    ax_power.axhline(y=1.0, color="green", linestyle="--", lw=2, alpha=0.7, label="Input Reference (1.0)")
    ax_power.axvline(x=L_sim * 1e6, color="r", linestyle="--", lw=1.0, alpha=0.7)
    ax_power.set_xlabel("z [um]")
    ax_power.set_ylabel("Integrated Power")
    ax_power.set_xlim(0, L_sim * 1e6)
    ax_power.set_ylim(0, min(2.0, max(power_in_core_z) * 1.2))
    ax_power.grid(True, alpha=0.3)
    ax_power.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig