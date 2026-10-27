# PHISE Documentation

**PHISE** (PHotonic Interferometry Simulation for Exoplanets) is a Python package for simulation and analysis of interferometric instruments using layered photonic chips for exoplanet detection.

PHISE was originally designed for the specific needs of a PhD thesis on photonic kernel nulling ([VForiel/PhD-Theory](https://github.com/VForiel/PhD-Theory)). Its scope reflects these needs.

> ⚠️ **Development Status Warning**
> 
> PHISE is currently under active development as part of a PhD research project by Vincent Foriel. The package is **highly unstable** and subject to breaking changes at any time without notice. The API, functionalities, and internal architecture may change significantly as research progresses.
> 
> **Important notes:**
> - This library is tightly coupled to ongoing thesis work and is not yet production-ready
> - Documentation may not always reflect the current state of the code
> - Use at your own risk
> 
> If you're visiting this documentation now, please be aware that you're looking at a work-in-progress research tool.

## Overview

PHISE provides a comprehensive framework for modeling interferometric instruments with photonic beam combiners, from the observed scene to the detector. With PHISE you can:

- **Simulate the full chain** in a single `Context` object: target and companions → telescope array → photonic chip → camera. Changing a parameter (hour angle, declination, wavelength…) automatically updates the dependent quantities.
- **Follow Earth rotation**: telescope positions are projected onto the sky plane over the whole observation. Ready-made VLTI (4 UTs) and LIFE configurations are provided.
- **Model photonic kernel nullers**:
  - `SuperKN`: 14 phase shifters, 7 outputs, 3 kernels.
  - `N4x4_T8` / `N4x4_D8`: 4×4 MMI-based chips using crosstalk matrices measured on a real component, plus an ideal variant.
- **Include manufacturing defects and active control**: intrinsic OPD errors, tunable phase shifters, input attenuation and OPD.
- **Calibrate the chip** with trial-and-error, obstruction-based least-squares fitting, or Hooke & Jeeves direct search.
- **Simulate the detector**: photon noise, dark current, read-out noise, full-well saturation, gain, dark calibration, and pixel images of each output. A helper computes a reference exposure time.
- **Visualise the instrument response**: numerical and analytical transmission maps and their gradients, phasor diagrams (Laugiergrams), projected baselines.
- **Quantify detection performance**: Monte-Carlo generation of H0/H1 observations, more than ten test statistics (Kolmogorov–Smirnov, Cramér–von Mises, Anderson–Darling, Wasserstein…), ROC curves, Neyman–Pearson analysis, and detection-capability maps in contrast vs separation.
- **Generate synthetic datasets** for machine-learning-based calibration.
- **Work fast and safely**: `astropy` units at every API boundary, `numba` JIT-compiled numerical kernels.

## Architecture

The following UML diagram shows the main components and their relationships:

```{mermaid}
classDiagram
    %% Core simulation orchestrator
    class Context {
        +Interferometer interferometer
        +Target target
        +Quantity h (hour angle)
        +Quantity Δh (time span)
        +Quantity Γ (cophasing error)
        +bool monochromatic
        +str name
        +_update_p()
        +_update_pf()
        +plot_transmission_map()
        +plot_projected_baselines()
    }
    
    %% Instrument components
    class Interferometer {
        +Quantity l (latitude)
        +Quantity λ (wavelength)
        +Quantity Δλ (bandwidth)
        +Quantity fov (field of view)
        +float η (efficiency)
        +List~Telescope~ telescopes
        +Chip chip
        +Camera camera
        +str name
    }
    
    class Telescope {
        +Quantity a (collecting area)
        +Quantity r (position)
        +str name
    }
    
    class Chip {
        <<abstract>>
    }
    
    class SuperKN {
        +Quantity φ (phase shifters)
        +Quantity σ (OPD errors)
        +Quantity λ0 (reference wavelength)
        +ndarray output_order
        +ndarray input_attenuation
        +Quantity input_opd
        +int nb_raw_outputs
        +int nb_processed_outputs
        +get_matrices()
        +apply_opd()
    }
    
    class Camera {
        +Quantity e (exposure time)
        +bool ideal
        +str name
        +detect()
    }
    
    %% Scene components
    class Target {
        +Quantity δ (declination)
        +Quantity S (spectral flux)
        +Quantity m (magnitude)
        +CompanionList companions
        +str name
        +plot()
    }
    
    class Companion {
        +float c (contrast)
        +Quantity ρ (angular separation)
        +Quantity θ (parallactic angle)
        +str name
    }
    
    class CompanionList {
        +append()
        +extend()
    }
    
    %% Numerical modules
    class coordinates {
        <<module>>
        +projected_baselines()
        +hour_angle_transform()
    }
    
    class signals {
        <<module>>
        +photon_flux()
        +signal_propagation()
    }
    
    class phase {
        <<module>>
        +optical_path_difference()
        +phase_screen()
    }
    
    class mmi {
        <<module>>
        +transfer_matrix()
        +combiner_output()
    }
    
    class test_statistics {
        <<module>>
        +null_depth()
        +detection_statistics()
    }
    
    %% Relationships
    Context *-- Interferometer : contains
    Context *-- Target : observes
    Context ..> coordinates : uses
    Context ..> signals : uses
    Context ..> phase : uses
    
    Interferometer *-- "1..*" Telescope : array of
    Interferometer *-- Chip : beam combiner
    Interferometer *-- Camera : detector
    
    Chip <|-- SuperKN : implements
    Chip ..> mmi : uses
    Chip ..> phase : uses
    
    Target *-- CompanionList : has
    CompanionList o-- "0..*" Companion : contains
    
    %% Module dependencies
    signals ..> coordinates : uses
    test_statistics ..> signals : uses

    %% Parent references (auto-update pattern)
    Telescope --> Interferometer : _parent_interferometer
    Chip --> Interferometer : _parent_interferometer
    Camera --> Interferometer : _parent_interferometer
    Interferometer --> Context : _parent_ctx
    Target --> Context : _parent_ctx
    Companion --> Target : _parent_target
```

### Key Design Patterns

**Context Auto-Update Pattern**: The `Context` class automatically propagates parameter changes through the instrument chain. When you modify observation parameters (hour angle, target position, etc.), dependent quantities are automatically recomputed.

**Units at API Boundaries**: All user-facing APIs use `astropy.units.Quantity` for physical quantities. Internally, values are converted to native Python types for performance in numba-compiled code.

**Component Hierarchy**: 
- `Context` orchestrates the simulation combining an `Interferometer` and a `Target`
- `Interferometer` manages the telescope array, photonic chip, and detector
- `Chip` implementations (like `SuperKN`) define the optical architecture
- Numerical modules provide low-level algorithms for signal processing

## Model Scope

PHISE relies on two core modelling choices:

- **Scalar approximation**: the light collected by each telescope is represented by a single complex amplitude per source and per wavelength. The spatial profile of the wavefront and polarisation are not modelled.
- **Matrix description of photonic components**: every component (MMI, Y-splitter, phase shifter, crosstalk) is described by a complex transfer matrix. Propagation through the chip is the product of these matrices applied to the vector of input fields, $\psi_\text{out} = M_n \cdots M_2 M_1\, \psi_\text{in}$. Output intensities are the squared moduli $|\psi_\text{out}|^2$, summed incoherently over the sources.

| Domain | ✅ Supported | 🚧 Planned | ⛔ Out of scope |
|---|---|---|---|
| **Scene** | Star + any number of point-like companions with contrast, separation and position angle | | Resolved sources (stellar diameter, exozodiacal disk) |
| **Spectrum** | Monochromatic or band-integrated observation (flat spectrum, sub-band sampling) | | Source-specific spectra: PHISE focuses on detection at a given wavelength, not on spectral characterisation |
| **Array geometry** | 2D telescope layout projected with hour angle, declination and latitude; ideal delay lines | 3D layout, realistic delay lines | |
| **Time** | Series of exposures over an hour-angle span | Earth rotation during an exposure | |
| **Atmosphere** | Random Gaussian piston (RMS `Γ`) per telescope and per exposure | Piston with temporal coherence | |
| **Light injection** | Single-mode field per telescope with global throughput `η` | Injection fluctuations (tip-tilt, Strehl) | Wavefront modelling: fields are scalar complex amplitudes |
| **Photonic chip** | Complex transfer matrices, phase shifters, manufacturing errors, measured crosstalk | | |
| **Detector** | Photon, dark and read-out noise, saturation, gain, Gaussian spot per output | | Non-linearity, pixel response non-uniformity, ADC effects |

## Modelling Details

This section details the physical model behind the table above, so that results can be interpreted correctly.

### Scene
- The star and its companions are **unresolved point sources**. Stellar leakage due to the finite stellar diameter is not modelled.
- A companion's photon flux is `c × star flux` at every wavelength: the contrast is **achromatic** (same spectrum as the star).
- There is no exozodiacal light, thermal background or sky background.

### Photon flux
- The spectral flux `f` is assumed **flat over the band**. The photon rate per telescope is
  $\dot N_i = \eta\, f\, a_i\, \Delta\lambda\, \lambda / (h c)$, with a single global throughput `η` shared by all telescopes and achromatic.
- **Monochromatic mode** (`monochromatic=True`): every photon of the band is propagated at the central wavelength `λ`. Bandwidth smearing is ignored.
- **Polychromatic mode**: the band is sampled at `spectral_samples` wavelengths. Each sample is simulated as a monochromatic 1 nm sub-band, then the outputs are integrated with the trapezoidal rule.

### Array geometry and injection
- Telescope positions are **2D** (no altitude differences). They are projected onto the plane perpendicular to the line of sight using hour angle, declination and latitude.
- Delay lines are ideal: the on-axis star reaches all inputs in phase, apart from the piston error. Off-axis sources receive the geometric phase $2\pi\, p_i \sin\rho / \lambda$, where $p_i$ is the projected position along the source direction.
- Each telescope delivers one **scalar single-mode field** of amplitude $\sqrt{\dot N_i}$. Injection-efficiency fluctuations, tip-tilt, scintillation and polarisation effects are not modelled.

### Atmosphere and time sampling
- The residual piston is a **Gaussian OPD** of RMS `Γ`, drawn independently for each telescope and each exposure. It is frozen during the exposure, uncorrelated between exposures, and achromatic in length (no differential dispersion).
- An observation of duration `Δh` is split into `Δh / e` **instantaneous snapshots**, where `e` is the exposure time. Earth rotation during an exposure is neglected.

### Photonic chips
Chips are modelled as **complex transfer matrices** acting on the input fields:
- **`SuperKN`**: ideal lossless 2×2 nullers (Hadamard) and quadrature cross-recombiners, plus ideal 50/50 Y-splitters. Chromaticity is approximated by scaling the MMI phases by `λ0/λ`. Split ratios are achromatic.
- **`N4x4_T8`, `N4x4_D8`**: an ideal combiner matrix placed between fixed input/output **crosstalk matrices measured on a real chip**. These matrices are neither unitary nor wavelength-dependent. The shifter phases are evaluated at `λ0`, so these models have **no chromatic response**. `N4x4_T8_Ideal` replaces the crosstalk matrices with the identity.
- Phase shifters and manufacturing errors (`φ`, `σ`) are pure, static OPDs. Material dispersion, thermal crosstalk and drift are not modelled.

### Detector
- Each chip output falls on its **own detector area** as a centred 2D Gaussian spot of width `spot_size`.
- The noise chain is: quantum efficiency → Poisson photon noise (Gaussian approximation above 2×10⁹ photons) → Poisson dark current → Gaussian read-out noise → hard full-well clipping → gain and rounding → subtraction of a master dark averaged over 1000 frames.
- Non-linearity, pixel response non-uniformity, ADC quantisation, excess noise factor and cosmic rays are not modelled.

## Development Notes

- **Authorship**: the code was written almost entirely by hand. This documentation was generated with AI tools.
- **Unicode identifiers**: variables, including many API parameters, use Greek letters and symbols (`λ`, `Δλ`, `φ`, `σ`, `ρ`, `θ`, `Γ`, `η`…) so that the code reads like the physical expressions it implements. These characters can be typed with configurable keyboard shortcuts, available natively on Linux and macOS and through [WinCompose](https://github.com/samhocevar/wincompose) on Windows.
- **Review**: the project does not yet have a strict contribution process with review by an independent contributor. Both the code and this documentation may therefore contain errors. Reports are welcome on the [GitHub repository](https://github.com/VForiel/PHISE/issues).

## Quick Links

```{toctree}
:maxdepth: 2

quickstart.md
api_references.md
contribute.md
```

## Installation

See the [Quickstart Guide](quickstart.md) for installation instructions.

## Citation

If you use PHISE in scientific work, please cite the repository and related publications. Lead author: Vincent Foriel.

## Support

Issues and contributions are welcome on the [GitHub repository](https://github.com/VForiel/PHISE). Please note the development status warnings above before reporting issues.