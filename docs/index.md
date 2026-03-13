# PHISE Documentation

**PHISE** (PHotonic Interferometry Simulation for Exoplanets) is a Python package for simulation and analysis of interferometric instruments using layered photonic chips for exoplanet detection.

> ⚠️ **Development Status Warning**
> 
> PHISE is currently under active development as part of a PhD research project by Vincent Foriel. The package is **highly unstable** and subject to breaking changes at any time without notice. The API, functionalities, and internal architecture may change significantly as research progresses.
> 
> **Important notes:**
> - This library is tightly coupled to ongoing thesis work and is not yet production-ready
> - The `src/analysis/` module contains thesis-specific analysis scripts and will be moved to a separate repository when PHISE is published
> - Documentation may not always reflect the current state of the code
> - Use at your own risk for research purposes only
> 
> If you're visiting this documentation now, please be aware that you're looking at a work-in-progress research tool.

## Overview

PHISE provides a comprehensive framework for modeling interferometric instruments with photonic beam combiners. The package includes:

- **High-level classes** to model the complete instrument chain: telescopes, interferometer arrays, photonic chips (kernel nullers), detectors, and target scenes
- **Numerical modules** for signal propagation, coordinate transformations, MMI transfer matrices, phase calculations, and detection statistics
- **Pre-configured architectures** like SuperKN for 4-telescope kernel nulling
- **Visualization tools** for transmission maps, projected baselines, null depth analysis, and more
- **Scientific computing performance** via numba JIT compilation and numpy vectorization

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

## Features

### Instrument Modeling
- Multi-telescope interferometer configurations
- Photonic kernel nulling architectures (MMI-based beam combiners)
- Realistic detector models with quantum efficiency, dark current, and readout noise
- Wavelength-dependent transmission and chromatic effects

### Scene Modeling
- Target stars with spectral flux and magnitude
- Multiple companions with configurable contrast and angular separation
- Field-of-view coverage for transmission map analysis

### Numerical Simulation
- Projected baseline calculations accounting for Earth rotation
- Photon flux propagation through optical chain
- Phase and amplitude modulation by photonic circuits
- Statistical detection analysis (null depth, SNR, test statistics)

### Visualization
- Transmission maps showing null depth across field of view
- Projected baseline configurations at different hour angles
- Output photon distributions
- Temporal response of the instrument

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