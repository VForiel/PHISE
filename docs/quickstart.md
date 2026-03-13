# 🚀 Quick Start

This guide will help you get started with PHISE, from installation to running your first simulation.

## Installation

PHISE requires Python 3.11 or higher. There are two recommended installation methods:

### Method 1: Conda Environment (Recommended)

The easiest way to get started is using the provided conda environment:

```bash
conda env create -f environment.yml
conda activate phise
```

This will install all required dependencies including numpy, astropy, scipy, matplotlib, numba, and ipywidgets.

### Method 2: Development Install with pip

For development work or if you prefer pip:

```bash
pip install -e .
```

This installs PHISE in editable mode, allowing you to modify the source code and see changes immediately.

## First Simulation

Here's a minimal example to create and run a basic interferometric observation:

```python
import phise
from astropy import units as u
import numpy as np

# Create telescopes
tel1 = phise.Telescope(a=50*u.m**2, r=[0, 0]*u.m, name="Tel 1")
tel2 = phise.Telescope(a=50*u.m**2, r=[10, 0]*u.m, name="Tel 2")
tel3 = phise.Telescope(a=50*u.m**2, r=[5, 8.66]*u.m, name="Tel 3")
tel4 = phise.Telescope(a=50*u.m**2, r=[5, -8.66]*u.m, name="Tel 4")

# Create a photonic chip (4-telescope kernel nuller)
chip = phise.SuperKN(
    φ=np.zeros(14)*u.nm,  # Phase shifters
    σ=np.zeros(14)*u.nm,  # OPD errors
    λ0=1550*u.nm,         # Reference wavelength
    name="SuperKN"
)

# Create a camera
camera = phise.Camera(e=1*u.s, ideal=True, name="Ideal Camera")

# Assemble the interferometer
interferometer = phise.Interferometer(
    l=45*u.deg,              # Latitude
    λ=1550*u.nm,             # Central wavelength
    Δλ=100*u.nm,             # Bandwidth
    fov=100*u.mas,           # Field of view
    η=0.8,                   # Optical efficiency
    telescopes=[tel1, tel2, tel3, tel4],
    chip=chip,
    camera=camera,
    name="4T Kernel Nuller"
)

# Create a target star
target = phise.Target(
    δ=45*u.deg,              # Declination
    S=1e10*u.photon/u.s,     # Spectral flux
    m=5,                      # Magnitude
    name="Target Star"
)

# Optional: Add a companion (exoplanet)
companion = phise.Companion(
    c=1e-4,                  # Contrast relative to star
    ρ=50*u.mas,              # Angular separation
    θ=0*u.rad,               # Position angle
    name="Exoplanet"
)
target.companions.append(companion)

# Create observation context
ctx = phise.Context(
    interferometer=interferometer,
    target=target,
    h=0*u.hourangle,         # Hour angle
    Δh=1*u.h,                # Observation duration
    Γ=10*u.nm,               # Cophasing error (RMS)
    monochromatic=False,
    name="Example Observation"
)

print(ctx)
```

## Basic Visualization

PHISE provides several visualization methods:

### Transmission Map

Visualize the null depth across the field of view:

```python
import matplotlib.pyplot as plt

# Generate transmission map
ctx.plot_transmission_map(
    resolution=100,          # Number of pixels per axis
    output='kernel',         # Which output to display
    vmin=-10, vmax=0        # Display range in magnitude
)
plt.show()
```

### Projected Baselines

Show how telescope positions project onto the sky:

```python
ctx.plot_projected_baselines()
plt.show()
```

## Working with Physical Units

PHISE uses `astropy.units` for all physical quantities. This ensures dimensional correctness:

```python
from astropy import units as u

# Correct: use Quantity objects
wavelength = 1550 * u.nm
baseline = 10 * u.m
angle = 50 * u.mas

# Unit conversion is automatic
print(wavelength.to(u.um))  # 1.55 μm
print(angle.to(u.arcsec))   # 0.05 arcsec

# Setting parameters
ctx.h = 2 * u.hourangle  # Change hour angle
ctx.Γ = 20 * u.nm        # Update cophasing error

# Context auto-updates dependent quantities
print(ctx.p)  # Projected baselines automatically recomputed
```

## Context Auto-Update Pattern

One of PHISE's key features is automatic propagation of parameter changes:

```python
# Changing hour angle automatically updates projected positions
ctx.h = 3 * u.hourangle
# ctx.p is now recomputed

# Changing telescope area automatically updates photon flux
tel1.a = 100 * u.m**2
# ctx.pf is now recomputed

# Changing wavelength triggers flux recalculation
interferometer.λ = 1600 * u.nm
# ctx.pf is now recomputed
```

This design eliminates manual update calls and reduces errors.

## Accessing Simulation Results

The `Context` object provides access to computed quantities:

```python
# Projected telescope positions (accounting for Earth rotation)
print(ctx.p)  # Shape: (N_tel, N_tel, 2)

# Total photon flux from target
print(ctx.pf)  # photons/s

# Get chip output matrices
M_bright, M_null = ctx.interferometer.chip.get_matrices(
    λ=ctx.interferometer.λ,
    p=ctx.p
)

print(f"Bright output matrix shape: {M_bright.shape}")
print(f"Null output matrix shape: {M_null.shape}")
```

## Running Tests

To ensure your installation is working correctly:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_context.py

# Run with coverage
pytest --cov=phise tests/
```

## Next Steps

- **[API Reference](api_references.md)**: Detailed documentation of all classes and modules
- **[Contribute](contribute.md)**: Guidelines for contributing to PHISE
- **Analysis Scripts**: Explore `src/analysis/` for advanced examples (transmission maps, calibration, noise sensitivity, etc.)
- **Notebooks**: Check out the Jupyter notebooks in the repository root for interactive demonstrations

## Common Pitfalls

### Units are Required

```python
# ❌ Wrong - will raise TypeError
tel = phise.Telescope(a=50, r=[0, 0])

# ✅ Correct - always use astropy.units.Quantity
tel = phise.Telescope(a=50*u.m**2, r=[0, 0]*u.m)
```

### Array Shapes Matter

```python
# ❌ Wrong - position must be 2D
tel = phise.Telescope(a=50*u.m**2, r=[0, 0, 0]*u.m)

# ✅ Correct - [x, y] position
tel = phise.Telescope(a=50*u.m**2, r=[0, 0]*u.m)
```

### Import from Installed Package

```python
# ✅ Correct - import from installed phise package
import phise
from phise.classes import Context, Interferometer
from phise.modules import signals, coordinates

# ❌ Wrong - don't import from src/ directly
# from src.phise.classes import Context  # This won't work
```

## Getting Help

If you encounter issues:

1. Check that you're using Python 3.11+
2. Verify all dependencies are installed: `pip list | grep -E "numpy|astropy|scipy|numba"`
3. Review the error message carefully - type and unit errors are common
4. Open an issue on [GitHub](https://github.com/VForiel/Tunable-Kernel-Nulling) with:
   - Minimal code to reproduce the problem
   - Full error traceback
   - Python version and OS

Remember: PHISE is under active development. See the [main documentation page](index.md) for important warnings about stability.