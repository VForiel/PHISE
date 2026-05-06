# PHISE AI Coding Agent Instructions

## Project Overview
PHISE (PHotonic Interferometry Simulation for Exoplanets) is a Python package for simulation and analysis of interferometric instruments using layered photonic chips for exoplanet detection. It provides high-level classes to model the complete instrument chain from telescopes to detector, numerical modules for signal processing and analysis, and tools for visualizing instrument responses. Built for scientific computing with performance optimization via numba JIT compilation.

## Architecture Fundamentals

### Component-Based Simulation Model
The core abstraction is a **Context** that orchestrates the interaction between instrument components and observation parameters:
- `Context` combines an `Interferometer`, a `Target`, and observational parameters (hour angle, integration time, cophasing error)
- The interferometer chain: `Telescope` array → `Chip` (photonic recombiner) → `Camera` (detector)
- Signal flow: Target scene generates photon flux → Telescope array collects light → Photonic chip combines beams → Camera detects nulled/constructive outputs

**Key files:** `src/phise/classes/context.py`, `src/phise/classes/interferometer.py`

### Component Categories
1. **Observation Context** (`classes/context.py`): Top-level orchestrator holding instrument, target, and acquisition settings
2. **Instrument Components**:
   - **Telescope** (`classes/telescope.py`): Individual apertures with positions and diameters
   - **Interferometer** (`classes/interferometer.py`): Telescope array configuration with baselines
   - **Chip** (`classes/chip.py`, `classes/chip_new.py`): Photonic beam combiner (MMI-based kernel nullers)
   - **Camera** (`classes/camera.py`): Detector with quantum efficiency, dark current, readout noise
3. **Scene Components**:
   - **Target** (`classes/target.py`): Star properties (coordinates, spectrum, magnitude)
   - **Companion** (`classes/companion.py`): Planet/companion with position and contrast
4. **Numerical Modules**:
   - **coordinates** (`modules/coordinates.py`): Projected baselines, hour angle transformations
   - **signals** (`modules/signals.py`): Photon flux calculations, signal propagation
   - **phase** (`modules/phase.py`): Optical path differences, phase screens
   - **mmi** (`modules/mmi.py`): Multi-Mode Interferometer transfer matrices
   - **test_statistics** (`modules/test_statistics.py`): Detection statistics, null depth analysis
5. **Architectures** (`classes/archs/`): Pre-configured photonic chip designs (e.g., SuperKN for 4-telescope kernel nulling)

## Critical Conventions

### Units: astropy.Quantity at API Boundaries
**MANDATORY PATTERN**: All user-facing parameters use `astropy.units.Quantity`. Internal storage converts to native Python types for performance.

```python
# ✅ Correct - API accepts Quantity
def __init__(self, diameter: u.Quantity = 8*u.m, position: u.Quantity = [0, 0]*u.m):
    self.diameter = diameter  # Store as Quantity OR...
    self._diameter_m = diameter.to(u.m).value  # Convert to float internally

# ❌ Wrong - never accept raw floats for physical quantities
def __init__(self, diameter: float, position: list):
```

**When adding new components**: Always use `u.Quantity` for distances (u.m, u.AU, u.arcsec, u.mas), wavelengths (u.nm, u.um), angles (u.deg, u.rad), times (u.s, u.h), fluxes (photon counts, magnitudes).

### Performance Architecture
**High-level interface with low-level performance**: The API is designed with two layers:

1. **User-facing API**: Accepts `astropy.Quantity` objects for all physical quantities (distances, wavelengths, angles, times, etc.)
2. **Internal storage**: Converts to native Python types (float, numpy arrays) with fixed or dimensionless units for performance

**Conversion pattern**:
```python
# At API boundary (constructor, setters)
def __init__(self, baseline: u.Quantity = 10*u.m):
    self.baseline = baseline  # High-level API

@property
def baseline(self):
    return self._baseline_internal * u.m  # Return as Quantity

@baseline.setter
def baseline(self, value: u.Quantity):
    self._baseline_internal = value.to(u.m).value  # Convert on set
```

**Performance optimization priorities**:
- Use **numba JIT compilation** for computationally intensive loops (signal propagation, coordinate transformations)
- Prefer numpy vectorized operations over Python loops
- Cache expensive computations when possible (projected baselines, transfer matrices)
- Use `@nb.njit` decorator for performance-critical numerical functions

The goal is to provide a user-friendly, physically intuitive API while achieving the best possible computational performance for scientific simulations.

### Context Auto-Update Pattern
The `Context` class automatically propagates parameter changes to recompute derived quantities:
- Changing `h` (hour angle) triggers `_update_p()` to recompute projected telescope positions
- Changing target or interferometer triggers `_update_pf()` to recompute photon flux
- Use property setters with `if self._initialized:` guard to trigger updates only after full initialization

**Example from `context.py`**:
```python
@property
def h(self) -> u.Quantity:
    return self._h * self._h_unit

@h.setter
def h(self, value: u.Quantity):
    self._h = value.to(u.hourangle).value
    self._h_unit = u.hourangle
    if self._initialized:
        self._update_p()  # Auto-recompute projected positions
```

### Tests in Separate Directory
**PROJECT SPECIFIC**: Unit tests are in `/tests` directory, not embedded in modules:
```
tests/
  test_classes_basic.py      # Test Context, Interferometer, Target, etc.
  test_context.py            # Test Context auto-update behavior
  test_modules_coordinates.py # Test coordinate transformations
  test_modules_phase_mmi.py  # Test MMI and phase calculations
  test_modules_signals_stats.py # Test signal propagation and statistics
```

Run tests with pytest from project root:
```powershell
pytest tests/
```

## Build & Development Workflow

### Installation & Testing
```powershell
# Install in development mode (recommended)
pip install -e .

# OR: Create conda environment
conda env create -f environment.yml
conda activate phise

# Run all tests
pytest

# Run specific test file
pytest tests/test_context.py

# Run with coverage
pytest --cov=phise tests/
```

### Package Structure
- **Source**: `src/phise/` (installed package)
- **Classes**: `src/phise/classes/` (high-level components)
- **Modules**: `src/phise/modules/` (numerical algorithms)
- **Tests**: `tests/` (unit tests with pytest)
- **Analysis scripts**: `src/analysis/` (demonstration scripts and notebooks)
- **Documentation**: `docs/sphinx/` (Sphinx docs, auto-generated from docstrings)

**Important**: Tests use `import phise` to test the installed package.

## Common Patterns

### Adding New Instrument Components
1. Create class in `src/phise/classes/`
2. Accept `u.Quantity` parameters in `__init__`
3. Convert to native types and store in private attributes (e.g., `_diameter_m`)
4. Implement properties with getters/setters for unit validation
5. Add `__str__` and `__repr__` for readable output
6. Export in `src/phise/classes/__init__.py`

### Adding New Numerical Modules
1. Create module file in `src/phise/modules/`
2. Use `@nb.njit` for performance-critical functions
3. Accept numpy arrays (not Quantities) for numba compatibility
4. Document expected units in docstrings
5. Export in `src/phise/modules/__init__.py`

### Working with Photonic Architectures
Photonic chip designs are in `src/phise/classes/archs/`:
- **SuperKN** (`archs/superkn.py`): 4-telescope kernel nuller with MMI combiners
- Each architecture defines the optical layout, combiner matrices, and output mappings
- Use pre-configured architectures via `from phise.classes.archs import SuperKN`

### Visualization Methods
Many components implement `.plot()` methods:
- `Context.plot_transmission_map()`: Shows null depth across field of view
- `Context.plot_projected_baselines()`: Displays telescope positions and baselines
- `Target.plot()`: Shows companions in angular coordinates
- Use matplotlib, save figures or return Axes for customization

## Key File References

- **Entry point**: `src/phise/__init__.py` exports all classes and modules
- **Main context**: `src/phise/classes/context.py` orchestrates simulations
- **Photonic chip**: `src/phise/classes/chip.py` models beam combiners
- **MMI matrices**: `src/phise/modules/mmi.py` computes transfer functions
- **Test statistics**: `src/phise/modules/test_statistics.py` for detection analysis

## Documentation & CI

- **Docs**: Sphinx with auto-generated API reference from docstrings
- **Build docs**: `cd docs/sphinx && make html` (Windows: `make.bat html`)
- **Docstrings**: All public APIs must have numpy-style docstrings for Sphinx autodoc
- **ReadTheDocs**: Documentation hosted at https://phise.readthedocs.io/

## Code Quality Requirements (CRITICAL)

### Educational Philosophy
**PHISE is a scientific research project with educational clarity** - the code must be rigorous and scientifically accurate, but explained so that any scientist can understand it, even if they are not experts in photonic interferometry. This means:

- **Explain the physics**: Every interferometric/photonic concept must be explained in docstrings and comments
- **Provide context**: Don't assume users know why a particular algorithm or formula is used
- **Use clear variable names**: Prefer descriptive names like `baseline_projected` over `bp`, `null_depth` over `nd` (units are already in `astropy.Quantity` objects)
- **Add educational comments**: Explain the "why" not just the "what"
- **Include references**: When implementing published algorithms, cite the paper/textbook (especially for kernel nulling, MMI matrices, test statistics)
- **Validate physically**: Tests should verify that results make physical sense, not just that code runs

### Every Function Must Have
1. **English docstring** (numpy-style for Sphinx autodoc)
   - Explain the physical concept being modeled (e.g., "Compute projected baselines accounting for Earth rotation")
   - Define all parameters with units and physical meaning
   - Include mathematical formulas when relevant (using LaTeX in docstrings)
   - Example: `"""Calculate null depth in magnitudes. Returns -2.5 * log10(flux_ratio)."""`
2. **Clear English comments** explaining non-obvious logic
   - Explain physical reasoning behind algorithmic choices
   - Clarify coordinate systems, sign conventions, normalizations
   - Example: `# Project baselines onto sky plane perpendicular to target direction`
3. **Unit test** in `/tests` directory validating both correctness AND physical coherence
   - Test edge cases and boundary conditions
   - Verify conservation laws (photon counts, flux conservation through chip)
   - Check dimensional analysis (units consistency)
   - Example: Verify that null outputs have lower flux than bright outputs
4. **Documentation generation** - ensure Sphinx autodoc can process it

### Testing Philosophy
Tests must verify:
- ✅ Code executes without errors
- ✅ **Physical results are coherent** (units, magnitudes, conservation laws)
- ✅ Edge cases and boundary conditions
- ✅ Numerical stability (no NaNs, infs, or unexpected singularities)

Example test structure:
```python
def test_projected_baselines():
    """Test that projected baselines change correctly with hour angle."""
    # Create simple 2-telescope interferometer
    tel1 = Telescope(position=[0, 0]*u.m, diameter=1*u.m)
    tel2 = Telescope(position=[10, 0]*u.m, diameter=1*u.m)
    interf = Interferometer([tel1, tel2])
    
    # At h=0, baseline should be purely East-West
    ctx = Context(interf, target, h=0*u.hourangle, Δh=0*u.h, Γ=0*u.nm)
    assert np.allclose(ctx.p[0, 1], 0)  # No North-South component
    
    # Check physical coherence
    assert np.all(np.isfinite(ctx.p))  # No NaNs or infs
```

### Documentation Maintenance
- **Auto-generated docs**: All public APIs appear in `docs/sphinx/` via Sphinx autodoc
- **Static docs**: Add markdown files in `docs/sphinx/` for conceptual explanations (quickstart, API overview)
- **Keep synchronized**: Update docstrings when changing function signatures
- **Build locally**: Test doc build with `cd docs/sphinx && make html` before committing

## Agent Modification Logs

**Every modification session must create a log file** in `.github/agent-logs/` with format:
```
.github/agent-logs/YYYY.MM.DD-NN_<topic>.md
```

Where:
- `YYYY.MM.DD`: Date (e.g., 2025.11.26)
- `NN`: Sequential number if multiple sessions same day (01, 02, ...)
- `<topic>`: 1-2 word summary (e.g., `context-update`, `mmi-calibration`, `test-statistics`)

**Log content should include**:
- Summary of changes made
- Files modified
- New functions/classes added
- Tests added/updated
- Any breaking changes or migration notes
- Physics validation performed

Example: `.github/agent-logs/2025.11.26-01_copilot-instructions.md`

## Notebook Validation Protocol (CRITICAL)

**When modifying code that impacts notebook cells**, you MUST validate the changes by executing the affected code:

### Validation Methods (in order of preference)

1. **Direct execution via `run_notebook_cell` tool** (preferred when available)
   - Execute the modified notebook cells directly
   - Verify outputs match expectations
   - Check that no errors are raised

2. **Standalone Python script** (when direct execution not available)
   - Extract the relevant cell code into a temporary Python script
   - Add necessary imports and setup code
   - Execute via `run_in_terminal` to validate correctness

3. **Data validation without plots** (most reliable)
   - **PREFERRED**: Skip matplotlib visualization, validate data directly with `print()` statements
   - Check array shapes, value ranges, statistical properties (mean, std, min, max)
   - Verify physical coherence (units, magnitudes, conservation laws)
   - Example:
     ```python
     print(f"Null outputs shape: {nulls.shape}")
     print(f"Null depth range: [{nulls.min():.2e}, {nulls.max():.2e}]")
     print(f"Mean photon count: {photons.mean():.2e}")
     print(f"Flux conservation: {flux_in.sum():.2e} vs {flux_out.sum():.2e}")
     ```

4. **File-based visualization** (only if visual inspection required)
   - If plots are necessary, save to files with `plt.savefig('test_output.png')`
   - Use descriptive filenames indicating what is being tested
   - Analyze saved images to verify correctness
   - Clean up test output files after validation

### What to Validate
- ✅ Code executes without errors
- ✅ Output shapes and types are correct
- ✅ Numerical values are in expected ranges
- ✅ Physical quantities have correct units and magnitudes
- ✅ Visualizations (if needed) display expected features
- ✅ No NaNs, infs, or unexpected numerical issues

### Example Validation Pattern
```python
# GOOD: Validate data directly
import phise
from astropy import units as u
import numpy as np

# Create simple context
tel = phise.Telescope(position=[0, 0]*u.m, diameter=1*u.m)
interf = phise.Interferometer([tel])
target = phise.Target(ra=0*u.deg, dec=0*u.deg, magnitude=5)
ctx = phise.Context(interf, target, h=0*u.hourangle, Δh=1*u.h, Γ=10*u.nm)

# Validate without plotting
print(f"Projected positions shape: {ctx.p.shape}")  # Expected: (N_tel, N_tel, 2)
print(f"Photon flux: {ctx.pf:.2e}")  # Expected: positive finite value
print(f"All finite: {np.all(np.isfinite(ctx.p))}")  # Expected: True
assert ctx.p.shape[0] == len(interf.telescopes), "Wrong number of telescopes"
assert ctx.pf > 0, "Photon flux must be positive"
print("✓ Validation passed")
```

**Always validate changes before finalizing** - this catches bugs early and ensures physical coherence.

## What NOT to Do
- ❌ Don't use raw floats for physical quantities in APIs
- ❌ Don't hardcode array sizes - derive from number of telescopes/outputs
- ❌ Don't ignore units when reading existing code - maintain consistency
- ❌ Don't commit code without docstrings and unit tests in `/tests`
- ❌ Don't skip the agent modification log
- ❌ Don't modify notebook cells without validating the changes execute correctly
- ❌ Don't break the Context auto-update pattern - use property setters correctly
- ❌ Don't mix numba-compiled and non-numba code without understanding compatibility
- ❌ Don't assume zero hour angle - test across different observation times
