# PHISE
PHotonic Interferometric Simulation for Exoplanets

PHISE is a Python package for simulating nulling interferometers based on photonic beam combiners for direct exoplanet detection. It models the full chain, from the observed scene to the detector, and provides tools to calibrate the photonic chip and to quantify detection performance.

PHISE was originally designed for the specific needs of a PhD thesis on photonic kernel nulling ([VForiel/PhD-Theory](https://github.com/VForiel/PhD-Theory)). Its scope reflects these needs. It is still under active development, and the API may change without notice.

Documentation: https://phise.readthedocs.io/

## What PHISE can do

- **End-to-end simulation** in a single `Context` object: target and companions → telescope array → photonic chip → camera. Changing a parameter (hour angle, declination, wavelength…) automatically updates the dependent quantities.
- **Earth rotation**: telescope positions are projected onto the sky plane over the whole observation. Ready-made VLTI (4 UTs) and LIFE configurations are provided.
- **Photonic kernel nullers**:
  - `SuperKN`: 14 phase shifters, 7 outputs, 3 kernels.
  - `N4x4_T8` / `N4x4_D8`: 4×4 MMI-based chips using crosstalk matrices measured on a real component, plus an ideal variant.
- **Manufacturing defects and active control**: intrinsic OPD errors, tunable phase shifters, input attenuation and OPD.
- **Calibration algorithms**: trial-and-error, obstruction-based least-squares fitting, and Hooke & Jeeves direct search.
- **Detector model**: photon noise, dark current, read-out noise, full-well saturation, gain, dark calibration, and pixel images of each output. A helper computes a reference exposure time.
- **Instrument response visualisation**: numerical and analytical transmission maps and their gradients, phasor diagrams (Laugiergrams), projected baselines.
- **Detection statistics**: Monte-Carlo generation of H0/H1 observations, more than ten test statistics (Kolmogorov–Smirnov, Cramér–von Mises, Anderson–Darling, Wasserstein…), ROC curves, Neyman–Pearson analysis, and detection-capability maps in contrast vs separation.
- **Synthetic datasets** for machine-learning-based calibration.
- **Fast and safe to use**: `astropy` units at every API boundary, `numba` JIT-compiled numerical kernels.

## Model scope

PHISE relies on two core modelling choices:

- **Scalar approximation**: the light collected by each telescope is represented by a single complex amplitude per source and per wavelength. The spatial profile of the wavefront and polarisation are not modelled.
- **Matrix description of photonic components**: every component (MMI, Y-splitter, phase shifter, crosstalk) is described by a complex transfer matrix. Propagation through the chip is the product of these matrices applied to the vector of input fields. Output intensities are the squared moduli of the output fields, summed incoherently over the sources.

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

Modelling details (formulas and approximations) are given in the [documentation](https://phise.readthedocs.io/).

## Development notes

- **Authorship**: the code was written almost entirely by hand. The documentation was generated with AI tools.
- **Unicode identifiers**: variables, including many API parameters, use Greek letters and symbols (`λ`, `Δλ`, `φ`, `σ`, `ρ`, `θ`, `Γ`, `η`…) so that the code reads like the physical expressions it implements. These characters can be typed with configurable keyboard shortcuts, available natively on Linux and macOS and through [WinCompose](https://github.com/samhocevar/wincompose) on Windows.
- **Review**: the project does not yet have a strict contribution process with review by an independent contributor. Both the code and the documentation may therefore contain errors. Reports are welcome in the [issues](https://github.com/VForiel/PHISE/issues).
