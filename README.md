# Tunable Kernel-Nulling for Direct Exoplanet Detection

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## 📜 Abstract

This project focuses on the development and optimization of a tunable Kernel-Nulling interferometer for direct detection of exoplanets. The work combines numerical simulations, calibration algorithms, and statistical analysis techniques to achieve high-contrast detection capabilities using a four-telescope architecture with integrated photonic components.

## 🎯 Objectives

- **Direct exoplanet detection** with contrasts beyond 10⁻⁸
- **Phase aberration correction** using active photonic components with 14 electro-optic phase shifters
- **Performance optimization** through advanced calibration algorithms
- **Statistical analysis** of kernel-null depth distributions

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- PDM (Python Dependency Manager)

### Key Dependencies

- `numpy` - Numerical computations
- `astropy` - Astronomical units and calculations
- `matplotlib` - Plotting and visualization
- `scipy` - Scientific computing
- `numba` - High-performance numerical functions
- `ipywidgets` - Interactive widgets for Jupyter notebooks

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/Tunable-Kernel-Nulling.git
cd Tunable-Kernel-Nulling
```

2. Install project dependencies (using [PDM](https://pdm-project.org/)):
```bash
pdm install
```

Thes open the main simulation notebook "`numerical_simulation.ipynb`" and select the appropriate kernel for your environment.

## 🔬 Scientific Approach

### Architecture

The system employs a four-telescope Kernel-Nulling architecture using integrated optical components:

- **4 Telescopes**: Collecting light from target star and potential companions
- **14 Phase Shifters**: Electro-optic elements for phase correction
- **MMI Components**: Multi-mode interferometers for signal processing
- **7 Outputs**: 1 bright output + 6 dark outputs → 3 kernel outputs

### Key Features

1. **Calibration Algorithms**:
   - Genetic algorithm approach
   - Input obstruction method
   - Performance comparison and optimization

2. **Statistical Analysis**:
   - ROC curve analysis
   - P-value computation
   - Multiple test statistics (mean, median, Kolmogorov-Smirnov, etc.)

3. **Simulation Scenarios**:
   - **VLTI**: Ground-based, 8m telescopes, 130m baseline, λ=1.55μm
   - **LIFE**: Space-based, 2m telescopes, 600m baseline, λ=4μm

### Applications

- Direct imaging of exoplanets
- High-contrast astronomy
- Interferometric nulling techniques
- Statistical detection methods

## 📊 Key Results

- Achievable contrasts: 10⁻⁵ to 10⁻⁶ (limited by phase perturbations)
- Robust performance against first-order phase aberrations
- Statistical tests demonstrate reliable planet detection capabilities
- Successful calibration algorithms for component optimization

## 📚 Publications

This work has contributed to several scientific publications:

1. **SPIE Proceedings** - "Tunable Kernel-Nulling interferometry for direct exoplanet detection"
2. **A&A Paper (in preparation)** - "Tunable Kernel-Nulling for direct detection of exoplanets: 1. Calibration and performance"
3. **Statistical Analysis Paper (in preparation)** - "Statistical data analysis techniques for kernel-nulling interferometry"

## 👥 Contributors

- **Vincent Foriel** - PhD Student, Primary Developer
- **David Mary** - Supervisor, Statistical Analysis
- **Frantz Martinache** - Supervisor, Interferometry Expert
- **Nick Cvetojevic** - Photonics Specialist
- **Romain Laugier** - Kernel-Nulling Theory
- **Marc-Antoine Martinod** - Technical Support
- **Sylvie Robbe-Dubois** - Project Coordination
- **Roxanne Ligi** - Scientific Advisor

## 🏢 Affiliations

- **Université Côte d'Azur, Observatoire de la Côte d'Azur Nice**
- **CNRS, Laboratoire Lagrange, Nice, France**
- **KU Leuven University, Leuven, Belgium**

## 📞 Contact

For questions or collaborations:
- **Vincent Foriel**: vincent.foriel@oca.eu
- **Frantz Martinache**: frantz.martinache@oca.eu
- **David Mary**: david.mary@oca.eu
