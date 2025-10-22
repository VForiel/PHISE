---
title: Tunable Kernel Nulling
---

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

- Anaconda or Miniconda (or another Conda distribution)

### Key Dependencies

- `numpy` - Numerical computations
- `astropy` - Astronomical units and calculations
- `matplotlib` - Plotting and visualization
- `scipy` - Scientific computing
- `numba` - High-performance numerical functions
- `ipywidgets` - Interactive widgets for Jupyter notebooks

### Installation (use Conda)

1. Clone the repository:
```powershell
git clone https://github.com/your-username/Tunable-Kernel-Nulling.git
cd Tunable-Kernel-Nulling
```

2. Create the Conda environment from the provided `environment.yml`:
```powershell
conda env create -f environment.yml
```

3. Activate the environment:
```powershell
conda activate kn
```

4. Install the local project package editable (optional but useful for development):
```powershell
pip install -e .
```

## 🔬 Scientific Approach

### Architecture

The system employs a four-telescope Kernel-Nulling architecture using integrated optical components:

- **4 Telescopes**: Collecting light from target star and potential companions
- **14 Phase Shifters**: Electro-optic elements for phase correction
- **MMI Components**: Multi-mode interferometers for signal processing
- **7 Outputs**: 1 bright output + 6 dark outputs → 3 kernel outputs

## 📊 Key Results

- Achievable contrasts: 10⁻⁵ to 10⁻⁶ (limited by phase perturbations)
- Robust performance against first-order phase aberrations
- Statistical tests demonstrate reliable planet detection capabilities

```{toctree}
:maxdepth: 2
:caption: Contents

api
modules
```
