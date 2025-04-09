# HoloPy

A Python Library for Holographic Cosmology and Holographic Gravity Simulations

[![PyPI version](https://badge.fury.io/py/holopy.svg)](https://badge.fury.io/py/holopy)
[![Documentation Status](https://readthedocs.org/projects/holopy/badge/?version=latest)](https://holopy.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://github.com/bryceweiner/holopy/actions/workflows/tests.yml/badge.svg)](https://github.com/bryceweiner/holopy/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/bryceweiner/holopy/branch/main/graph/badge.svg)](https://codecov.io/gh/bryceweiner/holopy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

HoloPy is a Python library designed for simulating and analyzing holographic cosmology and gravity systems. It provides tools for:

- Simulating photon propagation in causal patches
- Modeling holographic corrections to gravity
- Analyzing information processing rates in holographic systems
- Visualizing holographic space-time geometries

## Installation

```bash
pip install holopy
```

For development installation:

```bash
git clone https://github.com/bryceweiner/holopy.git
cd holopy
pip install -e ".[dev]"
```

## Quick Start

```python
import holopy as hp
from holopy.models import PhotonSimulation
from holopy.visualization import plot_photon_path

# Create a photon simulation
sim = PhotonSimulation(
    initial_position=[0, 0, 0],
    initial_momentum=[1, 0, 0],
    patch_size=1.0,
    time_steps=1000
)

# Run the simulation
results = sim.run()

# Visualize the results
plot_photon_path(results)
```

## Features

- **Photon Simulation**: Simulate photon propagation in causal patches with holographic corrections
- **Gravity Models**: Implement various holographic gravity models
- **Information Processing**: Calculate information processing rates in holographic systems
- **Visualization**: Tools for visualizing holographic space-time geometries
- **Analysis**: Utilities for analyzing simulation results

## Documentation

Full documentation is available at [holopy.readthedocs.io](https://holopy.readthedocs.io).

## Examples

Check out the `examples` directory for more detailed examples:

- `photon_example.py`: Basic photon simulation in a causal patch
- `gravity_example.py`: Holographic gravity simulation
- `information_example.py`: Information processing rate calculation

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use HoloPy in your research, please cite:

```bibtex
@software{holopy2024,
  author = {Bryce Weiner},
  title = {HoloPy: A Python Library for Holographic Cosmology},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/bryceweiner/holopy}}
}
```

## Contact

For questions or support, please open an issue on GitHub or contact bryce.physics@gmail.com.

## Acknowledgments

- Based on research published in IPI Letters:
  - "E-mode Polarization Phase Transitions Reveal a Fundamental Parameter of the Universe" (2025)
  - "Holographic Information Rate as a Resolution to Contemporary Cosmological Tensions" (2025)
  - "Resolving the Hubble Tension in Holographic Theory" (2025)
- Thanks to all researchers working on holographic cosmology and holographic gravity theories

## References

1. Weiner, B. (2025). "E-mode Polarization Phase Transitions Reveal a Fundamental Parameter of the Universe." IPI Letters, 3(1):31-39. https://doi.org/10.59973/ipil.150
2. Weiner, B. (2025). "Holographic Information Rate as a Resolution to Contemporary Cosmological Tensions." IPI Letters, 3(2):8-22. https://doi.org/10.59973/ipil.170
3. Weiner, B. (2025). "Resolving the Hubble Tension in Holographic Theory." IPI Letters, X(X):X-X. https://doi.org/10.59973/ipil.xx 