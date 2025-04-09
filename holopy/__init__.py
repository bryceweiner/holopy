"""
HoloPy: A Python Library for Holographic Cosmology and Holographic gravity Simulations

HoloPy is a comprehensive library for exploring and validating the E8×E8 heterotic
structure framework and holographic universe theory. It provides tools for researchers,
physicists, and cosmologists to simulate, visualize, and analyze holographic
cosmology phenomena across multiple scales.

The library implements mathematical formulations, constants, and equations derived
from the holographic gravity framework, enabling the study of quantum decoherence,
emergent spacetime, and cosmological evolution from an information-theoretic
perspective.

Modules:
    constants: Fundamental constants and conversion factors
    e8: E8×E8 heterotic structure implementations
    info: Information current tensor calculations
    quantum: Quantum mechanics with information processing constraints
    gravity: Gravitational phenomena from information perspective
    cosmology: Cosmological models with holographic constraints
    dsqft: de Sitter/Quantum Field Theory correspondence framework
    utils: Support functions and visualization tools
    io: Data input/output utilities

For more information, see the documentation at https://holopy.readthedocs.io
"""

__version__ = '0.1.0'
__author__ = 'HoloPy Development Team'

# Import key subpackages for direct access
import holopy.constants

# Make key modules available at package level
from holopy import constants
from holopy import e8
from holopy import info
from holopy import quantum
from holopy import gravity
from holopy import cosmology
from holopy import dsqft
from holopy import utils
from holopy import io

# Import specific cosmology modules for easier access
from holopy.cosmology import expansion
from holopy.cosmology import hubble_tension
from holopy.cosmology import cmb
from holopy.cosmology import early_universe

# Import specific dsqft modules for easier access
from holopy.dsqft import CausalPatch
from holopy.dsqft import DSQFTSimulation

# Import specific utilities for easier access
from holopy.utils import math_utils
from holopy.utils import visualization

# Define what gets imported with "from holopy import *"
__all__ = [
    'constants',
    'e8',
    'info',
    'quantum',
    'gravity',
    'cosmology',
    'dsqft',
    'utils',
    'io',
    'expansion',
    'hubble_tension',
    'cmb',
    'early_universe',
    'CausalPatch',
    'DSQFTSimulation',
    'math_utils',
    'visualization'
]

# Show a welcome message when the package is imported
print(f"HoloPy v{__version__} - Python Library for Holographic Cosmology and Holographic gravity Simulations")
print("For more information, see https://holopy.readthedocs.io")
