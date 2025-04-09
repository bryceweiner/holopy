"""
E8 Package for HoloPy.

This package implements the E8×E8 heterotic structure, root systems, and projections
that form the mathematical foundation of the holographic universe theory.
"""

# Import and expose key functions and classes
from .root_system import (
    RootSystem,
    get_root_vectors
)

from .lattice import (
    E8Lattice
)

from .heterotic import (
    E8E8Heterotic
)

from .projections import (
    project_to_4d,
    compute_killing_form
)

# Define what gets imported with "from holopy.e8 import *"
__all__ = [
    # Classes
    'RootSystem',
    'E8Lattice',
    'E8E8Heterotic',
    
    # Functions
    'get_root_vectors',
    'project_to_4d',
    'compute_killing_form'
] 