"""
Information Module for HoloPy.

This package implements the information current tensor and related calculations
that form the foundation for information-based emergent spacetime.
"""

# Import and expose key functions and classes
from .current import (
    InfoCurrentTensor,
    compute_divergence
)

from .tensor import (
    compute_higher_order_functional
)

from .conservation import (
    information_conservation
)

from .processing import (
    information_flow
)

# Define what gets imported with "from holopy.info import *"
__all__ = [
    # Classes
    'InfoCurrentTensor',
    
    # Functions
    'compute_divergence',
    'compute_higher_order_functional',
    'information_conservation',
    'information_flow'
] 