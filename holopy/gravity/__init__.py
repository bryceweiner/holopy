"""
Gravity Module for HoloPy.

This module implements gravitational phenomena from an information-theoretic perspective,
including modified Einstein field equations, emergent spacetime metrics, and information-based
derivation of spacetime curvature in the holographic framework.
"""

from holopy.gravity.einstein_field import (
    ModifiedEinsteinField,
    compute_k_tensor
)
from holopy.gravity.emergent_metric import (
    InfoSpacetimeMetric,
    compute_curvature_from_info
)
from holopy.gravity.black_holes import (
    black_hole_entropy,
    hawking_radiation_rate,
    information_preservation
)
from holopy.gravity.spacetime import (
    metric_from_quantum_state,
    compute_riemann_tensor,
    compute_ricci_tensor,
    compute_ricci_scalar
)

__all__ = [
    'ModifiedEinsteinField',
    'compute_k_tensor',
    'InfoSpacetimeMetric',
    'compute_curvature_from_info',
    'black_hole_entropy',
    'hawking_radiation_rate',
    'information_preservation',
    'metric_from_quantum_state',
    'compute_riemann_tensor',
    'compute_ricci_tensor',
    'compute_ricci_scalar'
] 