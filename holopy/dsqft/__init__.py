"""
de Sitter/Quantum Field Theory (dS/QFT) Correspondence Module

This module implements the dS/QFT correspondence framework for holographic cosmology,
providing a comprehensive simulation framework for exploring the relationship between
quantum fields on the boundary and gravity in the bulk of de Sitter space.

The dS/QFT correspondence extends holographic principles to universes with positive
cosmological constant, providing a framework compatible with observational cosmology
while maintaining the mathematical elegance of holographic theories.

Key Components:
    - Bulk-boundary propagator with information processing constraints
    - Field-operator dictionary mapping boundary operators to bulk fields
    - Modified correlation functions with information processing effects
    - Information transport across the holographic boundary
    - Matter-entropy coupling mechanisms
    - Causal patch framework for defining observation regions
    - Simulation interface for holographic cosmology
    - Query system for extracting physical observables

This module integrates with the existing HoloPy functionality, particularly with the
E8 structure, quantum, gravity, and information modules.
"""

__all__ = [
    'BulkBoundaryPropagator',
    'FieldOperatorDictionary',
    'ModifiedCorrelationFunction',
    'InformationTransport',
    'MatterEntropyCoupling',
    'CausalPatch',
    'DSQFTSimulation',
    'DSQFTQuery'
]

# Import key classes for easier access
from holopy.dsqft.propagator import BulkBoundaryPropagator
from holopy.dsqft.dictionary import FieldOperatorDictionary
from holopy.dsqft.correlation import ModifiedCorrelationFunction
from holopy.dsqft.transport import InformationTransport
from holopy.dsqft.coupling import MatterEntropyCoupling, InformationManifestationTensor
from holopy.dsqft.causal_patch import CausalPatch
from holopy.dsqft.simulation import DSQFTSimulation
from holopy.dsqft.query import DSQFTQuery, QueryType, QueryResult

# Add submodule references for direct imports
from holopy.dsqft import propagator
from holopy.dsqft import dictionary
from holopy.dsqft import correlation
from holopy.dsqft import transport
from holopy.dsqft import coupling
from holopy.dsqft import causal_patch
from holopy.dsqft import simulation
from holopy.dsqft import query

# Version info
__version__ = '0.1.0' 