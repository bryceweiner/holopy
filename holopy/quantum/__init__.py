"""
Quantum Module for HoloPy.

This module implements quantum mechanics with information processing constraints,
including the modified Schrödinger equation, decoherence functional, and related
quantum measurements in the holographic framework.
"""

from holopy.quantum.modified_schrodinger import (
    WaveFunction, 
    ModifiedSchrodinger, 
    Evolution
)
from holopy.quantum.decoherence import (
    DecoherenceFunctional, 
    coherence_decay, 
    spatial_complexity,
    decoherence_evolution
)
from holopy.quantum.measurement import (
    quantum_measurement,
    expectation_value,
    measurement_probability
)
from holopy.quantum.entanglement import (
    entanglement_entropy,
    mutual_information,
    max_entanglement_rate
)

__all__ = [
    'WaveFunction',
    'ModifiedSchrodinger',
    'Evolution',
    'DecoherenceFunctional',
    'coherence_decay',
    'spatial_complexity',
    'decoherence_evolution',
    'quantum_measurement',
    'expectation_value',
    'measurement_probability',
    'entanglement_entropy',
    'mutual_information',
    'max_entanglement_rate'
] 