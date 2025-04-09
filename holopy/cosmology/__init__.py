"""
Cosmology Module for HoloPy.

This module implements cosmological models with holographic constraints,
including universe expansion with information constraints, Hubble tension analysis,
CMB power spectrum computation, and early universe simulations.
"""

from holopy.cosmology.expansion import (
    HolographicExpansion,
    scale_factor_evolution,
    hubble_parameter
)
from holopy.cosmology.hubble_tension import (
    HubbleTensionAnalyzer,
    clustering_coefficient_constraint,
    predict_h0
)
from holopy.cosmology.cmb import (
    CMBSpectrum,
    compute_power_spectrum,
    e8_correction_factor
)
from holopy.cosmology.early_universe import (
    simulate_early_universe,
    compute_critical_transitions,
    inflation_parameters
)

__all__ = [
    'HolographicExpansion',
    'scale_factor_evolution',
    'hubble_parameter',
    'HubbleTensionAnalyzer',
    'clustering_coefficient_constraint',
    'predict_h0',
    'CMBSpectrum',
    'compute_power_spectrum',
    'e8_correction_factor',
    'simulate_early_universe',
    'compute_critical_transitions',
    'inflation_parameters'
] 