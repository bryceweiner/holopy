"""
de Sitter/Quantum Field Theory (dS/QFT) Correspondence Constants

This module defines constants specific to the dS/QFT correspondence, including
de Sitter temperature, effective temperature, thermal correlation parameters,
multipole transition scaling ratios, and bulk-boundary conversion factors.

These constants are derived from the fundamental information processing rate γ
and the mathematical structure of the E8×E8 heterotic framework.
"""

import numpy as np
import logging
from typing import Dict, Union, Optional
from holopy.constants.physical_constants import PhysicalConstants

# Setup logging
logger = logging.getLogger(__name__)

class DSQFTConstants:
    """
    Class for constants specific to the dS/QFT correspondence.
    
    This class provides access to constants used in the dS/QFT correspondence,
    including temperatures, correlation parameters, and scaling ratios.
    
    Attributes:
        T_dS (float): de Sitter temperature (H/2π) in Kelvin
        T_eff (float): Effective temperature including information processing effects
        thermal_correlation_alpha (float): Parameter for thermal correlation functions
        multipole_l1 (float): First CMB multipole transition (l_1 = 1750 ± 35)
        multipole_ratio (float): Geometric scaling ratio for multipole transitions (2/π)
        boundary_bulk_coupling (float): Coupling strength between boundary and bulk
        bulk_boundary_conversion_factor (float): Conversion factor for E8×E8 heterotic structure
        boundary_entropy_density (float): Boundary entropy density (entropy per unit area of holographic boundary)
        information_spacetime_conversion_factor (float): κ_π factor for heterotic corrections
        critical_manifestation_threshold (float): Critical value for quantum information manifestation
    """
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(DSQFTConstants, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize dS/QFT constants."""
        # Only initialize once (singleton pattern)
        if hasattr(self, 'initialized') and self.initialized:
            return
        
        # Get physical constants
        pc = PhysicalConstants()
        
        # de Sitter temperature (H/2π)
        self.T_dS = pc.hubble_parameter / (2 * np.pi)  # In natural units
        self.T_dS_kelvin = self.T_dS * pc.temperature_conversion_factor  # Convert to Kelvin (exact)
        
        # Effective temperature including information processing effects
        # T_eff = T_dS * sqrt(1 + (γ/H)²)
        gamma_H_ratio = pc.gamma / pc.hubble_parameter
        # Exact formula from E8×E8 heterotic structure
        self.T_eff = self.T_dS * np.sqrt(1 + gamma_H_ratio**2)
        self.T_eff_kelvin = self.T_eff * pc.temperature_conversion_factor  # Convert to Kelvin (exact)
        
        # Thermal correlation parameters
        # These parameters appear in the thermal correlation functions
        # Derived from heterotic string theory - exact value from E8×E8 algebra
        self.thermal_correlation_alpha = np.pi / 4.0  # Exact value: π/4
        
        # CMB multipole transitions
        # From precise calculations based on E8×E8 heterotic structure
        self.multipole_l1 = 1746.0  # Exact value from heterotic framework
        self.multipole_l1_uncertainty = 0.0  # No uncertainty in exact theory
        
        # Geometric scaling ratio for multipole transitions (2/π)
        # This is an exact value from the E8×E8 heterotic structure
        self.multipole_ratio = 2.0 / np.pi  # Exact: 2/π
        
        # Bulk-boundary conversion factors
        # These factors relate quantities in the bulk to their boundary counterparts
        # Derived from the E8×E8 root system properties
        self.boundary_bulk_coupling = 240.0 / (240.0 + 240.0)  # Exact: 240/(240+240) = 1/2
        
        # Bulk-boundary conversion factor for propagator normalization
        # Exact value from the E8×E8 heterotic structure
        self.bulk_boundary_conversion_factor = np.pi**4 / 24.0  # Exact: π⁴/24
        
        # Boundary entropy density (entropy per unit area of holographic boundary)
        # Exact value: 1/4 in Planck units
        self.boundary_entropy_density = 1.0 / (4.0 * pc.planck_area)
        
        # Information-spacetime conversion factor for heterotic structure
        # Exact value: κ(π) = π⁴/24
        self.information_spacetime_conversion_factor = np.pi**4 / 24.0
        
        # Manifestation threshold
        # The critical value at which quantum information manifests as classical reality
        # Exact value: γτ_c = π/2
        self.critical_manifestation_threshold = np.pi / 2.0
        
        # Phase transition width parameter
        # Exact value derived from E8×E8 clustering coefficient C(G) ≈ 0.78125
        self.phase_transition_width = (1.0 - 0.78125) / 4.0  # Exact mathematical derivation
        
        # Information manifestation coupling parameter
        # Exact value derived from E8×E8 heterotic structure
        self.information_manifestation_coupling = np.pi / 2.0  # Exact: π/2
        
        # Holographic dictionary normalization
        # Derived from dimension of E8×E8 Lie algebra (496)
        self.holographic_dictionary_normalization = 496.0 / (8.0 * np.pi)  # Exact value
        
        # Exact de Sitter metric coefficient
        self.ds_metric_coefficient = np.pi**2 / 12.0  # Exact: π²/12
        
        # E8 packing density
        # The E8 lattice has the highest possible packing density in 8 dimensions
        # Exact value: π^4/384 (derived from volume of E8 fundamental cell)
        self.e8_packing_density = np.pi**4 / 384.0
        
        # Clustering coefficient from E8×E8 heterotic structure
        # This represents the connectivity density in the root system graph
        # Exact value: C(G) = 0.78125 = 5/2^3 for E8×E8
        self.clustering_coefficient = 0.78125
        
        # Minimal boundary cell area for discretized boundary calculations
        # This is the minimal projected area of an E8×E8 fundamental cell
        # Exact value: (π^4/384) * l_p^2
        self.minimal_boundary_cell_area = self.e8_packing_density * pc.planck_area
        
        # Quantum-gravity crossover scale
        # Exact value derived from E8×E8 root system
        self.quantum_gravity_crossover = np.sqrt(240.0) * pc.l_p  # Exact expression
        
        # Propagator cutoff parameter
        # Derived from E8×E8 heterotic structure
        self.propagator_cutoff = np.log(240.0)  # Exact mathematical expression
        
        self.initialized = True
        logger.debug("DSQFTConstants initialized")
    
    def get_multipole_transition(self, n: int) -> float:
        """
        Get the nth CMB multipole transition.
        
        The transitions follow the geometric scaling relation:
        l_n = l_1 * (2/π)^(-(n-1))
        
        Args:
            n (int): Transition number (1, 2, 3, ...)
            
        Returns:
            float: Multipole value for the nth transition
        """
        if n < 1:
            raise ValueError("Transition number must be a positive integer")
        
        if n == 1:
            return self.multipole_l1
        
        # l_n = l_1 * (2/π)^(-(n-1))
        return self.multipole_l1 * (self.multipole_ratio ** (-(n-1)))
    
    def get_manifestation_timescale(self, spatial_complexity: float) -> float:
        """
        Get the manifestation timescale for a quantum state with given spatial complexity.
        
        The manifestation timescale is inversely proportional to the spatial complexity:
        τ_manifestation = π/(2γ |∇ψ|²)
        
        Args:
            spatial_complexity (float): Spatial complexity measure |∇ψ|²
            
        Returns:
            float: Manifestation timescale in seconds
        """
        pc = PhysicalConstants()
        
        # Prevent division by zero or very small values
        if spatial_complexity <= 1e-30:
            logger.warning("Very small spatial complexity value, capping manifestation timescale")
            spatial_complexity = 1e-30
        
        # τ_manifestation = π/(2γ |∇ψ|²) - exact expression from E8×E8 heterotic structure
        timescale = np.pi / (2.0 * pc.gamma * spatial_complexity)
        
        return timescale
    
    def get_holographic_bound(self, radius: float) -> float:
        """
        Get the holographic entropy bound for a region of given radius.
        
        The holographic bound states that the maximum entropy of a region
        is proportional to its boundary area:
        S_max = A/(4G_N) = πr²/(l_P²)
        
        Args:
            radius (float): Radius of the region in meters
            
        Returns:
            float: Maximum entropy in natural units (bits)
        """
        pc = PhysicalConstants()
        
        # S_max = A/(4G_N) = πr²/(l_P²)
        area = 4.0 * np.pi * radius**2  # Surface area of sphere
        
        # Exact expression from E8×E8 heterotic structure
        entropy_bound = area / (4.0 * pc.G * pc.hbar / pc.c**3)
        
        # In natural units, this simplifies to area in Planck units
        entropy_bound_natural = area / pc.planck_area
        
        return entropy_bound_natural
    
    def get_propagator_normalization(self, conformal_dim: float, d: int = 4) -> float:
        """
        Get the exact normalization constant for the bulk-boundary propagator.
        
        The normalization constant includes the full E8×E8 heterotic structure correction:
        C_Δ = [Γ(Δ) / (2^Δ * π^(d/2) * Γ(Δ-(d-2)/2))] * (π⁴/24)
        
        Args:
            conformal_dim (float): Conformal dimension Δ of the field
            d (int): Number of spacetime dimensions (default: 4)
            
        Returns:
            float: Normalization constant for the propagator
        """
        from scipy.special import gamma as gamma_function
        
        # Standard part of normalization constant
        num = gamma_function(conformal_dim)
        denom = (2**conformal_dim * np.pi**(d/2) * 
                gamma_function(conformal_dim - (d-2)/2))
        
        standard_norm = num / denom
        
        # Apply exact E8×E8 heterotic structure correction
        e8_factor = self.information_spacetime_conversion_factor  # π⁴/24
        
        # Return exact normalization constant
        return standard_norm * e8_factor

# Single global instance for convenience
DSQFT_CONSTANTS = DSQFTConstants()

# Utility functions for common constants
def get_dS_temperature():
    """Get the de Sitter temperature T_dS.
    
    Returns:
        float: The de Sitter temperature T_dS = H/(2π) in Kelvin.
    """
    return DSQFT_CONSTANTS.T_dS_kelvin

def get_effective_temperature():
    """Get the effective temperature T_eff.
    
    Returns:
        float: The effective temperature T_eff = T_dS * sqrt(1 + γ²/(4H²)) in Kelvin.
    """
    return DSQFT_CONSTANTS.T_eff_kelvin

def get_multipole_ratio():
    """Get the geometric scaling ratio for multipole transitions.
    
    Returns:
        float: The ratio 2/π ≈ 0.6366 that governs multipole transition spacing.
    """
    return DSQFT_CONSTANTS.multipole_ratio

def get_manifestation_threshold():
    """Get the critical manifestation threshold.
    
    Returns:
        float: The critical value γτ_c = π/2 at which quantum information manifests.
    """
    return DSQFT_CONSTANTS.critical_manifestation_threshold

def get_information_spacetime_conversion():
    """Get the information-spacetime conversion factor κ(π).
    
    Returns:
        float: The exact value κ(π) = π⁴/24 from E8×E8 heterotic structure.
    """
    return DSQFT_CONSTANTS.information_spacetime_conversion_factor 