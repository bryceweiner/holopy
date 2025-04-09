"""
Conversion Factors Module for HoloPy.

This module implements conversion factors between different unit systems
and relationships between physical quantities in the holographic framework.
"""

import numpy as np
from scipy import constants as scipy_constants
from .physical_constants import PhysicalConstants

class ConversionFactors:
    """
    Class containing conversion factors between different physical quantities
    and unit systems relevant to the holographic framework.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConversionFactors, cls).__new__(cls)
            cls._instance._initialize_factors()
        return cls._instance
    
    def _initialize_factors(self):
        """Initialize all conversion factors."""
        pc = PhysicalConstants()
        
        # Planck units conversions
        self.length_planck = np.sqrt(scipy_constants.hbar * scipy_constants.G / scipy_constants.c**3)  # Planck length (m)
        self.time_planck = np.sqrt(scipy_constants.hbar * scipy_constants.G / scipy_constants.c**5)  # Planck time (s)
        self.mass_planck = np.sqrt(scipy_constants.hbar * scipy_constants.c / scipy_constants.G)  # Planck mass (kg)
        self.temperature_planck = np.sqrt(scipy_constants.hbar * scipy_constants.c**5 / (scipy_constants.G * scipy_constants.k))  # Planck temperature (K)
        self.energy_planck = self.mass_planck * scipy_constants.c**2  # Planck energy (J)
        
        # Information-related conversion factors
        self.info_spacetime_factor = pc.kappa_pi  # π^4/24, dimensionless
        self.entropy_area_factor = 1 / (4 * scipy_constants.hbar * scipy_constants.G / scipy_constants.c**3)  # 1/4 in Planck units
        self.info_entropy_factor = np.log(2)  # Convert bits to nats
        
        # Holographic gravity conversion factors
        self.info_mass_factor = pc.gamma * scipy_constants.c**2 / scipy_constants.G  # kg/bit/s
        self.info_curvature_factor = 8 * np.pi * scipy_constants.G / scipy_constants.c**4  # m/kg
        
        # Cosmology conversion factors
        self.hubble_gamma_ratio = 8 * np.pi  # H/γ ≈ 8π
        self.redshift_distance_factor = scipy_constants.c / pc.hubble_parameter  # m/(unit redshift)
        
        # Natural unit conversions (ħ=c=1)
        self.hbar_to_si = scipy_constants.hbar  # J·s
        self.c_to_si = scipy_constants.c  # m/s
        self.natural_to_si_energy = scipy_constants.hbar * scipy_constants.c  # J·m
        self.natural_to_si_length = scipy_constants.hbar / (scipy_constants.c * self.mass_planck)  # m
        
        # E8 length scale
        self.e8_length_scale = self.length_planck * np.sqrt(pc.kappa_pi)  # m

# Single global instance for convenience
CONVERSION_FACTORS = ConversionFactors()

def get_length_in_planck_units(length_si):
    """
    Convert a length in SI units (meters) to Planck units.
    
    Args:
        length_si (float): Length in meters
        
    Returns:
        float: Length in Planck units (dimensionless)
    """
    return length_si / ConversionFactors().length_planck

def get_time_in_planck_units(time_si):
    """
    Convert a time in SI units (seconds) to Planck units.
    
    Args:
        time_si (float): Time in seconds
        
    Returns:
        float: Time in Planck units (dimensionless)
    """
    return time_si / ConversionFactors().time_planck

def get_mass_in_planck_units(mass_si):
    """
    Convert a mass in SI units (kilograms) to Planck units.
    
    Args:
        mass_si (float): Mass in kilograms
        
    Returns:
        float: Mass in Planck units (dimensionless)
    """
    return mass_si / ConversionFactors().mass_planck

def get_entropy_from_area(area_si):
    """
    Calculate entropy from area according to the holographic principle.
    
    Args:
        area_si (float): Area in square meters
        
    Returns:
        float: Entropy in natural units (nats)
    """
    return area_si * ConversionFactors().entropy_area_factor

def get_entropy_from_bits(bits):
    """
    Convert information in bits to entropy in natural units.
    
    Args:
        bits (float): Information content in bits
        
    Returns:
        float: Entropy in natural units (nats)
    """
    return bits * ConversionFactors().info_entropy_factor

def get_information_from_mass(mass_si, time_si):
    """
    Calculate the information processing capacity from mass and time.
    
    Args:
        mass_si (float): Mass in kilograms
        time_si (float): Time interval in seconds
        
    Returns:
        float: Information processing capacity in bits
    """
    return mass_si * time_si / ConversionFactors().info_mass_factor 