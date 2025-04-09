"""
Physical Constants Module

This module provides fundamental physical constants needed for holographic
calculations, using the most precise values available from CODATA.
"""

import numpy as np

def get_gamma(hubble_parameter: float = None) -> float:
    """
    Get the information manifestation rate γ.
    
    The information manifestation rate γ is empirically determined to be
    γ = H/8π where H is the Hubble parameter.
    
    Args:
        hubble_parameter (float, optional): Custom Hubble parameter value
        
    Returns:
        float: Information manifestation rate γ in s⁻¹
    """
    if hubble_parameter is None:
        # Use standard value γ ≈ 1.89 × 10⁻²⁹ s⁻¹
        return 1.89e-29
    else:
        # Calculate γ = H/8π for custom Hubble parameter
        return hubble_parameter / (8 * np.pi)

def get_clustering_coefficient(network_type: str = 'E8xE8') -> float:
    """
    Get the clustering coefficient for a specific network type.
    
    The clustering coefficient C(G) quantifies the degree to which nodes
    in a network tend to cluster together. For the E8×E8 heterotic structure,
    this is empirically determined to be C(G) ≈ 0.78125.
    
    Args:
        network_type (str): Type of network ('E8xE8', 'SO32', or 'custom')
        
    Returns:
        float: Clustering coefficient C(G)
    """
    if network_type.upper() == 'E8XE8':
        # Standard value for E8×E8 heterotic structure
        return 0.78125
    elif network_type.upper() == 'SO32':
        # Value for SO(32) heterotic structure
        return 0.75000
    else:
        # Default to E8×E8 value
        return 0.78125

def get_planck_area() -> float:
    """
    Get the Planck area.
    
    The Planck area is the square of the Planck length, representing the
    fundamental quantum of area in holographic theories.
    
    Returns:
        float: Planck area in m²
    """
    return PHYSICAL_CONSTANTS.planck_area

def get_planck_mass() -> float:
    """
    Get the Planck mass.
    
    The Planck mass is a fundamental mass scale in quantum gravity,
    defined as sqrt(ℏc/G).
    
    Returns:
        float: Planck mass in kg
    """
    return PHYSICAL_CONSTANTS.planck_mass

def get_planck_length() -> float:
    """
    Get the Planck length.
    
    The Planck length is the fundamental length scale in quantum gravity,
    defined as sqrt(ℏG/c³).
    
    Returns:
        float: Planck length in m
    """
    return PHYSICAL_CONSTANTS.planck_length

def get_planck_time() -> float:
    """
    Get the Planck time.
    
    The Planck time is the fundamental time scale in quantum gravity,
    defined as sqrt(ℏG/c⁵).
    
    Returns:
        float: Planck time in s
    """
    return PHYSICAL_CONSTANTS.planck_time

def get_planck_energy() -> float:
    """
    Get the Planck energy.
    
    The Planck energy is the fundamental energy scale in quantum gravity,
    defined as sqrt(ℏc⁵/G).
    
    Returns:
        float: Planck energy in J
    """
    return PHYSICAL_CONSTANTS.planck_energy

class PhysicalConstants:
    """
    Class containing fundamental physical constants.
    
    All constants are in SI units unless otherwise specified.
    Values are from CODATA 2018 recommended values.
    """
    
    def __init__(self):
        """Initialize physical constants."""
        # Fundamental constants
        self.c = 2.99792458e8  # Speed of light (m/s)
        self.hbar = 1.054571817e-34  # Reduced Planck constant (J⋅s)
        self.G_newton = 6.67430e-11  # Gravitational constant (m³/kg⋅s²)
        self.G = self.G_newton  # Alias for gravitational constant
        self.k_boltzmann = 1.380649e-23  # Boltzmann constant (J/K)
        
        # Electromagnetic constants
        self.epsilon_0 = 8.8541878128e-12  # Vacuum permittivity (F/m)
        self.mu_0 = 1.25663706212e-6  # Vacuum permeability (N/A²)
        self.electron_charge = 1.602176634e-19  # Elementary charge (C)
        self.elementary_charge = self.electron_charge  # Alias for elementary charge
        
        # Particle masses
        self.electron_mass = 9.1093837015e-31  # Electron mass (kg)
        self.proton_mass = 1.67262192369e-27  # Proton mass (kg)
        self.neutron_mass = 1.67492749804e-27  # Neutron mass (kg)
        
        # Atomic and nuclear constants
        self.fine_structure = 7.297352569e-3  # Fine structure constant
        self.rydberg = 10973731.568160  # Rydberg constant (m⁻¹)
        self.bohr_radius = 5.29177210903e-11  # Bohr radius (m)
        self.atomic_mass = 1.66053906660e-27  # Atomic mass unit (kg)
        
        # Cosmological constants
        self.hubble_parameter = 67.4 * 1000 / (3.086e22)  # Hubble parameter (s⁻¹)
        self.gamma = get_gamma(self.hubble_parameter)  # Information manifestation rate (s⁻¹)
        self.cosmological_constant = 1.1056e-52  # Cosmological constant (m⁻²)
        
        # Planck units (base)
        self.planck_length = np.sqrt(self.hbar * self.G_newton / self.c**3)  # Planck length (m)
        self.planck_mass = np.sqrt(self.hbar * self.c / self.G_newton)  # Planck mass (kg)
        self.planck_time = self.planck_length / self.c  # Planck time (s)
        self.planck_temperature = self.planck_mass * self.c**2 / self.k_boltzmann  # Planck temperature (K)
        
        # Planck units (derived)
        self.planck_area = self.planck_length**2  # Planck area (m²)
        self.planck_volume = self.planck_length**3  # Planck volume (m³)
        self.planck_density = self.planck_mass / self.planck_volume  # Planck density (kg/m³)
        self.planck_energy = self.planck_mass * self.c**2  # Planck energy (J)
        self.planck_momentum = self.planck_mass * self.c  # Planck momentum (kg⋅m/s)
        self.planck_force = self.planck_energy / self.planck_length  # Planck force (N)
        self.planck_power = self.planck_energy / self.planck_time  # Planck power (W)
        self.planck_pressure = self.planck_force / self.planck_area  # Planck pressure (Pa)
        self.planck_charge = np.sqrt(4 * np.pi * self.epsilon_0 * self.hbar * self.c)  # Planck charge (C)
        self.planck_electric_field = self.planck_force / self.planck_charge  # Planck electric field (V/m)
        self.planck_magnetic_field = self.planck_electric_field / self.c  # Planck magnetic field (T)
        
        # Planck unit aliases (for convenience)
        self.l_p = self.planck_length  # Planck length
        self.m_p = self.planck_mass  # Planck mass
        self.t_p = self.planck_time  # Planck time
        self.T_p = self.planck_temperature  # Planck temperature
        self.E_p = self.planck_energy  # Planck energy
        self.rho_p = self.planck_density  # Planck density
        self.P_p = self.planck_pressure  # Planck pressure
        self.q_p = self.planck_charge  # Planck charge
        
        # Temperature conversion factors
        self.temperature_conversion_factor = 1.0  # Natural units to Kelvin
        self.temperature_energy_factor = self.k_boltzmann  # K to J
        self.temperature_mass_factor = self.k_boltzmann / self.c**2  # K to kg
        
        # Mathematical constants
        self.pi = np.pi
        self.e = np.e
        
        # Information-theoretic constants
        self.information_entropy_factor = self.pi**4 / 24  # κ(π) factor
        self.kappa_pi = self.information_entropy_factor  # Alias for κ(π)
        self.clustering_coefficient = get_clustering_coefficient()  # C(G) for E8×E8 network
        
        # Holographic constants
        self.bulk_boundary_factor = self.pi * self.gamma / self.hubble_parameter
        self.manifestation_scale = self.c / self.gamma  # Length scale for information manifestation

# Create a singleton instance
PHYSICAL_CONSTANTS = PhysicalConstants()

# Export both the class and the singleton instance
__all__ = [
    'PhysicalConstants', 
    'PHYSICAL_CONSTANTS', 
    'get_gamma', 
    'get_clustering_coefficient',
    'get_planck_area',
    'get_planck_mass',
    'get_planck_length',
    'get_planck_time',
    'get_planck_energy'
] 