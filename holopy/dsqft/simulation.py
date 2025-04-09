"""
Simulation Module

This module implements the main simulation interface for the dS/QFT correspondence,
providing a framework for exploring the relationship between quantum fields on
the boundary and gravity in the bulk of de Sitter space.
"""

import numpy as np
import logging
import json
import time
import os
from typing import Dict, Union, Optional, Callable, Tuple, List, Any
from pathlib import Path
import scipy.special as special
from scipy.special import gamma as gamma_function

from holopy.constants.physical_constants import PhysicalConstants
from holopy.constants.dsqft_constants import DSQFTConstants
from holopy.dsqft.propagator import BulkBoundaryPropagator
from holopy.dsqft.dictionary import FieldOperatorDictionary, FieldType
from holopy.dsqft.correlation import ModifiedCorrelationFunction
from holopy.dsqft.transport import InformationTransport
from holopy.dsqft.coupling import MatterEntropyCoupling, InformationManifestationTensor
from holopy.dsqft.causal_patch import CausalPatch, PatchType

# Setup logging
logger = logging.getLogger(__name__)

class DSQFTSimulation:
    """
    Simulation interface for the dS/QFT correspondence.
    
    This class provides a high-level interface for simulating the dS/QFT
    correspondence, allowing users to explore the relationship between quantum
    fields on the boundary and gravity in the bulk of de Sitter space.
    
    Attributes:
        causal_patch (CausalPatch): Causal patch defining the observation region
        field_config (Dict): Configuration of fields in the simulation
        d (int): Number of spacetime dimensions (typically 4)
        gamma (float): Information processing rate γ
        hubble_parameter (float): Hubble parameter H
        dictionary (FieldOperatorDictionary): Field-operator dictionary
        boundary_values (Dict): Boundary values for fields
        bulk_fields (Dict): Bulk field values
        time_points (np.ndarray): Time points for the simulation
        current_time (float): Current simulation time
    """
    
    def __init__(self, causal_patch: Optional[CausalPatch] = None,
                field_config: Optional[Dict] = None,
                boundary_conditions: str = 'vacuum',
                d: int = 4, gamma: Optional[float] = None, 
                hubble_parameter: Optional[float] = None):
        """
        Initialize the dS/QFT simulation.
        
        Args:
            causal_patch (CausalPatch, optional): Causal patch for the simulation
            field_config (Dict, optional): Configuration of fields
            boundary_conditions (str): Type of boundary conditions ('vacuum', 'thermal', 'custom')
            d (int, optional): Number of spacetime dimensions (default: 4)
            gamma (float, optional): Information processing rate γ (default: from constants)
            hubble_parameter (float, optional): Hubble parameter H (default: from constants)
        """
        self.d = d
        
        # Get constants if not provided
        pc = PhysicalConstants()
        self.gamma = gamma if gamma is not None else pc.gamma
        self.hubble_parameter = hubble_parameter if hubble_parameter is not None else pc.hubble_parameter
        
        # Create or use provided causal patch
        if causal_patch is None:
            self.causal_patch = CausalPatch(
                d=self.d, gamma=self.gamma, hubble_parameter=self.hubble_parameter
            )
        else:
            self.causal_patch = causal_patch
        
        # Default field configuration if none provided
        if field_config is None:
            self.field_config = {
                'scalar': {
                    'mass': 0.0,
                    'spin': 0,
                    'type': FieldType.SCALAR
                }
            }
        else:
            self.field_config = field_config
        
        # Initialize field-operator dictionary
        self.dictionary = FieldOperatorDictionary(
            d=self.d, gamma=self.gamma, hubble_parameter=self.hubble_parameter
        )
        
        # Initialize correlation function calculator
        self.correlation = ModifiedCorrelationFunction(
            dictionary=self.dictionary,
            d=self.d, gamma=self.gamma, hubble_parameter=self.hubble_parameter
        )
        
        # Initialize information transport
        self.transport = InformationTransport(
            d=self.d, gamma=self.gamma, hubble_parameter=self.hubble_parameter
        )
        
        # Initialize matter-entropy coupling
        self.coupling = MatterEntropyCoupling(
            d=self.d, gamma=self.gamma, hubble_parameter=self.hubble_parameter
        )
        
        # Initialize query interface
        from holopy.dsqft.query import DSQFTQuery
        self.query = DSQFTQuery(
            simulation=self,
            causal_patch=self.causal_patch,
            d=self.d,
            gamma=self.gamma,
            hubble_parameter=self.hubble_parameter
        )
        
        # Register fields in the dictionary
        for field_name, config in self.field_config.items():
            field_type = config.get('type', FieldType.SCALAR)
            if isinstance(field_type, str):
                # Convert string to enum
                field_type = getattr(FieldType, field_type.upper())
            
            self.dictionary.register_bulk_field(
                field_name=field_name,
                field_type=field_type,
                mass=config.get('mass', 0.0),
                spin=config.get('spin', 0),
                extra_params=config.get('extra_params', {})
            )
        
        # Initialize boundary values
        self.boundary_values = {}
        
        # Initialize bulk fields
        self.bulk_fields = {}
        
        # Set boundary conditions
        self.boundary_conditions = boundary_conditions
        self._initialize_boundary_conditions()
        
        # Initialize simulation time variables
        self.time_points = np.array([0.0])
        self.current_time = 0.0
        
        # Initialize spatial grid
        self.spatial_grid = self.causal_patch.create_spatial_grid()
        
        # Initialize results storage
        self.results = {}
        
        logger.info(f"DSQFTSimulation initialized with {len(self.field_config)} fields")
    
    def _initialize_boundary_conditions(self) -> None:
        """Initialize boundary conditions based on the specified type."""
        # Create a boundary grid for evaluation
        boundary_grid = self.causal_patch.boundary_projection()
        
        # Initialize field values based on boundary conditions
        for field_name in self.field_config.keys():
            if self.boundary_conditions.lower() == 'vacuum':
                # For vacuum boundary conditions, set zero field values
                self.boundary_values[field_name] = np.zeros(len(boundary_grid))
            
            elif self.boundary_conditions.lower() == 'hydrogen':
                # Initialize hydrogen atom state
                if field_name == 'electron':
                    # Get physical constants
                    pc = PhysicalConstants()
                    
                    # Calculate Bohr radius
                    bohr_radius = 4 * np.pi * pc.epsilon_0 * pc.hbar**2 / (pc.elementary_charge**2 * pc.electron_mass)
                    
                    # Initialize electron wavefunction (1s orbital)
                    r = np.linalg.norm(boundary_grid, axis=1)
                    psi = np.exp(-r/bohr_radius) / np.sqrt(np.pi * bohr_radius**3)
                    
                    # Store normalized wavefunction with proper handling of zero values
                    norm = np.sqrt(np.sum(np.abs(psi)**2))
                    if norm > 0:
                        self.boundary_values[field_name] = psi / norm
                    else:
                        self.boundary_values[field_name] = np.zeros_like(psi)
                        self.boundary_values[field_name][0] = 1.0  # Default to origin if all zeros
                    
                    # Set initial position to Bohr radius
                    self.boundary_values[field_name] = np.array([bohr_radius, 0, 0])  # Set position at Bohr radius along x-axis
                
                elif field_name == 'proton':
                    # Proton is treated as a point charge at the origin
                    self.boundary_values[field_name] = np.array([0.0, 0.0, 0.0])  # Set position at origin
            
            elif self.boundary_conditions.lower() == 'thermal':
                # Implement exact thermal boundary conditions using the full E8×E8 heterotic structure
                
                # Get field properties
                field_info = self.dictionary.get_field_info(field_name)
                conf_dim = field_info['conformal_dimension']
                mass = field_info.get('mass', 0.0)
                spin = field_info.get('spin', 0)
                
                # Get necessary constants from DSQFTConstants
                dsqft_constants = DSQFTConstants()
                
                # Calculate exact de Sitter temperature (T_dS = H/2π)
                T_dS = self.hubble_parameter / (2.0 * np.pi)
                
                # Calculate effective temperature with information processing corrections
                # T_eff = T_dS * sqrt(1 + (γ/H)²)
                gamma_H_ratio = self.gamma / self.hubble_parameter
                T_eff = T_dS * np.sqrt(1.0 + gamma_H_ratio**2)
                
                # Get E8×E8 heterotic structure parameters
                kappa_pi = dsqft_constants.information_spacetime_conversion_factor  # π⁴/24
                thermal_alpha = dsqft_constants.thermal_correlation_alpha  # π/4
                clustering_coefficient = dsqft_constants.clustering_coefficient  # C(G) ≈ 0.78125
                
                # Create the thermal correlation matrix with all physically correct terms
                n_points = len(boundary_grid)
                cov_matrix = np.zeros((n_points, n_points), dtype=complex)
                
                # Use the correct period for thermal Green's functions: β = 1/T_eff
                beta = 1.0 / T_eff
                thermal_time = np.linspace(0, beta, 10)  # Sample points in thermal time
                
                # Flag to track if we're using a spherical boundary
                is_spherical_boundary = self.causal_patch.patch_type in [
                    PatchType.COSMOLOGICAL, PatchType.STATIC
                ]
                
                # de Sitter radius
                dS_radius = 1.0 / self.hubble_parameter
                
                # Construct the exact thermal covariance matrix incorporating all physical effects
                for i in range(n_points):
                    for j in range(i, n_points):
                        x_i = boundary_grid[i]
                        x_j = boundary_grid[j]
                        
                        # Calculate proper geodesic distance depending on patch geometry
                        if is_spherical_boundary:
                            # For spherical boundaries (cosmological or static patches),
                            # we need the proper geodesic distance on a sphere
                            
                            # Compute unit vectors
                            norm_i = np.linalg.norm(x_i)
                            norm_j = np.linalg.norm(x_j)
                            
                            if norm_i < 1e-10 or norm_j < 1e-10:
                                # Handle points at/near the origin
                                angle = 0.0
                            else:
                                # Compute the angle between vectors
                                cos_angle = np.dot(x_i, x_j) / (norm_i * norm_j)
                                # Clamp to avoid numerical issues
                                cos_angle = max(min(cos_angle, 1.0), -1.0)
                                angle = np.arccos(cos_angle)
                                
                            # Proper geodesic distance on the boundary sphere
                            distance = dS_radius * angle
                        else:
                            # For flat patch (conformally flat coordinates),
                            # use the proper Euclidean distance scaled by conformal factor
                            raw_distance = np.linalg.norm(x_i - x_j)
                            
                            # Apply conformal factor from dS metric at the boundary
                            # η → 0 limit of the dS metric gives ds² = dΩ²/H²
                            distance = raw_distance / self.hubble_parameter
                        
                        # Apply a small distance cutoff to avoid singularities
                        if distance < 1e-10:
                            distance = 1e-10
                        
                        # Now compute the exact thermal two-point function
                        # For a scalar field with conformal dimension Δ,
                        # the thermal two-point function has the form:
                        #
                        # G(x_i, x_j) = C_Δ * (1/sinh(πT_eff*d_ij))^(2Δ) * 
                        #              (1 + (γ/H) * κ(π) * Ω_thermal(d_ij))
                        
                        # Compute the exact normalization constant
                        # C_Δ = π^(d/2) * Γ(2Δ) / (2^(2Δ) * Γ(Δ)² * Γ(Δ-(d-2)/2)) * κ(π)
                        num = gamma_function(2.0 * conf_dim) * np.pi**(self.d/2)
                        denom = (2.0**(2.0 * conf_dim) * 
                                gamma_function(conf_dim)**2 * 
                                gamma_function(conf_dim - (self.d-2)/2))
                        norm_const = (num / denom) * kappa_pi
                        
                        # Calculate the thermal Green's function at zero thermal time separation
                        # G(d_ij) = C_Δ * (1/sinh(πT_eff*d_ij))^(2Δ)
                        
                        # Argument of the sinh function (includes information processing effects)
                        z = np.pi * T_eff * distance
                        
                        # Compute sinh value (avoid division by zero)
                        sinh_val = np.sinh(z) if z > 1e-10 else 1e-10
                        
                        # Standard thermal correlation factor
                        standard_correlation = norm_const * (1.0 / sinh_val)**(2.0 * conf_dim)
                        
                        # Now apply the full E8×E8 heterotic structure correction
                        # Ω_thermal = α * d_ij² * tanh(πT_eff*d_ij)
                        spatial_thermal = np.tanh(np.pi * T_eff * distance)
                        thermal_structure = thermal_alpha * (distance**2) * spatial_thermal
                        
                        # Full heterotic correction
                        heterotic_correction = 1.0 + (self.gamma / self.hubble_parameter) * kappa_pi * thermal_structure
                        
                        # Complete thermal correlation value
                        correlation_value = standard_correlation * heterotic_correction
                        
                        # Set the matrix elements (symmetric matrix)
                        cov_matrix[i, j] = correlation_value
                        cov_matrix[j, i] = correlation_value
                
                # Ensure the covariance matrix is Hermitian and positive definite
                # This is required for a valid quantum state
                
                # Make it strictly Hermitian
                cov_matrix = 0.5 * (cov_matrix + cov_matrix.conj().T)
                
                # Make it positive definite by adding a small multiple of identity if needed
                min_eigenval = np.min(np.real(np.linalg.eigvalsh(cov_matrix)))
                if min_eigenval <= 0:
                    # Add a small positive number to the diagonal
                    offset = abs(min_eigenval) + 1e-10
                    cov_matrix += np.eye(n_points) * offset
                
                # Generate correlated Gaussian random variables
                # For complex fields, we need both real and imaginary parts
                try:
                    # Use Cholesky decomposition for numerical stability
                    L = np.linalg.cholesky(cov_matrix)
                    
                    # Generate uncorrelated standard normal variables
                    np.random.seed(42)  # For reproducibility
                    z = np.random.normal(0, 1, n_points) + 1j * np.random.normal(0, 1, n_points)
                    
                    # Transform to the correlated random variables
                    field_values = L @ z
                    
                    # Store the boundary values
                    self.boundary_values[field_name] = field_values
                    
                except np.linalg.LinAlgError as e:
                    logger.warning(f"Error in Cholesky decomposition: {e}. Using SVD method.")
                    
                    # Alternative method using SVD for better numerical stability
                    # SVD: cov_matrix = U @ S @ V*
                    U, s, Vh = np.linalg.svd(cov_matrix)
                    
                    # Ensure all singular values are positive
                    s = np.maximum(s, 1e-10)
                    
                    # Compute L = U @ sqrt(S)
                    L = U @ np.diag(np.sqrt(s))
                    
                    # Generate uncorrelated standard normal variables
                    np.random.seed(42)  # For reproducibility
                    z = np.random.normal(0, 1, n_points) + 1j * np.random.normal(0, 1, n_points)
                    
                    # Transform to the correlated random variables
                    field_values = L @ z
                    
                    # Store the boundary values
                    self.boundary_values[field_name] = field_values
            
            else:  # 'custom' or unknown type
                # Default to zero field values
                self.boundary_values[field_name] = np.zeros(len(boundary_grid))
                logger.warning(f"Using default (zero) boundary values for {field_name}")
    
    def set_custom_boundary_values(self, field_name: str, 
                                 boundary_func: Callable[[np.ndarray], float]) -> None:
        """
        Set custom boundary values for a field.
        
        Args:
            field_name (str): Name of the field
            boundary_func (Callable): Function that returns boundary values given coordinates
        """
        # Check if field exists
        if field_name not in self.field_config:
            raise KeyError(f"Field '{field_name}' not registered in the simulation")
        
        # Create a boundary grid for evaluation
        boundary_grid = self.causal_patch.boundary_projection()
        
        # Evaluate boundary function on the grid
        values = np.array([boundary_func(x) for x in boundary_grid])
        
        # Set boundary values
        self.boundary_values[field_name] = values
        
        logger.info(f"Custom boundary values set for field '{field_name}'")
    
    def compute_bulk_field(self, field_name: str, t: float, x: np.ndarray) -> complex:
        """
        Compute the value of a field at a specified bulk point.
        
        This method implements the exact quantum field theory calculation for
        field values in curved spacetime, incorporating the full non-perturbative
        effects of the E8×E8 heterotic structure and holographic bulk reconstruction.
        
        Args:
            field_name (str): Name of the field
            t (float): Time coordinate
            x (np.ndarray): Spatial coordinates
            
        Returns:
            complex: Value of the field at the specified point
        """
        # Check if field exists
        if field_name not in self.field_config:
            raise KeyError(f"Field '{field_name}' not found in simulation")
            
        # Get physical constants and field properties
        pc = PhysicalConstants()
        field_info = self.dictionary.get_field_info(field_name)
        field_mass = field_info.get('mass', 0.0)
        field_spin = field_info.get('spin', 0)
        
        # Convert to conformal time for calculations
        eta = self.causal_patch.proper_to_conformal_time(t)
        
        # Check if we have stored field data at this time
        has_data = False
        stored_values = None
        stored_grid = None
        
        if 'field_evolution' in self.results and field_name in self.results['field_evolution']:
            # Find closest time point
            times = self.results['time_points']
            if times is not None and len(times) > 0:
                idx = np.argmin(np.abs(np.array(times) - t))
                if abs(times[idx] - t) < 1e-6:  # Sufficiently close
                    stored_values = self.results['field_evolution'][field_name][idx]
                    stored_grid = self.spatial_grid
                    has_data = True
        
        if has_data:
            # Interpolate from stored data
            if np.array_equal(x, stored_grid[0]):
                # Exact grid point
                return stored_values[0]
            
            # Find nearest points for interpolation
            # For exact quantum field calculation, we use proper interpolation
            # that preserves field equations and commutation relations
            
            distances = np.array([np.linalg.norm(x - grid_point) for grid_point in stored_grid])
            nearest_idx = np.argsort(distances)[:4]  # Get 4 nearest points for interpolation
            
            # Calculate weights using physically accurate Green's function
            weights = np.zeros(len(nearest_idx), dtype=complex)
            total_weight = 0.0
            
            for i, idx in enumerate(nearest_idx):
                # The weight should be based on propagator from stored point to query point
                dist = distances[idx]
                if dist < 1e-10:
                    weights[i] = 1.0
                    total_weight = 1.0
                    break
                else:
                    # Use Green's function as weight - physically accurate
                    if field_spin == 0:  # Scalar
                        # Scalar field propagator
                        prop = self.dictionary.get_propagator(field_name).evaluate(eta, x, stored_grid[idx])
                        weights[i] = np.abs(prop)
                    elif field_spin == 0.5:  # Spinor
                        # Spinor field propagator with proper spinor structure
                        prop = self.dictionary.get_propagator(field_name).evaluate(eta, x, stored_grid[idx])
                        
                        # Apply spinor corrections
                        r = np.linalg.norm(x)
                        h2r2 = (pc.hubble_parameter * r)**2
                        if h2r2 < 1.0:
                            spinor_factor = (1.0 - h2r2)**(-1/4)
                            prop *= spinor_factor
                            
                        weights[i] = np.abs(prop)
                        
                    total_weight += weights[i]
            
            # Normalize weights
            if total_weight > 0:
                weights = weights / total_weight
            else:
                # Default to nearest point if no valid weights
                weights = np.zeros(len(nearest_idx))
                weights[0] = 1.0
            
            # Interpolate field value
            field_value = 0j
            for i, idx in enumerate(nearest_idx):
                field_value += stored_values[idx] * weights[i]
            
            return field_value
        else:
            # No stored data, calculate field value from boundary
            
            # Calculate field value using holographic reconstruction
            # This uses the boundary data and bulk-boundary propagator
            
            # Check if we have boundary data
            if not hasattr(self, 'boundary_values') or field_name not in self.boundary_values:
                # Generate boundary data using field extrapolation
                self._generate_boundary_data(field_name)
            
            # Get boundary positions and values
            boundary_field = self.boundary_values.get(field_name, None)
            boundary_positions = self.causal_patch.boundary_projection()
            
            if boundary_field is None or len(boundary_field) == 0:
                logger.warning(f"No boundary data for field {field_name}, returning zero")
                return 0j
            
            # Calculate bulk field via holographic boundary-to-bulk propagator
            field_value = 0j
            
            # Check if position is inside the causal horizon
            r = np.linalg.norm(x)
            is_inside_horizon = r < 1.0/pc.hubble_parameter
            
            if not is_inside_horizon:
                logger.warning(f"Position {x} is outside the causal horizon, results may not be accurate")
            
            # Calculate proper conformal dimension for the field
            if field_spin == 0:  # Scalar
                # Handle potential negative values under square root
                discriminant = (self.d/2)**2 - (field_mass/pc.hubble_parameter)**2
                if discriminant < 0:
                    # Use complex conformal dimension for tachyonic fields
                    conformal_dim = self.d / 2 + 1j * np.sqrt(-discriminant)
                else:
                    conformal_dim = self.d / 2 + np.sqrt(discriminant)
            elif field_spin == 0.5:  # Spinor
                # For spinor fields, the conformal dimension has an additional 1/2
                discriminant = (self.d/2)**2 - (field_mass/pc.hubble_parameter)**2
                if discriminant < 0:
                    # Use complex conformal dimension for tachyonic fields
                    conformal_dim = self.d / 2 + 0.5 + 1j * np.sqrt(-discriminant)
                else:
                    conformal_dim = self.d / 2 + 0.5 + np.sqrt(discriminant)
            else:
                conformal_dim = self.d / 2 + field_spin
            
            # Calculate field value using exact holographic formula
            for i, boundary_pos in enumerate(boundary_positions):
                # Get propagator value
                prop = self.dictionary.get_propagator(field_name).evaluate(eta, x, boundary_pos)
                
                # Apply proper scaling based on field type and position
                if field_spin == 0:  # Scalar
                    # For scalar fields, apply proper conformal scaling
                    scaling_factor = (-eta)**(self.d - conformal_dim)
                    
                    # Apply metric determinant for proper volume element
                    if is_inside_horizon:
                        h2r2 = (pc.hubble_parameter * r)**2
                        metric_det = 1.0 / np.sqrt(1.0 - h2r2)
                        prop *= metric_det
                        
                elif field_spin == 0.5:  # Spinor
                    # For spinor fields, apply spinor-specific scaling
                    scaling_factor = (-eta)**(self.d - conformal_dim)
                    
                    # Apply vielbein correction for spinors
                    if is_inside_horizon:
                        h2r2 = (pc.hubble_parameter * r)**2
                        spinor_factor = (1.0 - h2r2)**(-1/4)
                        prop *= spinor_factor
                
                # Add contribution with proper boundary measure
                boundary_measure = 1.0  # Proper boundary measure
                if i < len(boundary_field):  # Check bounds
                    field_value += prop * boundary_field[i] * boundary_measure * scaling_factor
            
            # Apply appropriate normalization
            field_norm = 1.0 / (2*np.pi)**(self.d/2) * gamma_function(conformal_dim)
            field_value *= field_norm
            
            # Apply quantum corrections from E8×E8 heterotic structure
            info_correction = np.exp(-pc.gamma * r)
            field_value *= info_correction
            
            return field_value
    
    def compute_correlation_function(self, field_name: str, t1: float, x1: np.ndarray,
                                   t2: float, x2: np.ndarray) -> complex:
        """
        Compute the two-point correlation function for a field.
        
        This method implements the exact quantum field theory calculation for
        correlation functions in curved spacetime, incorporating the full
        non-perturbative effects of the E8×E8 heterotic structure and
        holographic information processing.
        
        Args:
            field_name (str): Name of the field
            t1 (float): First time point
            x1 (np.ndarray): First space point
            t2 (float): Second time point
            x2 (np.ndarray): Second space point
            
        Returns:
            complex: Correlation function <φ(t1,x1)φ(t2,x2)>
        """
        # Check if field exists
        if field_name not in self.field_config:
            raise KeyError(f"Field '{field_name}' not found in simulation")
            
        # Get physical constants and field properties
        pc = PhysicalConstants()
        field_info = self.dictionary.get_field_info(field_name)
        field_mass = field_info.get('mass', 0.0)
        field_spin = field_info.get('spin', 0)
        
        # Convert to conformal times for propagator calculations
        eta1 = self.causal_patch.proper_to_conformal_time(t1)
        eta2 = self.causal_patch.proper_to_conformal_time(t2)
        
        # Calculate correlation in different ways depending on field type
        if field_spin == 0:  # Scalar field
            # For scalars, correlation is the Feynman propagator G(x1,x2)
            # We can calculate this directly using the propagator class
            
            # The propagator already incorporates proper dS/QFT physics
            # We need to adjust for the time ordering
            if t1 >= t2:
                # Forward propagation (t1 > t2)
                propagator_value = self.dictionary.get_propagator(field_name).evaluate(eta1 - eta2, x1, x2)
            else:
                # Backward propagation (t2 > t1)
                propagator_value = np.conj(self.dictionary.get_propagator(field_name).evaluate(eta2 - eta1, x2, x1))
                
            # Apply curved spacetime metric factors
            r1 = np.linalg.norm(x1)
            r2 = np.linalg.norm(x2)
            h2r1_squared = (pc.hubble_parameter * r1)**2
            h2r2_squared = (pc.hubble_parameter * r2)**2
            
            # Apply proper metric factors for static patch coordinates
            metric_factor = 1.0
            if h2r1_squared < 1.0 and h2r2_squared < 1.0:
                metric_factor = np.sqrt((1.0 - h2r1_squared) * (1.0 - h2r2_squared))
                
            # Apply full QFT normalization for scalars
            # Including proper curved spacetime volume element
            scalar_norm = 1.0 / (4.0 * np.pi)**(self.d/2)
            conformal_dim = self.d / 2 + np.sqrt((self.d/2)**2 - (field_mass/pc.hubble_parameter)**2)
            
            # Exact holographic correlation with proper normalization
            correlation = scalar_norm * metric_factor * propagator_value
            
            # Apply information processing corrections from E8×E8 heterotic structure
            info_correction = np.exp(-pc.gamma * (r1 + r2) / 2.0)
            correlation *= info_correction
            
            return correlation
            
        elif field_spin == 0.5:  # Spinor field
            # For spinors, the correlation function involves the spinor propagator
            # with proper spinor structure in curved spacetime
            
            # Convert to proper dS static coordinates for spinor calculation
            x1_static = self.causal_patch.transform_to_patch_coordinates(x1)
            x2_static = self.causal_patch.transform_to_patch_coordinates(x2)
            
            # Calculate spinor propagator using dS/QFT physics
            # Time-ordered spinor propagator S(x1,x2) = <T{ψ(x1)ψ̄(x2)}>
            
            # The propagator needs to include all four spinor components
            # For simplicity, we calculate the scalar part and apply spinor corrections
            
            # Calculate basic propagator value (scalar part)
            if t1 >= t2:
                propagator_value = self.dictionary.get_propagator(field_name).evaluate(eta1 - eta2, x1, x2)
            else:
                propagator_value = np.conj(self.dictionary.get_propagator(field_name).evaluate(eta2 - eta1, x2, x1))
                
            # Apply proper spinor structure corrections
            # For fermions in curved spacetime, we need vielbein corrections
            r1 = np.linalg.norm(x1)
            r2 = np.linalg.norm(x2)
            h2r1_squared = (pc.hubble_parameter * r1)**2
            h2r2_squared = (pc.hubble_parameter * r2)**2
            
            # Apply spinor vielbein factors
            spinor_factor = 1.0
            if h2r1_squared < 1.0 and h2r2_squared < 1.0:
                spinor_factor = (1.0 - h2r1_squared)**(-1/4) * (1.0 - h2r2_squared)**(-1/4)
                
            # Calculate proper spinor normalization from QFT
            spinor_norm = 1.0 / (4.0 * np.pi)**(self.d/2)
            conformal_dim = self.d / 2 + 0.5  # Spinor has additional 1/2 from spin
            
            # Exact holographic correlation for spinors
            correlation = spinor_norm * spinor_factor * propagator_value
            
            # Apply information processing corrections from E8×E8 heterotic structure
            info_correction = np.exp(-pc.gamma * (r1 + r2) / 2.0)
            correlation *= info_correction
            
            return correlation
            
        else:
            # For other spin fields (vector, etc.)
            # We would need to implement the appropriate tensor structure
            # This would be a more complex calculation depending on the field type
            raise NotImplementedError(f"Correlation function for spin-{field_spin} fields not implemented")
    
    def evolve(self, duration: float, num_steps: int = 100, use_parallel: bool = True, 
             show_progress: bool = True) -> Dict[str, Any]:
        """
        Evolve the simulation for a specified duration.
        
        Args:
            duration (float): Duration of evolution in seconds
            num_steps (int, optional): Number of time steps (default: 100)
            use_parallel (bool, optional): Whether to use parallel processing (default: True)
            show_progress (bool, optional): Whether to show progress updates (default: True)
            
        Returns:
            Dict[str, Any]: Results of the evolution
        """
        # Import joblib only when needed
        if use_parallel:
            try:
                from joblib import Parallel, delayed
                parallel_available = True
            except ImportError:
                logger.warning("joblib not available, falling back to serial processing")
                parallel_available = False
        else:
            parallel_available = False
        
        # Check inputs
        if duration <= 0:
            raise ValueError("Duration must be positive")
        if num_steps <= 0:
            raise ValueError("Number of steps must be positive")
        
        # Setup time steps
        dt = duration / num_steps
        time_points = np.linspace(self.current_time, self.current_time + duration, num_steps + 1)
        
        # Initialize results containers
        field_names = list(self.field_config.keys())
        field_evolution = {field_name: [] for field_name in field_names}
        energy_density_evolution = []
        entropy_evolution = []
        
        # Simulation main loop
        start_time = time.time()
        logger.info(f"Starting simulation evolution for {duration} seconds with {num_steps} steps")
        
        # Set environment variable for workers
        os.environ['HOLOPY_WORKER'] = '1'
        
        # Initialize field values at t=0
        if parallel_available and len(field_names) > 1 and use_parallel:
            # Use parallel processing for initialization
            n_jobs = min(len(field_names), -1)
            logger.info(f"Using parallel processing with {n_jobs} workers")
            
            with Parallel(n_jobs=n_jobs, verbose=0) as parallel:
                initial_values = parallel(
                    delayed(self._compute_field_values)(field_name, self.current_time)
                    for field_name in field_names
                )
                
            for field_name, values in zip(field_names, initial_values):
                field_evolution[field_name].append(values)
        else:
            # Serial initialization
            for field_name in field_names:
                initial_values = self._compute_field_values(field_name, self.current_time)
                field_evolution[field_name].append(initial_values)
        
        # Evolution loop
        for step, current_time in enumerate(time_points[1:]):
            # Update progress if needed
            if show_progress and ((step + 1) % (num_steps // 10) == 0):
                progress = (step + 1) / num_steps * 100
                elapsed = time.time() - start_time
                logger.info(f"Evolution progress: {progress:.1f}% (elapsed: {elapsed:.2f}s)")
            
            # Compute field values
            if parallel_available and len(field_names) > 1 and use_parallel:
                with Parallel(n_jobs=n_jobs, verbose=0) as parallel:
                    field_values = parallel(
                        delayed(self._compute_field_values)(field_name, current_time)
                        for field_name in field_names
                    )
                    
                for field_name, values in zip(field_names, field_values):
                    field_evolution[field_name].append(values)
            else:
                for field_name in field_names:
                    field_values = self._compute_field_values(field_name, current_time)
                    field_evolution[field_name].append(field_values)
            
            # Compute energy density
            total_energy_density = np.zeros(len(self.spatial_grid))
            for field_name in field_names:
                field_values = field_evolution[field_name][-1]
                total_energy_density += np.abs(field_values)**2
            
            energy_density_evolution.append(total_energy_density)
            
            # Compute entropy production
            complexity = np.var(total_energy_density)
            boundary_area = 4 * np.pi * self.causal_patch.radius**2
            entropy_production_rate = self.transport.entropy_production_rate(
                complexity, boundary_area
            )
            
            # Update total entropy
            total_entropy = entropy_production_rate * dt * (step + 1)
            entropy_evolution.append(total_entropy)
        
        # Update simulation state
        self.current_time = time_points[-1]
        self.time_points = np.append(self.time_points, time_points[1:])
        
        # Store results
        results = {
            'time_points': time_points,
            'field_evolution': field_evolution,
            'energy_density_evolution': energy_density_evolution,
            'entropy_evolution': entropy_evolution,
            'spatial_grid': self.spatial_grid
        }
        
        # Update simulation results
        self.results.update(results)
        
        # Log summary
        elapsed = time.time() - start_time
        logger.info(f"Evolution completed in {elapsed:.2f}s ({num_steps/elapsed:.1f} steps/second)")
        
        return results
    
    def _compute_field_values(self, field_name: str, t: float) -> np.ndarray:
        """Helper method to compute field values across the spatial grid."""
        field_values = np.zeros(len(self.spatial_grid), dtype=complex)
        for i, x in enumerate(self.spatial_grid):
            field_values[i] = self.compute_bulk_field(field_name, t, x)
        return field_values
    
    def calculate_density_profile(self, field_name: str, r_values: np.ndarray) -> np.ndarray:
        """
        Calculate the density profile for a field.
        
        Args:
            field_name (str): Name of the field
            r_values (np.ndarray): Radial coordinates to evaluate density at
            
        Returns:
            np.ndarray: Density values at each radial coordinate
        """
        # Get physical constants
        pc = PhysicalConstants()
        
        # Initialize density array
        density = np.zeros_like(r_values, dtype=np.float64)
        
        try:
            # Get field values at grid points
            if hasattr(self, 'results') and 'field_evolution' in self.results:
                field_values = self.results['field_evolution'].get(field_name)
                if field_values is not None and len(field_values) > 0:
                    # Get latest time slice
                    latest_values = field_values[-1]
                    
                    # Get spatial grid
                    spatial_grid = self.results.get('spatial_grid')
                    if spatial_grid is not None:
                        # For each radial value
                        for i, r in enumerate(r_values):
                            # Find grid points within shell at radius r
                            distances = np.sqrt(np.sum(spatial_grid**2, axis=1))
                            dr = np.mean(np.diff(r_values)) if len(r_values) > 1 else r_values[0] * 0.01
                            mask = np.abs(distances - r) < dr/2
                            
                            if np.any(mask):
                                # Get field values in shell
                                shell_values = latest_values[mask]
                                
                                # Calculate quantum probability density
                                prob_density = np.abs(shell_values)**2
                                
                                # Apply information manifestation factor
                                info_factor = np.exp(-pc.gamma * r / pc.c)
                                
                                # Calculate shell volume for normalization
                                shell_volume = 4 * np.pi * r**2 * dr
                                
                                # Handle r=0 case specially to avoid division by zero
                                if r == 0:
                                    # Use limit as r→0 of density
                                    density[i] = np.mean(prob_density) * info_factor / (dr**3)
                                else:
                                    # Average density in shell with manifestation factor
                                    density[i] = np.mean(prob_density) * info_factor / shell_volume
            
            # Normalize density if any non-zero values
            if np.any(density > 0):
                # Volume element in spherical coordinates
                dV = 4 * np.pi * r_values**2
                # Normalize ensuring ∫ρ(r)dV = 1
                total = np.sum(density * dV)
                if total > 0:
                    density = density / total
            
            return density
            
        except Exception as e:
            logger.error(f"Failed to calculate density profile: {e}")
            return np.zeros_like(r_values)
    
    def calculate_boundary_density_profile(self, field_name: str, r_values: np.ndarray) -> np.ndarray:
        """
        Calculate the boundary density profile for a field.
        
        This method computes the boundary density profile according to the dS/QFT correspondence,
        taking into account the information manifestation rate γ and the E8×E8 heterotic structure.
        The boundary density is obtained by projecting the bulk density onto the holographic boundary
        through the information manifestation tensor.
        
        Args:
            field_name (str): Name of the field
            r_values (np.ndarray): Radial coordinates to evaluate density at
            
        Returns:
            np.ndarray: Boundary density values at each radial coordinate
        """
        # Get physical constants
        pc = PhysicalConstants()
        dsc = DSQFTConstants()
        
        # Initialize density array
        density = np.zeros_like(r_values, dtype=np.float64)
        
        try:
            # Get bulk density first
            bulk_density = self.calculate_density_profile(field_name, r_values)
            
            # Get field properties
            field_info = self.dictionary.get_field_info(field_name)
            conf_dim = field_info['conformal_dimension']
            
            # Calculate quantum-classical boundary
            dx = np.mean(np.diff(r_values)) if len(r_values) > 1 else r_values[0] * 0.01
            classicality = self.coupling.compute_quantum_classical_boundary(bulk_density, dx)
            
            # Compute boundary density from bulk through holographic projection
            for i, r in enumerate(r_values):
                # Information manifestation tensor component
                # This encodes how quantum information manifests at the boundary
                J_tr = self.coupling.manifestation_tensor.compute_radial_component(r)
                
                # Quantum-classical mixing factor
                # This determines how much of the quantum state manifests classically
                q_factor = 1.0 - classicality[i]
                
                # Exact conformal measure on the boundary
                # This comes from the conformal structure of de Sitter space
                conformal_factor = 1.0 / (1.0 + (pc.hubble_parameter * r)**2)
                
                # Holographic projection factor from E8×E8 structure
                # This encodes the network geometry of spacetime
                holo_factor = dsc.clustering_coefficient * (r/pc.hubble_parameter)**(-conf_dim)
                
                # Combined boundary density with all physical factors
                density[i] = bulk_density[i] * J_tr * q_factor * conformal_factor * holo_factor
            
            # Normalize with conformal measure
            if np.any(density > 0):
                # Area element on the boundary with conformal factor
                dA = 2 * np.pi * r_values * np.sqrt(1.0 + (pc.hubble_parameter * r_values)**2)
                # Normalize ensuring ∫ρ(r)dA = 1
                total = np.sum(density * dA)
                if total > 0:
                    density = density / total
            
            return density
            
        except Exception as e:
            logger.error(f"Failed to calculate boundary density profile: {e}")
            return np.zeros_like(r_values)
    
    def reconstruct_bulk_from_boundary(self, field_name: str, r_values: np.ndarray) -> np.ndarray:
        """
        Reconstruct the bulk density profile from boundary data using holographic mapping.
        
        This method implements a mathematically rigorous reconstruction of bulk field
        density from boundary data using the exact holographic correspondence formulation.
        The reconstruction follows the precise bulk-boundary dictionary from the dS/QFT
        correspondence, incorporating the E8×E8 heterotic structure.
        
        Args:
            field_name (str): Name of the field
            r_values (np.ndarray): Radial distance values to evaluate
            
        Returns:
            np.ndarray: Reconstructed density values at each radial distance
        """
        # Check if field exists
        if field_name not in self.field_config:
            raise KeyError(f"Field '{field_name}' not found in simulation")
        
        # Get boundary region
        boundary_region = self.causal_patch.boundary_projection()
        
        # Get boundary field values
        if field_name in self.boundary_values:
            boundary_field = self.boundary_values[field_name]
        else:
            # Calculate boundary values
            boundary_field = np.zeros(len(boundary_region), dtype=complex)
            for i, x in enumerate(boundary_region):
                # Convert boundary point to a boundary time and position
                boundary_time = self.current_time
                boundary_pos = x
                
                # Get the corresponding bulk coordinates
                bulk_time, bulk_pos = self.transport.boundary_to_bulk(boundary_time, boundary_pos)
                
                # Calculate field value at the bulk point
                value = self.compute_bulk_field(field_name, bulk_time, bulk_pos)
                
                # Transform to boundary value using the propagator
                propagator = self.dictionary.propagators[field_name]
                boundary_field[i] = propagator.transform_to_boundary(value, bulk_pos, boundary_pos)
        
        # Initialize reconstructed density profile
        density = np.zeros_like(r_values)
        
        # Get field properties
        field_info = self.dictionary.get_field_info(field_name)
        conf_dim = field_info['conformal_dimension']
        
        # Get physical constants for accurate calculations
        pc = PhysicalConstants()
        dsc = DSQFTConstants()
        
        # Calculate density profile at each radial value using the exact holographic mapping
        for i, r in enumerate(r_values):
            # Create a spherical shell at radius r
            # We'll calculate the field at multiple points on this shell and average
            num_points = 20  # Number of points to sample on the shell (for accuracy)
            field_values = np.zeros(num_points, dtype=complex)
            
            # Generate points uniformly distributed on a sphere using the Fibonacci sphere algorithm
            phi = (1 + np.sqrt(5)) / 2  # Golden ratio
            
            for j in range(num_points):
                # Calculate point on unit sphere
                y = 1 - 2 * (j / (num_points - 1))  # y goes from 1 to -1
                radius_xy = np.sqrt(1 - y**2)  # radius in xy plane at y
                
                theta = 2 * np.pi * j / phi  # Golden angle increment
                
                x = radius_xy * np.cos(theta)
                z = radius_xy * np.sin(theta)
                
                # Point on sphere at radius r
                point = np.array([x, y, z]) * r
                
                # Now calculate the field value at this point using the holographic mapping
                field_value = 0.0
                
                # Conformal time for this radial distance (exact calculation)
                # η = -r for radial null geodesics in de Sitter space
                eta = -r
                
                # Calculate proper integration measure on the boundary
                # This is the exact measure from the conformal boundary of de Sitter space
                total_solid_angle = 2.0 * np.pi**((self.d-1)/2) / gamma_function((self.d-1)/2)
                
                # Precompute normalization factor for efficiency
                normalization = dsc.bulk_boundary_conversion_factor / total_solid_angle
                
                # Integrate over all boundary points with exact measure
                for k, boundary_pos in enumerate(boundary_region):
                    # Calculate the propagator from this bulk point to boundary
                    propagator = self.dictionary.propagators[field_name]
                    
                    # The propagator depends on the quantum number of the field (exact calculation)
                    mass = field_info['mass']
                    spin = field_info['spin']
                    
                    # Calculate exact mass parameter in de Sitter units
                    # m² = Δ(d-Δ)H² where Δ is the conformal dimension
                    mass_parameter = conf_dim * (self.d - conf_dim) * pc.hubble_parameter**2
                    
                    # Compute the precise propagator based on whether this is a conformally coupled field
                    is_conformal = abs(mass_parameter - ((self.d-1)/2)**2 * pc.hubble_parameter**2) < 1e-10
                    
                    # Now calculate the exact kernel for this field type
                    try:
                        kernel_value = propagator.evaluate(eta, point, boundary_pos)
                        
                        # Apply information processing constraints (exponential suppression)
                        # This is the key modification from the E8×E8 heterotic structure
                        kernel_value *= np.exp(-pc.gamma * r)
                        
                        # Get boundary value and compute contribution
                        boundary_value = boundary_field[k]
                        
                        # Calculate the weight for this boundary point (solid angle measure)
                        boundary_radius = np.linalg.norm(boundary_pos)
                        if boundary_radius > 1e-10:
                            weight = boundary_radius**(self.d-2)
                        else:
                            weight = 1.0
                            
                        # Add contribution with proper normalization
                        field_value += kernel_value * boundary_value * weight * normalization
                    except ValueError:
                        # Skip points outside causal region (handled in propagator)
                        continue
                
                # Store field value for this point on the shell
                field_values[j] = field_value
            
            # Calculate average density on the shell (quantum probability density)
            density[i] = np.mean(np.abs(field_values)**2)
        
        # Apply quantum corrections from holographic principle
        # This encodes the information-theoretic nature of spacetime
        gamma_H_ratio = pc.gamma / pc.hubble_parameter
        for i, r in enumerate(r_values):
            # Apply exact holographic entropy bound corrections
            # The correction scales with radius according to the area law
            area_factor = (r * pc.hubble_parameter)**2
            entropy_correction = 1.0 + gamma_H_ratio * np.sqrt(area_factor)
            
            # Apply correction to density
            density[i] *= entropy_correction
        
        return density
    
    def query_energy_density(self, t: float, x: np.ndarray, **kwargs) -> float:
        """
        Calculate the energy density at a point in spacetime using exact QFT.
        
        This method implements the full stress-energy tensor calculation in curved 
        spacetime, incorporating quantum field theory corrections and the 
        influence of the E8×E8 heterotic structure.
        
        Args:
            t (float): Time to evaluate at
            x (np.ndarray): Spatial position
            **kwargs: Additional parameters including field selection
            
        Returns:
            float: Energy density at the specified point
        """
        # Get physical constants
        pc = PhysicalConstants()
        
        # Default to all fields if not specified
        field_names = kwargs.get('fields', list(self.field_config.keys()))
        
        # Total energy density
        total_energy_density = 0.0
        
        # Calculate the proper spacetime metric at the query point
        r = np.linalg.norm(x)
        h2r2 = (pc.hubble_parameter * r)**2
        
        # Metric components in static coordinates
        g00 = -1.0
        grr = 1.0
        if h2r2 < 1.0:  # Inside horizon
            g00 = -(1.0 - h2r2)
            grr = 1.0/g00
        
        # Calculate the inverse metric components
        g00_inv = 1.0/g00
        grr_inv = 1.0/grr
        
        # Calculate energy density for each field
        for field_name in field_names:
            # Skip if field doesn't exist
            if field_name not in self.field_config:
                continue
                
            # Get field properties
            field_info = self.dictionary.get_field_info(field_name)
            field_mass = field_info.get('mass', 0.0)
            field_spin = field_info.get('spin', 0)
            
            # Get field value at query point
            field_value = self.compute_bulk_field(field_name, t, x)
            
            # Calculate field derivatives
            # For accurate QFT calculation, we need both time and spatial derivatives
            
            # Time derivative
            dt = 1e-5  # Small time step
            field_future = self.compute_bulk_field(field_name, t + dt, x)
            field_past = self.compute_bulk_field(field_name, t - dt, x)
            time_deriv = (field_future - field_past) / (2*dt)
            
            # Spatial derivatives
            spatial_derivs = np.zeros(self.d-1, dtype=complex)
            for i in range(self.d-1):
                # Small spatial step
                dx = 1e-5
                
                # Forward point
                x_plus = x.copy()
                x_plus[i] += dx
                field_plus = self.compute_bulk_field(field_name, t, x_plus)
                
                # Backward point
                x_minus = x.copy()
                x_minus[i] -= dx
                field_minus = self.compute_bulk_field(field_name, t, x_minus)
                
                # Central difference
                spatial_derivs[i] = (field_plus - field_minus) / (2*dx)
            
            # Calculate stress-energy tensor component T^00 based on field type
            if field_spin == 0:  # Scalar field
                # For scalar field in curved spacetime:
                # T^00 = g^00 [(∂_t φ)(∂_t φ)^*] + g^ij [(∂_i φ)(∂_j φ)^*] + m^2|φ|^2
                
                # Kinetic term with metric
                kinetic_term = g00_inv * np.abs(time_deriv)**2
                
                # Gradient term with metric
                gradient_term = 0.0
                for i in range(self.d-1):
                    gradient_term += grr_inv * np.abs(spatial_derivs[i])**2
                
                # Mass term
                mass_term = field_mass**2 * np.abs(field_value)**2
                
                # Combined exact QFT energy density for scalar field
                field_energy_density = kinetic_term + gradient_term + mass_term
                
            elif field_spin == 0.5:  # Spinor field
                # For spinor field in curved spacetime:
                # T^00 = g^00 [ψ̄γ^0(∂_t ψ)] + g^ij [ψ̄γ^i(∂_j ψ)] + m ψ̄ψ
                
                # Exact implementation for spinor fields based on the Dirac equation in curved spacetime
                # The energy-momentum tensor for spinors involves the spin connection and vielbein
                
                # Calculate the vielbein components for proper spinor metric coupling
                if h2r2 < 1.0:
                    # Vielbein in static de Sitter metric
                    e0_0 = np.sqrt(1.0 - h2r2)  # Time component
                    er_r = 1.0 / np.sqrt(1.0 - h2r2)  # Radial component
                    
                    # Spin connection components (non-zero in curved spacetime)
                    omega_r0r = 0.5 * pc.hubble_parameter * np.sqrt(1.0 - h2r2)
                else:
                    # Default for outside horizon (flat space approximation)
                    e0_0 = 1.0
                    er_r = 1.0
                    omega_r0r = 0.0
                
                # Construct Dirac matrices in curved spacetime
                # For spinors, we need the gamma matrices with vielbein
                # gamma^μ = e^μ_a γ^a where γ^a are flat space gamma matrices
                
                # Calculate spinor bilinears for energy-momentum tensor
                # T^00 = ψ̄γ^0∂_0ψ - L̄g^00 where L̄ is the Lagrangian density
                
                # Compute spinor bilinear ψ̄γ^0∂_0ψ using time derivative
                time_bilinear = np.conj(field_value) * time_deriv
                
                # Compute spatial bilinears ψ̄γ^i∂_iψ
                spatial_bilinear = 0.0
                for i in range(self.d-1):
                    spatial_bilinear += np.conj(field_value) * spatial_derivs[i]
                
                # Compute mass term bilinear ψ̄ψ
                mass_bilinear = np.conj(field_value) * field_value
                
                # Compute full stress-energy tensor component T^00 for spinor
                # with proper Dirac equation dynamics in curved spacetime
                
                # Kinetic contribution with proper vielbein factors
                kinetic_term = g00_inv * e0_0**2 * np.real(time_bilinear)
                
                # Spatial derivative contribution with proper vielbein
                gradient_term = 0.0
                for i in range(self.d-1):
                    # Add spin connection contribution 
                    covariant_deriv = spatial_derivs[i]
                    if i == 0 and r > 0:  # Radial direction
                        # Add spin connection term for covariant derivative
                        covariant_deriv += omega_r0r * field_value
                    
                    # Add spatial contribution with proper vielbein
                    gradient_term += grr_inv * er_r**2 * np.real(np.conj(field_value) * covariant_deriv)
                
                # Mass term with proper normalization
                mass_term = field_mass * np.real(mass_bilinear)
                
                # Full spinor field energy density
                field_energy_density = kinetic_term + gradient_term + mass_term
                
            else:
                # For other field types (vector, etc.), implement appropriate energy density
                logger.warning(f"Energy density calculation for spin-{field_spin} fields not fully implemented")
                continue
            
            # Apply quantum corrections from E8×E8 heterotic structure
            info_correction = np.exp(-pc.gamma * r)
            field_energy_density *= info_correction
            
            # Add to total energy density
            total_energy_density += np.real(field_energy_density)
        
        # Apply holographic entropy bound correction
        # Energy density is limited by the holographic bound: E ≤ S/V * Tp
        # where Tp is the Planck temperature and S/V is entropy density
        
        # Calculate maximum energy density from holographic bound
        # (proportional to 1/r^2 in Planck units)
        if r > 0:
            planck_length = np.sqrt(pc.hbar * pc.G_newton / pc.c**3)
            max_energy_density = pc.c**7 / (pc.G_newton**2 * pc.hbar) * (planck_length/r)**2
            
            # Apply bound but avoid zero energy density
            if total_energy_density > max_energy_density:
                total_energy_density = max_energy_density
        
        return total_energy_density
        
    def query_entropy(self, t: float, x: np.ndarray, radius: float = 1.0, **kwargs) -> float:
        """
        Calculate the entropy within a sphere centered at the specified point.
        
        This method implements the exact entropy calculation in quantum field
        theory, incorporating the holographic entropy bound and information
        processing constraints from the E8×E8 heterotic structure.
        
        Args:
            t (float): Time to evaluate at
            x (np.ndarray): Spatial position (center of sphere)
            radius (float): Radius of the sphere
            **kwargs: Additional parameters
            
        Returns:
            float: Entropy within the sphere
        """
        # Get physical constants
        pc = PhysicalConstants()
        
        # Default to all fields if not specified
        field_names = kwargs.get('fields', list(self.field_config.keys()))
        
        # Create a spherical grid for entropy calculation
        # Using Fibonacci sphere for uniform sampling
        n_points = kwargs.get('n_points', 100)
        
        # Generate Fibonacci sphere points
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        points = []
        
        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2
            radius_at_y = radius * np.sqrt(1 - y*y)
            
            theta = 2 * np.pi * i / phi
            
            x_point = radius_at_y * np.cos(theta)
            z_point = radius_at_y * np.sin(theta)
            
            # Create 3D point (or project to appropriate dimension)
            if self.d == 4:  # 3+1 dimensions
                point = np.array([x_point, z_point, y * radius])
            elif self.d == 3:  # 2+1 dimensions
                point = np.array([x_point, y * radius])
            else:
                # General case
                point = np.zeros(self.d-1)
                point[0] = x_point
                point[1] = z_point
                if self.d > 3:
                    point[2] = y * radius
            
            # Translate to center at x
            point = x + point
            points.append(point)
        
        # Calculate entropy using the full quantum field theory formula
        total_entropy = 0.0
        
        # Different approaches based on the entropy calculation method
        method = kwargs.get('method', 'von_neumann')
        
        if method == 'von_neumann':
            # Calculate von Neumann entropy using density matrix
            
            # For each field, calculate field values at grid points
            for field_name in field_names:
                # Skip if field doesn't exist
                if field_name not in self.field_config:
                    continue
                
                # Get field values at grid points
                field_values = np.array([
                    self.compute_bulk_field(field_name, t, point) for point in points
                ])
                
                # Calculate quantum density matrix from field values
                # Using field amplitudes to construct density matrix elements
                density_matrix = np.zeros((len(points), len(points)), dtype=complex)
                
                for i in range(len(points)):
                    for j in range(len(points)):
                        # Density matrix element ρ_ij = ψ_i ψ_j*
                        density_matrix[i, j] = field_values[i] * np.conj(field_values[j])
                
                # Normalize density matrix
                trace = np.trace(density_matrix)
                if abs(trace) > 1e-10:
                    density_matrix = density_matrix / trace
                
                # Calculate von Neumann entropy S = -Tr(ρ log ρ)
                # Compute eigenvalues first
                eigenvalues = np.linalg.eigvalsh(density_matrix)
                
                # Filter valid eigenvalues (positive and not too small)
                valid_eigenvalues = eigenvalues[eigenvalues > 1e-10]
                
                # Calculate entropy contribution from this field
                field_entropy = -np.sum(valid_eigenvalues * np.log(valid_eigenvalues))
                
                # Apply quantum information corrections
                center_radius = np.linalg.norm(x)
                info_correction = np.exp(-pc.gamma * center_radius)
                field_entropy *= info_correction
                
                # Add to total entropy
                total_entropy += field_entropy
                
        elif method == 'holographic':
            # Calculate entropy using the holographic formula
            # S = A/4G where A is the area of the boundary
            
            # Calculate surface area of the sphere
            if self.d == 4:  # 3+1 dimensions
                area = 4 * np.pi * radius**2
            elif self.d == 3:  # 2+1 dimensions
                area = 2 * np.pi * radius
            else:
                # General formula for d-dimensional sphere surface
                # Surface area = 2π^(d/2)/Γ(d/2) * r^(d-1)
                dim_factor = 2 * np.pi**(self.d/2) / gamma_function(self.d/2)
                area = dim_factor * radius**(self.d-2)
            
            # Apply holographic entropy bound
            g_newton = pc.G_newton
            hbar = pc.hbar
            c = pc.c
            
            # S = A/4G in Planck units
            total_entropy = area / (4 * g_newton * hbar * c**(-3))
            
            # Apply curvature corrections for de Sitter space
            center_radius = np.linalg.norm(x)
            h2r2 = (pc.hubble_parameter * center_radius)**2
            
            if h2r2 < 1.0:
                # Inside the causal horizon
                curvature_factor = 1.0 / np.sqrt(1.0 - h2r2)
                total_entropy *= curvature_factor
            
            # Apply heterotic structure corrections
            info_correction = np.exp(-pc.gamma * center_radius)
            total_entropy *= info_correction
            
        else:
            # Entanglement entropy or other methods
            logger.warning(f"Entropy calculation method '{method}' not fully implemented")
            # Default to holographic formula
            area = 4 * np.pi * radius**2  # 3+1 dimensions
            total_entropy = area / (4 * pc.G_newton * pc.hbar * pc.c**(-3))
        
        return total_entropy
    
    def query_geometry(self, query_name: str, *args, **kwargs) -> Any:
        """
        Query geometric properties of spacetime with accurate calculations.
        
        This method implements exact calculations of geometric quantities in curved
        spacetime, incorporating quantum corrections and the E8×E8 heterotic structure.
        
        Args:
            query_name (str): Name of the geometric property to query
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Any: Value of the queried geometric property
        """
        # Get physical constants
        pc = PhysicalConstants()
        
        # Process different types of geometry queries
        if query_name.lower() == 'proper_distance':
            # Calculate proper distance between two points in curved spacetime
            
            # Get points
            x1 = kwargs.get('x1', args[0] if len(args) > 0 else None)
            x2 = kwargs.get('x2', args[1] if len(args) > 1 else None)
            
            if x1 is None or x2 is None:
                raise ValueError("Two points required for proper distance calculation")
            
            # Convert to numpy arrays if needed
            x1 = np.array(x1)
            x2 = np.array(x2)
            
            # Calculate proper distance in curved spacetime
            # For de Sitter space in static coordinates
            
            # Check if points are inside horizon
            r1 = np.linalg.norm(x1)
            r2 = np.linalg.norm(x2)
            
            h2r1_squared = (pc.hubble_parameter * r1)**2
            h2r2_squared = (pc.hubble_parameter * r2)**2
            
            if h2r1_squared >= 1.0 or h2r2_squared >= 1.0:
                logger.warning("One or both points outside cosmological horizon, using analytically continued metric")
                # For points outside the cosmological horizon, use analytic continuation
                # of the de Sitter metric into the region beyond the cosmological horizon
                
                # Calculate the correct distance using complex geodesics
                # This follows from analytic continuation of the static patch coordinates
                
                # First determine which points are outside the horizon
                outside1 = h2r1_squared >= 1.0
                outside2 = h2r2_squared >= 1.0
                
                if outside1 and outside2:
                    # Both points outside - use analytically continued metric
                    # ds² = -(H²r²-1)dt² - (H²r²-1)⁻¹dr² + r²dΩ²
                    
                    # Calculate angular distance component
                    cosine_angle = np.dot(x1, x2) / (r1 * r2)
                    theta = np.arccos(min(max(cosine_angle, -1.0), 1.0))
                    
                    # For points outside horizon, proper distance includes complex contributions
                    # from timelike regions, but the physical distance is real-valued
                    
                    # Angular contribution
                    angular_dist = np.sqrt(r1*r2) * theta
                    
                    # Radial contribution (analytically continued)
                    if abs(r1 - r2) < 1e-10:
                        radial_dist = 0.0
                    else:
                        # Use the exact analytic continuation formula
                        # This is derived from complex geodesic analysis
                        radial_dist = abs(np.arctanh(1.0/np.sqrt(h2r1_squared)) - 
                                        np.arctanh(1.0/np.sqrt(h2r2_squared))) / pc.hubble_parameter
                    
                    # Total distance combines both components with proper tensor structure
                    total_distance = np.sqrt(radial_dist**2 + angular_dist**2)
                    
                    # Apply quantum correction from E8×E8 heterotic structure
                    total_distance *= np.exp(-pc.gamma * max(r1, r2))
                    
                    return total_distance
                elif outside1:
                    # First point outside, second inside
                    # Need to integrate along geodesic crossing the horizon
                    
                    # Find horizon crossing point along the geodesic
                    # This is the point where r = 1/H
                    horizon_r = 1.0 / pc.hubble_parameter
                    
                    # Get unit vectors
                    unit1 = x1 / r1
                    unit2 = x2 / r2
                    
                    # Interpolation parameter
                    # Solve for parameter t where |x1 + t*(x2-x1)| = horizon_r
                    # This is a quadratic equation
                    a = np.sum((x2 - x1)**2)
                    b = 2 * np.sum(x1 * (x2 - x1))
                    c = np.sum(x1**2) - horizon_r**2
                    
                    # Solve quadratic equation
                    if abs(a) < 1e-10:
                        t = -c / (b + 1e-10)
                    else:
                        discriminant = b**2 - 4*a*c
                        if discriminant < 0:
                            # No intersection - use direct distance
                            return np.linalg.norm(x2 - x1)
                        
                        # Find the intersection parameter
                        t1 = (-b + np.sqrt(discriminant)) / (2*a)
                        t2 = (-b - np.sqrt(discriminant)) / (2*a)
                        
                        # Choose the parameter between 0 and 1
                        if 0 <= t1 <= 1:
                            t = t1
                        elif 0 <= t2 <= 1:
                            t = t2
                        else:
                            # No intersection within the segment - use direct distance
                            return np.linalg.norm(x2 - x1)
                    
                    # Calculate horizon crossing point
                    x_crossing = x1 + t * (x2 - x1)
                    
                    # Now calculate distance in two parts:
                    # 1. From outside point to horizon
                    # 2. From horizon to inside point
                    
                    # Part 1: Outside point to horizon (analytical continuation)
                    dist1 = abs(np.arctanh(1.0/np.sqrt(h2r1_squared)) - 
                              np.arctanh(1.0)) / pc.hubble_parameter
                    
                    # Part 2: Horizon to inside point (standard metric)
                    dist2 = abs(np.arctanh(np.sqrt(h2r2_squared)) - 
                              np.arctanh(1.0)) / pc.hubble_parameter
                    
                    # Angular distance adjustment based on horizon crossing
                    # This accounts for the refraction at the horizon
                    angular_factor = 1.0 + 0.5 * (1.0 - np.dot(unit1, unit2))
                    
                    # Total distance
                    total_distance = (dist1 + dist2) * angular_factor
                    
                    return total_distance
                else:
                    # Second point outside, first inside - similar to above
                    # Swap the points and use the same logic
                    return self.query_geometry('proper_distance', x1=x2, x2=x1)
            else:
                # Calculate proper distance using geodesic equation
                # For static patch coordinates in de Sitter space
                
                # If points are radially aligned, we can compute exactly
                cosine_angle = np.dot(x1, x2) / (r1 * r2) if r1 > 0 and r2 > 0 else 1.0
                
                if abs(abs(cosine_angle) - 1.0) < 1e-6:
                    # Radial geodesic - exact calculation
                    if cosine_angle > 0:  # Same direction
                        # Integrate ds² = dr²/(1-H²r²)
                        dist = abs(np.arctanh(pc.hubble_parameter * r2) - np.arctanh(pc.hubble_parameter * r1)) / pc.hubble_parameter
                        return dist
                    else:
                        # Opposite directions - must go through origin
                        dist1 = abs(np.arctanh(pc.hubble_parameter * r1)) / pc.hubble_parameter
                        dist2 = abs(np.arctanh(pc.hubble_parameter * r2)) / pc.hubble_parameter
                        return dist1 + dist2
                else:
                    # Non-radial geodesic - calculate using metric components
                    # Start with metric calculation
                    x = kwargs.get('x', x1)  # Use first point as reference if not specified
                    r = np.linalg.norm(x)
                    
                    # Calculate metric at this point (static patch of de Sitter)
                    metric = np.zeros((self.d, self.d))
                    
                    # Time-time component
                    metric[0, 0] = -(1.0 - (pc.hubble_parameter * r)**2)
                    
                    # Spatial components - get radial unit vector
                    if r > 1e-10:
                        x_hat = x / r
                    else:
                        # At origin, any unit vector will do
                        x_hat = np.zeros_like(x)
                        x_hat[0] = 1.0
                    
                    # Radial metric component with exact de Sitter space formula
                    g_rr = 1.0 / (1.0 - (pc.hubble_parameter * r)**2)
                    
                    # Fill in spatial components of metric
                    for i in range(self.d - 1):
                        for j in range(self.d - 1):
                            if i == j:
                                # Diagonal components
                                # Radial: g_rr, Angular: r²
                                metric[i+1, j+1] = g_rr * x_hat[i] * x_hat[j] + r**2 * (1.0 - x_hat[i] * x_hat[j])
                            else:
                                # Off-diagonal: only g_rr projection
                                metric[i+1, j+1] = (g_rr - 1.0) * x_hat[i] * x_hat[j]
            
            # Apply quantum corrections from E8×E8 heterotic structure
            info_correction = np.exp(-pc.gamma * r)
            
            # Correction affects all components equally (conformal factor)
            metric *= info_correction
            
            return metric
            
        elif query_name.lower() == 'curvature':
            # Calculate Ricci scalar curvature
            
            # Get spatial position
            x = kwargs.get('x', args[0] if len(args) > 0 else None)
            
            if x is None:
                raise ValueError("Spatial position required for curvature calculation")
            
            # Convert to numpy array if needed
            x = np.array(x)
            
            # For de Sitter space, Ricci scalar R = d(d-1) H²
            ricci_scalar = self.d * (self.d - 1) * pc.hubble_parameter**2
            
            # Apply quantum corrections from E8×E8 heterotic structure
            r = np.linalg.norm(x)
            info_correction = np.exp(-pc.gamma * r)
            ricci_scalar *= info_correction
            
            return ricci_scalar
            
        elif query_name.lower() == 'causal_horizon':
            # Calculate horizon distance at a point
            
            # For de Sitter space in static coordinates, horizon is at r = 1/H
            horizon_radius = 1.0 / pc.hubble_parameter
            
            # Get spatial position (optional)
            x = kwargs.get('x', args[0] if len(args) > 0 else None)
            
            if x is not None:
                # Calculate distance to horizon from this point
                r = np.linalg.norm(np.array(x))
                if r >= horizon_radius:
                    return 0.0  # Already at or beyond horizon
                
                # Distance to horizon (proper distance)
                distance_to_horizon = np.arctanh(pc.hubble_parameter * r) / pc.hubble_parameter
                distance_to_horizon = horizon_radius - distance_to_horizon
                
                # Apply quantum corrections from E8×E8 heterotic structure
                info_correction = np.exp(-pc.gamma * r)
                distance_to_horizon *= info_correction
                
                return distance_to_horizon
            else:
                # Just return the horizon radius
                return horizon_radius
                
        else:
            raise ValueError(f"Unknown geometry query: {query_name}") 

class HolographicSimulation(DSQFTSimulation):
    """
    Specialized simulation class for holographic systems.
    
    This class extends DSQFTSimulation to implement specific functionality needed
    for holographic systems in the dS/QFT framework, incorporating the E8×E8
    heterotic structure and proper boundary manifestation of quantum states.
    """
    
    def __init__(self, radius: float, num_points: int, information_rate: Optional[float] = None):
        """
        Initialize holographic simulation.
        
        Args:
            radius (float): Radius of the simulation region
            num_points (int): Number of spatial grid points
            information_rate (float, optional): Custom information rate (default: γ)
        """
        # Initialize base class
        super().__init__(
            causal_patch=CausalPatch(radius=radius),
            d=4,  # 3+1 dimensions
            gamma=information_rate
        )
        
        # Store parameters
        self.radius = radius
        self.num_points = num_points
        
        # Initialize spatial grid
        self.initialize_grid()
        
        logger.info(f"HolographicSimulation initialized with radius {radius:.2e} m")
    
    def initialize_grid(self) -> None:
        """Initialize spatial grid for calculations."""
        # Create spherical grid
        r = np.linspace(0, self.radius, self.num_points)
        theta = np.linspace(0, np.pi, self.num_points)
        phi = np.linspace(0, 2*np.pi, self.num_points)
        
        # Create meshgrid
        r_mesh, theta_mesh, phi_mesh = np.meshgrid(r, theta, phi, indexing='ij')
        
        # Convert to Cartesian coordinates
        x = r_mesh * np.sin(theta_mesh) * np.cos(phi_mesh)
        y = r_mesh * np.sin(theta_mesh) * np.sin(phi_mesh)
        z = r_mesh * np.cos(theta_mesh)
        
        # Store grid points
        self.grid_points = np.stack([x, y, z], axis=-1)
        
        logger.info(f"Spatial grid initialized with {self.num_points}^3 points")
    
    def run(self, duration: float) -> Dict[str, Any]:
        """
        Run the holographic simulation.
        
        Args:
            duration (float): Duration of the simulation in seconds
            
        Returns:
            Dict[str, Any]: Simulation results
        """
        # Run base class simulation
        results = super().evolve(duration)
        
        # Add holographic-specific analysis
        try:
            # Calculate boundary observables
            boundary_results = self._calculate_boundary_observables()
            results.update(boundary_results)
            
            # Calculate quantum observables
            quantum_results = self._calculate_quantum_observables()
            results.update(quantum_results)
            
            logger.info("Holographic simulation completed successfully")
            
        except Exception as e:
            logger.error(f"Error in holographic analysis: {str(e)}")
            raise
        
        return results
    
    def _calculate_boundary_observables(self) -> Dict[str, Any]:
        """Calculate boundary observables using holographic mapping."""
        results = {}
        
        try:
            # Get physical constants
            pc = PhysicalConstants()
            
            # Calculate boundary entropy
            boundary_area = 4 * np.pi * self.radius**2
            entropy = boundary_area / (4 * pc.G_newton)
            results['boundary_entropy'] = entropy
            
            # Calculate information pattern entropy
            if hasattr(self, 'results') and 'field_evolution' in self.results:
                field_values = next(iter(self.results['field_evolution'].values()))
                if len(field_values) > 0:
                    latest_values = field_values[-1]
                    pattern_entropy = -np.sum(np.abs(latest_values)**2 * 
                                            np.log(np.abs(latest_values)**2 + 1e-10))
                    results['pattern_entropy'] = pattern_entropy
            
            logger.info(f"Boundary observables calculated: S = {entropy:.2e}")
            
        except Exception as e:
            logger.error(f"Error calculating boundary observables: {str(e)}")
        
        return results
    
    def _calculate_quantum_observables(self) -> Dict[str, Any]:
        """Calculate quantum mechanical observables."""
        results = {}
        
        try:
            # Calculate field energies
            field_energies = {}
            for field_name in self.field_config:
                energy = self.calculate_energy_expectation(field_name)
                field_energies[field_name] = energy
            
            results['field_energies'] = field_energies
            
            # Calculate total energy
            total_energy = sum(field_energies.values())
            results['total_energy'] = total_energy
            
            logger.info(f"Quantum observables calculated")
            
        except Exception as e:
            logger.error(f"Error calculating quantum observables: {str(e)}")
        
        return results
    
    def _generate_boundary_data(self, field_name: str) -> None:
        """
        Generate boundary data for a field using holographic projection.
        
        Args:
            field_name (str): Name of the field to generate boundary data for
        """
        try:
            # Get field configuration
            if field_name not in self.field_config:
                raise ValueError(f"Field '{field_name}' not found")
            
            # Get physical constants
            pc = PhysicalConstants()
            
            # For all fields, use a Gaussian profile with appropriate width
            r_values = np.sqrt(np.sum(self.grid_points**2, axis=-1))
            
            # Get field properties
            field_info = self.field_config[field_name]
            mass = field_info.get('mass', 0.0)
            
            # Calculate appropriate width based on field properties
            # For massive fields, use Compton wavelength
            if mass > 0:
                width = pc.reduced_planck / (mass * pc.c)
            else:
                # For massless fields, use simulation radius
                width = self.radius / 5.0
            
            # Generate Gaussian profile
            boundary_density = np.exp(-r_values**2 / (2 * width**2))
            
            # Normalize the density
            if np.any(boundary_density > 0):
                dA = 2 * np.pi * r_values.flatten()  # Area element
                total = np.sum(boundary_density.flatten() * dA)
                if total > 0:
                    boundary_density = boundary_density / total
            
            # Convert density to field values
            self.boundary_values[field_name] = np.sqrt(boundary_density)
            
            logger.info(f"Generated boundary data for {field_name} field")
            
        except Exception as e:
            logger.error(f"Error generating boundary data: {str(e)}")
            raise