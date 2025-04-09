"""
Field-Operator Dictionary Module

This module implements the field-operator dictionary for the dS/QFT correspondence,
which maps fields in the bulk to operators on the boundary. The dictionary is a
fundamental component of the holographic framework, enabling the translation between
bulk gravitational physics and boundary quantum field theory.
"""

import numpy as np
import logging
from typing import Dict, Union, Optional, Callable, Tuple, List, Any
from enum import Enum

from holopy.constants.physical_constants import PhysicalConstants
from holopy.constants.dsqft_constants import DSQFTConstants
from holopy.dsqft.propagator import BulkBoundaryPropagator

# Setup logging
logger = logging.getLogger(__name__)

class FieldType(Enum):
    """Enumeration of field types in the bulk."""
    SCALAR = 1
    VECTOR = 2
    TENSOR = 3
    SPINOR = 4


class OperatorType(Enum):
    """Enumeration of operator types on the boundary."""
    SCALAR = 1
    VECTOR = 2
    TENSOR = 3
    SPINOR = 4
    COMPOSITE = 5


class FieldOperatorDictionary:
    """
    Field-operator dictionary for the dS/QFT correspondence.
    
    This class implements the dictionary that maps fields in the bulk to
    operators on the boundary, accounting for information processing constraints.
    It serves as the fundamental translation tool between the bulk gravitational
    description and the boundary quantum field theory.
    
    Attributes:
        d (int): Number of spacetime dimensions (typically 4)
        gamma (float): Information processing rate γ
        hubble_parameter (float): Hubble parameter H
        conformal_dimensions (Dict): Dictionary of conformal dimensions for fields
        bulk_boundary_map (Dict): Mapping between bulk fields and boundary operators
    """
    
    def __init__(self, d: int = 4, gamma: Optional[float] = None, 
                 hubble_parameter: Optional[float] = None):
        """
        Initialize the field-operator dictionary.
        
        Args:
            d (int, optional): Number of spacetime dimensions (default: 4)
            gamma (float, optional): Information processing rate γ (default: from constants)
            hubble_parameter (float, optional): Hubble parameter H (default: from constants)
        """
        self.d = d
        
        # Get constants if not provided
        pc = PhysicalConstants()
        self.gamma = gamma if gamma is not None else pc.gamma
        self.hubble_parameter = hubble_parameter if hubble_parameter is not None else pc.hubble_parameter
        
        # Initialize conformal dimensions for standard fields
        self.conformal_dimensions = {
            # For a massless scalar field
            (FieldType.SCALAR, 'massless'): (self.d - 2) / 2,
            
            # For a massive scalar field, Δ = d/2 + sqrt((d/2)² + m²/H²)
            # This will be computed dynamically based on mass
            
            # For a vector field (e.g. gauge field)
            (FieldType.VECTOR, 'massless'): (self.d - 1) / 2,
            
            # For the graviton (massless spin-2)
            (FieldType.TENSOR, 'massless'): self.d / 2,
            
            # For spinor fields
            (FieldType.SPINOR, 'massless'): (self.d - 1) / 2
        }
        
        # Initialize the dictionary mapping bulk fields to boundary operators
        self.bulk_boundary_map = {}
        
        # Initialize propagators dictionary
        self.propagators = {}
        
        logger.debug(f"FieldOperatorDictionary initialized with d={d}")
    
    def register_bulk_field(self, field_name: str, field_type: FieldType, 
                           mass: float = 0.0, spin: int = 0, 
                           extra_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a bulk field in the dictionary.
        
        Args:
            field_name (str): Name of the field
            field_type (FieldType): Type of the field (scalar, vector, etc.)
            mass (float, optional): Mass of the field in natural units (default: 0.0)
            spin (int, optional): Spin of the field (default: 0)
            extra_params (Dict[str, Any], optional): Additional parameters
        """
        # Compute conformal dimension based on field type and mass
        if field_type == FieldType.SCALAR:
            if mass == 0.0:
                conf_dim = self.conformal_dimensions[(FieldType.SCALAR, 'massless')]
            else:
                # For a massive scalar field, Δ = d/2 + sqrt((d/2)² + m²/H²)
                conf_dim = self.d/2 + np.sqrt((self.d/2)**2 + (mass/self.hubble_parameter)**2)
        
        elif field_type == FieldType.VECTOR:
            if mass == 0.0:
                conf_dim = self.conformal_dimensions[(FieldType.VECTOR, 'massless')]
            else:
                # For a massive vector field
                conf_dim = self.d/2 + np.sqrt((self.d/2 - 1)**2 + (mass/self.hubble_parameter)**2)
        
        elif field_type == FieldType.TENSOR:
            if mass == 0.0:
                conf_dim = self.conformal_dimensions[(FieldType.TENSOR, 'massless')]
            else:
                # For a massive tensor field
                conf_dim = self.d/2 + np.sqrt((self.d/2 - 2)**2 + (mass/self.hubble_parameter)**2)
        
        elif field_type == FieldType.SPINOR:
            if mass == 0.0:
                conf_dim = self.conformal_dimensions[(FieldType.SPINOR, 'massless')]
            else:
                # For a massive spinor field
                conf_dim = self.d/2 + np.sqrt((self.d/2 - 1/2)**2 + (mass/self.hubble_parameter)**2)
        
        # Create the field registration
        field_info = {
            'name': field_name,
            'type': field_type,
            'mass': mass,
            'spin': spin,
            'conformal_dimension': conf_dim,
            'extra_params': extra_params or {}
        }
        
        # Create corresponding boundary operator
        operator_info = {
            'name': f"O_{field_name}",
            'type': OperatorType.SCALAR if field_type == FieldType.SCALAR else OperatorType.TENSOR,
            'conformal_dimension': conf_dim,
            'spin': spin,
            'extra_params': extra_params or {}
        }
        
        # Associate field and operator in the dictionary
        self.bulk_boundary_map[field_name] = {
            'field': field_info,
            'operator': operator_info
        }
        
        # Create propagator for this field
        self.propagators[field_name] = BulkBoundaryPropagator(
            conformal_dim=conf_dim,
            d=self.d,
            gamma=self.gamma,
            hubble_parameter=self.hubble_parameter
        )
        
        logger.debug(f"Registered bulk field '{field_name}' with conformal dimension {conf_dim}")
    
    def get_field_info(self, field_name: str) -> Dict[str, Any]:
        """
        Get information about a registered bulk field.
        
        Args:
            field_name (str): Name of the field
            
        Returns:
            Dict[str, Any]: Information about the field
            
        Raises:
            KeyError: If the field is not registered
        """
        if field_name not in self.bulk_boundary_map:
            raise KeyError(f"Field '{field_name}' not registered in the dictionary")
        
        return self.bulk_boundary_map[field_name]['field']
    
    def get_operator_info(self, field_name: str) -> Dict[str, Any]:
        """
        Get information about the boundary operator corresponding to a bulk field.
        
        Args:
            field_name (str): Name of the bulk field
            
        Returns:
            Dict[str, Any]: Information about the boundary operator
            
        Raises:
            KeyError: If the field is not registered
        """
        if field_name not in self.bulk_boundary_map:
            raise KeyError(f"Field '{field_name}' not registered in the dictionary")
        
        return self.bulk_boundary_map[field_name]['operator']
    
    def get_propagator(self, field_name: str) -> BulkBoundaryPropagator:
        """
        Get the bulk-boundary propagator for a specific field.
        
        Args:
            field_name (str): Name of the field
            
        Returns:
            BulkBoundaryPropagator: Propagator for the field
            
        Raises:
            KeyError: If the field is not registered
        """
        if field_name not in self.propagators:
            raise KeyError(f"Propagator for field '{field_name}' not available")
        
        return self.propagators[field_name]
    
    def compute_boundary_operator_value(self, field_name: str, bulk_field_func: Callable,
                                       x_boundary: np.ndarray, eta_near_boundary: float = -1e-4) -> float:
        """
        Compute the value of a boundary operator from a bulk field.
        
        Args:
            field_name (str): Name of the bulk field
            bulk_field_func (Callable): Function representing the bulk field
            x_boundary (np.ndarray): Spatial coordinates on the boundary
            eta_near_boundary (float, optional): Small negative value of conformal time
            
        Returns:
            float: Value of the boundary operator
            
        Raises:
            KeyError: If the field is not registered
        """
        if field_name not in self.bulk_boundary_map:
            raise KeyError(f"Field '{field_name}' not registered in the dictionary")
        
        # Get field information
        field_info = self.bulk_boundary_map[field_name]['field']
        conf_dim = field_info['conformal_dimension']
        
        # For the standard dictionary, the boundary operator is obtained by:
        # O(x') = lim_{η→0} η^{-Δ} φ(η,x')
        
        # We use a small negative value of η instead of the limit
        # Get the bulk field value at the near-boundary point
        field_value = bulk_field_func(eta_near_boundary, x_boundary)
        
        # Compute the operator value
        operator_value = field_value * ((-eta_near_boundary) ** (-conf_dim))
        
        # Apply information processing modification
        # This factor accounts for the exponential decay due to the information processing rate
        info_mod = np.exp(self.gamma * abs(eta_near_boundary))
        operator_value *= info_mod
        
        return operator_value
    
    def compute_bulk_field_value(self, field_name: str, boundary_operator_func: Callable,
                                boundary_grid: np.ndarray, eta: float, x_bulk: np.ndarray) -> float:
        """
        Compute the value of a bulk field from a boundary operator.
        
        Args:
            field_name (str): Name of the bulk field
            boundary_operator_func (Callable): Function representing the boundary operator
            boundary_grid (np.ndarray): Grid of boundary points for integration
            eta (float): Conformal time in the bulk
            x_bulk (np.ndarray): Spatial coordinates in the bulk
            
        Returns:
            float: Value of the bulk field
            
        Raises:
            KeyError: If the field is not registered
        """
        if field_name not in self.propagators:
            raise KeyError(f"Propagator for field '{field_name}' not available")
        
        # Get the propagator for this field
        propagator = self.propagators[field_name]
        
        # Compute the field value using the propagator
        field_value = propagator.compute_field_from_boundary(
            boundary_operator_func, eta, x_bulk, boundary_grid
        )
        
        return field_value
    
    def verify_dictionary_properties(self, test_points: int = 50) -> Dict[str, bool]:
        """
        Verify that the field-operator dictionary satisfies key mathematical properties.
        
        Args:
            test_points (int, optional): Number of test points to use
            
        Returns:
            Dict[str, bool]: Results of verification tests
        """
        results = {}
        
        # First, we need to register a test field if none exists
        test_field_name = "_test_scalar"
        if test_field_name not in self.bulk_boundary_map:
            self.register_bulk_field(test_field_name, FieldType.SCALAR, mass=0.0)
        
        # 1. Test bulk-to-boundary and boundary-to-bulk mappings for consistency
        # We'll create a test bulk field, map it to the boundary, then back to bulk
        np.random.seed(42)  # For reproducibility
        
        # Define a simple Gaussian bulk field
        def test_bulk_field(eta, x):
            # A Gaussian field centered at the origin
            if isinstance(x, list) or (isinstance(x, np.ndarray) and x.ndim > 1):
                return np.array([np.exp(-np.sum(xi**2)) for xi in x])
            return np.exp(-np.sum(x**2))
        
        # Select test points
        eta_values = -np.logspace(-3, 1, test_points)  # Log-spaced points from near 0 to -10
        
        relative_errors = []
        for eta in eta_values:
            # Generate a random point
            x_point = np.random.uniform(-1.0, 1.0, self.d-1)
            
            # Original field value
            original_value = test_bulk_field(eta, x_point)
            
            # Map to boundary operators
            # First, create a boundary operator function by sampling multiple boundary points
            boundary_samples = []
            boundary_values = []
            
            # Generate boundary points
            for _ in range(test_points * 5):
                x_boundary = np.random.uniform(-2.0, 2.0, self.d-1)
                boundary_samples.append(x_boundary)
                boundary_values.append(
                    self.compute_boundary_operator_value(test_field_name, test_bulk_field, x_boundary)
                )
            
            # Create an interpolation function for the boundary operator
            from scipy.interpolate import LinearNDInterpolator
            
            # Only if we have valid values (non-NaN and finite)
            valid_indices = np.isfinite(boundary_values)
            valid_samples = [boundary_samples[i] for i in range(len(boundary_samples)) if valid_indices[i]]
            valid_values = [boundary_values[i] for i in range(len(boundary_values)) if valid_indices[i]]
            
            if len(valid_samples) > 5:  # Need at least few points for interpolation
                boundary_op_func = LinearNDInterpolator(
                    valid_samples, valid_values, fill_value=0.0
                )
                
                # Map back to bulk
                reconstructed_value = self.compute_bulk_field_value(
                    test_field_name, boundary_op_func, valid_samples, eta, x_point
                )
                
                # Compute relative error
                if abs(original_value) > 1e-10:
                    relative_errors.append(abs(reconstructed_value - original_value) / abs(original_value))
        
        # Check if the error is within acceptable limits
        results['bulk_boundary_consistency'] = (np.mean(relative_errors) < 0.2)
        
        # 2. Test scaling properties
        # For a scalar field, O(λx) = λ^(-Δ) O(x)
        
        scaling_errors = []
        x_base = np.random.uniform(-1.0, 1.0, self.d-1)
        
        for scale in np.logspace(-1, 1, 10):  # Scales from 0.1 to 10
            x_scaled = scale * x_base
            
            # Compute operator values
            base_value = self.compute_boundary_operator_value(test_field_name, test_bulk_field, x_base)
            scaled_value = self.compute_boundary_operator_value(test_field_name, test_bulk_field, x_scaled)
            
            # Expected scaling relation
            conf_dim = self.bulk_boundary_map[test_field_name]['field']['conformal_dimension']
            expected_scaled = base_value * scale**(-conf_dim)
            
            # Compute relative error
            if abs(expected_scaled) > 1e-10:
                scaling_errors.append(abs(scaled_value - expected_scaled) / abs(expected_scaled))
        
        # Check if the error is within acceptable limits
        results['scaling_properties'] = (np.mean(scaling_errors) < 0.1)
        
        # Clean up test field
        if test_field_name in self.bulk_boundary_map:
            del self.bulk_boundary_map[test_field_name]
            del self.propagators[test_field_name]
        
        return results 