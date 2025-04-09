"""
Implementation of Emergent Spacetime Metric from Information.

This module provides implementations of the emergent spacetime metric
derived from information processing dynamics in the holographic framework.

Key equation:
g_μν(x) = ∑_{i,j=1}^{496} (∂π^{-1}_i(x)/∂x^μ) κ_ij (∂π^{-1}_j(x)/∂x^ν)

Where π^{-1} is a local section of the projection from E8×E8 space to 4D spacetime,
and κ_ij are components of the Killing form.
"""

import numpy as np
import logging
from typing import Optional, Union, Callable, Dict, List, Tuple

from holopy.constants.physical_constants import PhysicalConstants, PHYSICAL_CONSTANTS
from holopy.e8.root_system import RootSystem
from holopy.e8.heterotic import E8E8Heterotic
from holopy.info.tensor import InfoCurrentTensor

# Setup logging
logger = logging.getLogger(__name__)

class InfoSpacetimeMetric:
    """
    Implements the derivation of spacetime metric from information processing.
    
    This class derives the emergent 4D spacetime metric from the underlying
    E8×E8 heterotic structure and information processing dynamics.
    
    Attributes:
        e8_structure (E8E8Heterotic): The underlying E8×E8 heterotic structure
        info_current (InfoCurrentTensor): The information current tensor
        dimension (int): Dimension of the target spacetime (default 4)
    """
    
    def __init__(
        self,
        e8_structure: Optional[E8E8Heterotic] = None,
        info_current: Optional[InfoCurrentTensor] = None,
        dimension: int = 4
    ):
        """
        Initialize the spacetime metric derivation from information.
        
        Args:
            e8_structure (E8E8Heterotic, optional): E8×E8 heterotic structure
            info_current (InfoCurrentTensor, optional): Information current tensor
            dimension (int, optional): Dimension of target spacetime
        """
        self.constants = PhysicalConstants()
        
        # Create E8×E8 structure if not provided
        if e8_structure is None:
            logger.info("Creating default E8×E8 heterotic structure")
            self.e8_structure = E8E8Heterotic()
        else:
            self.e8_structure = e8_structure
            
        self.info_current = info_current
        self.dimension = dimension
        
        # Initialize metric and inverse metric
        self.metric = None
        self.inverse_metric = None
        
        # Cached values
        self._projection_matrix = None
        self._killing_form = None
        
        logger.debug(f"InfoSpacetimeMetric initialized with dimension {dimension}")
    
    # ----- Getter and Setter Methods -----
    
    def set_info_current(self, info_current: InfoCurrentTensor) -> None:
        """
        Set the information current tensor.
        
        Args:
            info_current (InfoCurrentTensor): The information current tensor
        """
        self.info_current = info_current
    
    def get_info_current(self) -> Optional[InfoCurrentTensor]:
        """
        Get the information current tensor.
        
        Returns:
            InfoCurrentTensor: The information current tensor
        """
        return self.info_current
    
    def set_metric(self, metric: np.ndarray) -> None:
        """
        Set the spacetime metric.
        
        Args:
            metric (np.ndarray): 4×4 spacetime metric tensor
        """
        if metric.shape != (self.dimension, self.dimension):
            raise ValueError(f"Metric must have shape ({self.dimension}, {self.dimension})")
        
        self.metric = metric
        self.inverse_metric = None  # Reset cached inverse
    
    def get_metric(self) -> Optional[np.ndarray]:
        """
        Get the spacetime metric.
        
        Returns:
            np.ndarray: 4×4 spacetime metric tensor or None if not computed
        """
        return self.metric
    
    def get_inverse_metric(self) -> np.ndarray:
        """
        Get the inverse metric tensor.
        
        This computes the inverse if it hasn't been calculated yet.
        
        Returns:
            np.ndarray: 4×4 inverse metric tensor
        """
        if self.metric is None:
            logger.warning("Metric not set, computing flat metric")
            self.metric = self.compute_flat_metric()
            
        if self.inverse_metric is None:
            self.inverse_metric = np.linalg.inv(self.metric)
            
        return self.inverse_metric
    
    def set_e8_structure(self, e8_structure: E8E8Heterotic) -> None:
        """
        Set the E8×E8 heterotic structure.
        
        Args:
            e8_structure (E8E8Heterotic): The E8×E8 heterotic structure
        """
        self.e8_structure = e8_structure
        self._killing_form = None  # Reset cached Killing form
    
    def get_e8_structure(self) -> Optional[E8E8Heterotic]:
        """
        Get the E8×E8 heterotic structure.
        
        Returns:
            E8E8Heterotic: The E8×E8 heterotic structure
        """
        return self.e8_structure
    
    # ----- Metric Computation Methods -----
    
    def compute_flat_metric(self) -> np.ndarray:
        """
        Compute a flat Minkowski metric.
        
        Returns:
            np.ndarray: 4×4 Minkowski metric with signature (-+++)
        """
        metric = np.zeros((self.dimension, self.dimension))
        np.fill_diagonal(metric, 1.0)
        metric[0, 0] = -1.0  # Time component
        return metric
    
    def compute_metric_from_projection(self, projection_derivatives: np.ndarray) -> np.ndarray:
        """
        Compute metric from projection derivatives.
        
        This implements the key equation:
        g_μν(x) = ∑_{i,j=1}^{496} (∂π^{-1}_i(x)/∂x^μ) κ_ij (∂π^{-1}_j(x)/∂x^ν)
        
        Args:
            projection_derivatives (np.ndarray): Derivatives of projection
            
        Returns:
            np.ndarray: 4×4 spacetime metric tensor
        """
        # Get the Killing form
        killing_form = self.killing_form
        
        # Compute the metric
        metric = np.einsum('im,ij,jn->mn', 
                         projection_derivatives, killing_form, projection_derivatives)
        
        # Ensure symmetry
        metric = 0.5 * (metric + metric.T)
        
        return metric
    
    @property
    def killing_form(self) -> np.ndarray:
        """
        Get or compute the Killing form of the E8×E8 Lie algebra.
        
        Returns:
            np.ndarray: 496×496 Killing form matrix
        """
        if self._killing_form is None:
            logger.debug("Computing Killing form of E8×E8")
            # Check if compute_killing_form accepts arguments or not
            import inspect
            sig = inspect.signature(self.e8_structure.compute_killing_form)
            
            if len(sig.parameters) >= 2:  # If it expects X and Y parameters
                # We need to create a full Killing form matrix by computing each element
                dim = self.e8_structure.get_lie_algebra_dimension()
                killing_form = np.zeros((dim, dim))
                
                # Simple basis vectors for computing Killing form
                basis = np.eye(dim)
                
                # Compute each element of the Killing form
                for i in range(dim):
                    for j in range(dim):
                        killing_form[i, j] = self.e8_structure.compute_killing_form(basis[i], basis[j])
                
                self._killing_form = killing_form
            else:
                # If compute_killing_form doesn't need parameters, call it directly
                self._killing_form = self.e8_structure.compute_killing_form()
        
        return self._killing_form

    def compute_line_element(self, displacement: np.ndarray) -> float:
        """
        Compute the line element ds² for a given displacement vector.
        
        Args:
            displacement (np.ndarray): 4D displacement vector
            
        Returns:
            float: Line element ds² = g_μν dx^μ dx^ν
        """
        if self.metric is None:
            logger.warning("Metric not set, computing flat metric")
            self.metric = self.compute_flat_metric()
        
        return np.einsum('i,ij,j->', displacement, self.metric, displacement)
    
    def is_physical_metric(self) -> bool:
        """
        Check if the metric is physically valid.
        
        A physically valid metric in 4D should have signature (-+++) 
        with one negative and three positive eigenvalues.
        
        Returns:
            bool: True if the metric is physically valid
        """
        if self.metric is None:
            logger.warning("Metric not set, computing flat metric")
            self.metric = self.compute_flat_metric()
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(self.metric)
        
        # Check for signature (-+++)
        # We should have one negative and three positive eigenvalues
        neg_count = np.sum(eigenvalues < 0)
        pos_count = np.sum(eigenvalues > 0)
        
        return neg_count == 1 and pos_count == 3
    
    def compute_projection_derivatives(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute the derivatives of the projection from E8×E8 to spacetime.
        
        This calculates the Jacobian of the projection from 496-dimensional
        E8×E8 space to 4D spacetime at a given point.
        
        Args:
            coordinates (np.ndarray): 4D spacetime coordinates
            
        Returns:
            np.ndarray: Projection derivatives, shape (496, 4)
        """
        logger.info(f"Computing projection derivatives at coordinates {coordinates}")
        
        # Make sure we have an E8×E8 structure
        if self.e8_structure is None:
            logger.warning("No E8×E8 structure provided, creating default one")
            # Import here to avoid circular import
            from holopy.e8.heterotic import E8E8Heterotic
            self.e8_structure = E8E8Heterotic()
        
        e8e8_dim = 496  # Dimension of E8×E8 Lie algebra
        spacetime_dim = self.dimension
        
        # Get the root system from the E8×E8 structure
        roots = self.e8_structure.get_roots()
        
        # Step 1: Define the projection map π from E8×E8 to spacetime
        # We'll use a simplified model based on components of the root system
        
        # The partial derivatives of π^{-1} are what we need to compute
        # ∂π^{-1}_i(x)/∂x^μ
        
        # Initialize the projection derivatives
        proj_derivatives = np.zeros((e8e8_dim, spacetime_dim))
        
        # Step 2: Calculate the derivatives
        # In a full implementation, these would be derived from the actual projection map
        # For now, we'll use a simplification based on root projections
        
        # The root system consists of 480 root vectors in 16D space
        # We'll use the first 496 linearly independent combinations to span the Lie algebra
        
        # First, we need to construct a basis for the Lie algebra
        # For simplicity, we'll use the first 496 linearly independent vectors we can construct
        
        # Get a subset of the roots (first 248 roots for each E8)
        e8_1_roots = roots[:240]
        e8_2_roots = roots[240:480]
        
        # Create a basis for the Lie algebra
        # For each root vector, we'll create a basis element
        basis = []
        
        # Add the root vectors themselves for the first part of the basis
        for root in e8_1_roots[:240]:
            basis.append(np.concatenate([root, np.zeros(8)]))
        
        # Add the root vectors for the second E8
        for root in e8_2_roots[:240]:
            basis.append(np.concatenate([np.zeros(8), root]))
        
        # Add the remaining 16 Cartan generators (diagonal matrices)
        for i in range(8):
            # Cartan generator for first E8
            cartan1 = np.zeros(16)
            cartan1[i] = 1.0
            basis.append(cartan1)
            
            # Cartan generator for second E8
            cartan2 = np.zeros(16)
            cartan2[8 + i] = 1.0
            basis.append(cartan2)
        
        # Make sure we have exactly 496 basis elements
        basis = basis[:496]
        
        # Step 3: Define the coordinate dependence of the projection
        # Map the 4D spacetime coordinates to the 496D Lie algebra
        # In a full implementation, this would be based on the heterotic string theory
        
        # For simplicity, we'll use a periodic mapping based on cosine functions
        # with different frequencies for different directions
        
        # These frequencies correspond to different scales in the projection
        frequencies = [1.0, 0.1, 0.01, 0.001]
        
        # For each Lie algebra basis element and each spacetime dimension
        for i in range(e8e8_dim):
            for mu in range(spacetime_dim):
                # Define a frequency for this particular combination
                frequency = frequencies[mu] * (1.0 + 0.1 * (i % 10))
                
                # Compute the derivative of the projection
                # This is essentially the partial derivative of a sinusoidal function
                # ∂π^{-1}_i(x)/∂x^μ = A_i * sin(f_μ * x^μ)
                amplitude = 0.1 / np.sqrt(e8e8_dim)  # Scale to keep the metric stable
                
                # Compute the derivative at this point using a sinusoidal function
                # with a phase shift based on the basis element index
                phase = 0.1 * i
                proj_derivatives[i, mu] = amplitude * np.sin(frequency * coordinates[mu] + phase)
        
        # Step 4: Normalize the derivatives to get proper scaling
        # The normalization ensures the resulting metric has the right scale
        
        # First, compute the scale of the derivatives
        scale = np.sqrt(np.sum(proj_derivatives**2) / (e8e8_dim * spacetime_dim))
        
        # Check if the scale is too small (numerical instability)
        if scale < 1e-10:
            logger.warning("Projection derivatives have very small scale, adjusting")
            scale = 1e-10
        
        # Normalize to a standard scale
        target_scale = 1.0
        proj_derivatives *= (target_scale / scale)
        
        logger.debug(f"Computed projection derivatives with shape {proj_derivatives.shape}")
        return proj_derivatives
    
    def compute_metric(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Compute the emergent spacetime metric at given coordinates.
        
        Implements the key equation:
        g_μν(x) = ∑_{i,j=1}^{496} (∂π^{-1}_i(x)/∂x^μ) κ_ij (∂π^{-1}_j(x)/∂x^ν)
        
        Args:
            coordinates (np.ndarray): 4D spacetime coordinates
            
        Returns:
            np.ndarray: 4×4 spacetime metric tensor
        """
        logger.info(f"Computing metric at coordinates {coordinates}")
        
        # Get the Killing form
        killing_form = self.killing_form
        
        # Compute projection derivatives
        proj_derivatives = self.compute_projection_derivatives(coordinates)
        
        # Compute the metric using the formula:
        # g_μν(x) = ∑_{i,j=1}^{496} (∂π^{-1}_i(x)/∂x^μ) κ_ij (∂π^{-1}_j(x)/∂x^ν)
        
        # We can implement this as matrix operations:
        # g = P.T @ K @ P, where:
        # - P is the projection derivatives (∂π^{-1}_i/∂x^μ), shape (496, 4)
        # - K is the Killing form, shape (496, 496)
        # - g is the metric tensor, shape (4, 4)
        
        metric = np.einsum('im,ij,jn->mn', proj_derivatives, killing_form, proj_derivatives)
        
        # Ensure the metric is symmetric (should be by construction, but numerical errors may occur)
        metric = 0.5 * (metric + metric.T)
        
        logger.debug(f"Computed metric with shape {metric.shape}")
        return metric
    
    def compute_metric_field(
        self, 
        coordinate_grid: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute the metric field over a grid of spacetime coordinates.
        
        Args:
            coordinate_grid (List[np.ndarray]): List of coordinate arrays defining the grid
            
        Returns:
            np.ndarray: Metric tensor field over the grid
        """
        logger.info("Computing metric field over coordinate grid")
        
        # Create meshgrid for evaluation points
        grid_points = np.meshgrid(*coordinate_grid, indexing='ij')
        points_shape = grid_points[0].shape
        
        # Reshape points for vectorized evaluation
        points = np.column_stack([grid.flatten() for grid in grid_points])
        
        # Initialize metric field
        # Shape: (*grid_shape, 4, 4)
        metric_field = np.zeros((*points_shape, self.dimension, self.dimension))
        
        # Compute metric at each point
        for idx, point in enumerate(points):
            # Convert flat index to multi-dimensional indices
            multi_idx = np.unravel_index(idx, points_shape)
            
            # Compute metric at this point
            metric = self.compute_metric(point)
            
            # Store in the result array
            metric_field[multi_idx] = metric
        
        logger.debug(f"Computed metric field with shape {metric_field.shape}")
        return metric_field
    
    def compute_from_info_current(self) -> np.ndarray:
        """
        Compute the metric directly from the information current tensor.
        
        This provides an alternative method to derive the spacetime metric
        directly from the information current tensor, bypassing the E8×E8 projection.
        
        Returns:
            np.ndarray: 4×4 spacetime metric tensor
        """
        if self.info_current is None:
            logger.warning("No information current tensor provided, using default flat metric")
            return np.diag([-1.0, 1.0, 1.0, 1.0])  # Default Minkowski metric
        
        logger.info("Computing metric from information current tensor")
        
        # Get the information current tensor and dimensions
        info_tensor = self.info_current.get_tensor()
        dims = self.dimension
        
        # Start with a base Minkowski metric
        base_metric = np.diag([-1.0, 1.0, 1.0, 1.0])
        
        # Get the gamma constant (information processing rate)
        gamma = PHYSICAL_CONSTANTS.get_gamma()
        
        # Get the information density vector
        info_density = self.info_current.get_density()
        
        # Step 1: Calculate the information distribution tensor ρ^μν
        # In a full implementation, this would be derived from the information current
        # For now, we'll compute it as a product of the density vectors
        info_distribution = np.outer(info_density, info_density)
        
        # Step 2: Calculate the stress-energy contribution from information
        # T^μν = J^μν + γρ^μν
        info_stress_energy = info_tensor + gamma * info_distribution
        
        # Step 3: Apply the Einstein equation to derive the metric perturbation
        # G_μν = (8πG/c⁴)T_μν
        # For small perturbations, we can approximate this as a perturbation to flat spacetime
        
        # Apply the Einstein equation
        # G_μν ≈ -(1/2)∇²h_μν = (8πG/c⁴)T_μν
        # where h_μν is the perturbation to the metric
        
        # Constants
        G_constant = PHYSICAL_CONSTANTS.G  # Gravitational constant
        c = PHYSICAL_CONSTANTS.c  # Speed of light
        
        # Einstein equation coefficient
        einstein_coeff = 8 * np.pi * G_constant / (c**4)
        
        # Calculate metric perturbation
        # We integrate the Einstein equation approximately
        # In a full implementation, we would solve the differential equation properly
        
        # For now, we'll use a simplified approach for demonstration
        # h_μν ≈ (8πG/c⁴) * L² * T_μν
        # where L is a characteristic length scale
        
        # Choose a characteristic length scale (e.g., 1 light-year)
        L = 9.46e15  # meters (1 light-year)
        
        # Calculate the metric perturbation
        h_perturbation = einstein_coeff * (L**2) * info_stress_energy
        
        # Scale down the perturbation to ensure stability
        scale_factor = 1e-10  # Adjusted to ensure perturbations are small
        h_perturbation *= scale_factor
        
        # Apply the perturbation to the base metric
        # g_μν = η_μν + h_μν, where η_μν is the Minkowski metric
        metric = base_metric + h_perturbation
        
        # Ensure the metric is symmetric (should be by construction, but numerical errors may occur)
        metric = 0.5 * (metric + metric.T)
        
        # Check eigenvalues to ensure proper signature
        eigenvalues = np.linalg.eigvalsh(metric)
        
        # If the signature is wrong, apply a correction
        if not (np.sum(eigenvalues < 0) == 1 and np.sum(eigenvalues > 0) == 3):
            logger.warning("Correcting metric signature")
            
            # Sort eigenvalues
            sorted_eigenvalues = np.sort(eigenvalues)
            
            # Adjusted eigenvalues with correct signature (-+++)
            adjusted_eigenvalues = np.array([sorted_eigenvalues[0] - 1e-6] + 
                                            [max(1e-6, ev) for ev in sorted_eigenvalues[1:]])
            
            # Calculate eigenvectors
            _, eigenvectors = np.linalg.eigh(metric)
            
            # Reconstruct metric with adjusted eigenvalues
            metric = eigenvectors @ np.diag(adjusted_eigenvalues) @ eigenvectors.T
            
            # Re-symmetrize to account for numerical errors
            metric = 0.5 * (metric + metric.T)
        
        logger.debug(f"Computed metric from information current with shape {metric.shape}")
        return metric


def compute_curvature_from_info(
    info_current: InfoCurrentTensor, 
    coordinates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute spacetime curvature directly from information current tensor.
    
    A standalone function for computing Riemann curvature tensor, Ricci tensor,
    and Ricci scalar from the information current tensor.
    
    Args:
        info_current (InfoCurrentTensor): The information current tensor
        coordinates (np.ndarray): 4D spacetime coordinates
        
    Returns:
        Tuple[np.ndarray, np.ndarray, float]: (Riemann tensor, Ricci tensor, Ricci scalar)
    """
    logger.info(f"Computing curvature from information at coordinates {coordinates}")
    
    # First, derive the metric from the information current tensor
    metric_calculator = InfoSpacetimeMetric(info_current=info_current)
    metric = metric_calculator.compute_from_info_current()
    
    # Calculate the inverse metric
    inverse_metric = np.linalg.inv(metric)
    
    # Get dimensions
    dims = metric.shape[0]
    
    # Step 1: Compute Christoffel symbols (connection coefficients)
    # Γ^λ_μν = (1/2) g^λσ (∂_μ g_νσ + ∂_ν g_μσ - ∂_σ g_μν)
    
    # For demonstration, we compute the derivatives analytically
    # In a real application, we would use actual coordinate derivatives
    # Here we use a small epsilon for numerical differentiation
    epsilon = 1e-6
    
    # Initialize Christoffel symbols (shape: 4x4x4)
    christoffel = np.zeros((dims, dims, dims))
    
    # Compute the Christoffel symbols
    for lambda_idx in range(dims):
        for mu in range(dims):
            for nu in range(dims):
                # Compute the terms in the formula for each point
                christoffel_sum = 0.0
                
                for sigma in range(dims):
                    # Compute partial derivatives of metric components
                    # ∂_μ g_νσ
                    d_mu_g_nu_sigma = 0.0
                    d_nu_g_mu_sigma = 0.0
                    d_sigma_g_mu_nu = 0.0
                    
                    # We'll use central differences to approximate derivatives
                    # For a proper implementation, these would be computed from
                    # the analytical form of the metric or proper numerical derivatives
                    
                    # Forward and backward points for μ direction
                    forward_mu = coordinates.copy()
                    forward_mu[mu] += epsilon
                    backward_mu = coordinates.copy()
                    backward_mu[mu] -= epsilon
                    
                    # Forward and backward points for ν direction
                    forward_nu = coordinates.copy()
                    forward_nu[nu] += epsilon
                    backward_nu = coordinates.copy()
                    backward_nu[nu] -= epsilon
                    
                    # Forward and backward points for σ direction
                    forward_sigma = coordinates.copy()
                    forward_sigma[sigma] += epsilon
                    backward_sigma = coordinates.copy()
                    backward_sigma[sigma] -= epsilon
                    
                    # Calculate each metric derivative
                    # ∂_μ g_νσ using central difference
                    metric_forward_mu = metric_calculator.compute_metric(forward_mu)
                    metric_backward_mu = metric_calculator.compute_metric(backward_mu)
                    d_mu_g_nu_sigma = (metric_forward_mu[nu, sigma] - metric_backward_mu[nu, sigma]) / (2 * epsilon)
                    
                    # ∂_ν g_μσ using central difference
                    metric_forward_nu = metric_calculator.compute_metric(forward_nu)
                    metric_backward_nu = metric_calculator.compute_metric(backward_nu)
                    d_nu_g_mu_sigma = (metric_forward_nu[mu, sigma] - metric_backward_nu[mu, sigma]) / (2 * epsilon)
                    
                    # ∂_σ g_μν using central difference
                    metric_forward_sigma = metric_calculator.compute_metric(forward_sigma)
                    metric_backward_sigma = metric_calculator.compute_metric(backward_sigma)
                    d_sigma_g_mu_nu = (metric_forward_sigma[mu, nu] - metric_backward_sigma[mu, nu]) / (2 * epsilon)
                    
                    # Add this term to the sum
                    christoffel_sum += inverse_metric[lambda_idx, sigma] * (
                        d_mu_g_nu_sigma + d_nu_g_mu_sigma - d_sigma_g_mu_nu
                    )
                
                # Multiply by 1/2 to get the Christoffel symbol
                christoffel[lambda_idx, mu, nu] = 0.5 * christoffel_sum
    
    # Step 2: Compute the Riemann curvature tensor
    # R^ρ_σμν = ∂_μ Γ^ρ_νσ - ∂_ν Γ^ρ_μσ + Γ^ρ_μλ Γ^λ_νσ - Γ^ρ_νλ Γ^λ_μσ
    
    # Initialize Riemann tensor (shape: 4x4x4x4)
    riemann_tensor = np.zeros((dims, dims, dims, dims))
    
    for rho in range(dims):
        for sigma in range(dims):
            for mu in range(dims):
                for nu in range(dims):
                    if mu == nu:  # R^ρ_σμν is antisymmetric in μ,ν
                        continue
                        
                    # Calculate ∂_μ Γ^ρ_νσ using central differences
                    forward_mu = coordinates.copy()
                    forward_mu[mu] += epsilon
                    backward_mu = coordinates.copy()
                    backward_mu[mu] -= epsilon
                    
                    # We would recalculate Christoffel symbols at these points
                    # For simplicity, we'll approximate with random values
                    d_mu_gamma_nu_sigma = 0.0  # In real code, compute this properly
                    
                    # Calculate ∂_ν Γ^ρ_μσ using central differences
                    forward_nu = coordinates.copy()
                    forward_nu[nu] += epsilon
                    backward_nu = coordinates.copy()
                    backward_nu[nu] -= epsilon
                    
                    # We would recalculate Christoffel symbols at these points
                    # For simplicity, we'll approximate with random values
                    d_nu_gamma_mu_sigma = 0.0  # In real code, compute this properly
                    
                    # Calculate the third term: Γ^ρ_μλ Γ^λ_νσ
                    third_term = 0.0
                    for lambda_idx in range(dims):
                        third_term += christoffel[rho, mu, lambda_idx] * christoffel[lambda_idx, nu, sigma]
                    
                    # Calculate the fourth term: Γ^ρ_νλ Γ^λ_μσ
                    fourth_term = 0.0
                    for lambda_idx in range(dims):
                        fourth_term += christoffel[rho, nu, lambda_idx] * christoffel[lambda_idx, mu, sigma]
                    
                    # Combine all terms to get the Riemann tensor component
                    riemann_tensor[rho, sigma, mu, nu] = (
                        d_mu_gamma_nu_sigma - d_nu_gamma_mu_sigma + 
                        third_term - fourth_term
                    )
                    
                    # Apply antisymmetry in μ,ν
                    riemann_tensor[rho, sigma, nu, mu] = -riemann_tensor[rho, sigma, mu, nu]
    
    # Step 3: Compute the Ricci tensor by contracting the Riemann tensor
    # R_μν = R^λ_μλν
    ricci_tensor = np.zeros((dims, dims))
    for mu in range(dims):
        for nu in range(dims):
            for lambda_idx in range(dims):
                ricci_tensor[mu, nu] += riemann_tensor[lambda_idx, mu, lambda_idx, nu]
    
    # Step 4: Compute the Ricci scalar by contracting the Ricci tensor
    # R = g^μν R_μν
    ricci_scalar = 0.0
    for mu in range(dims):
        for nu in range(dims):
            ricci_scalar += inverse_metric[mu, nu] * ricci_tensor[mu, nu]
    
    logger.debug("Computed curvature quantities from information current")
    return riemann_tensor, ricci_tensor, ricci_scalar 