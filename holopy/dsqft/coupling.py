"""
Matter-Entropy Coupling Module

This module implements the coupling between matter and entropy continua through
the information manifestation tensor for the dS/QFT correspondence. This coupling
provides a mechanism for quantum-to-classical transitions and the emergence of
observable reality from quantum information.
"""

import numpy as np
import logging
from typing import Dict, Union, Optional, Callable, Tuple, List, Any
from scipy.special import gamma as gamma_function

from holopy.constants.physical_constants import PhysicalConstants
from holopy.constants.dsqft_constants import DSQFTConstants

# Setup logging
logger = logging.getLogger(__name__)

class InformationManifestationTensor:
    """
    Information manifestation tensor for coupling matter and entropy continua.
    
    This class implements the J^μν tensor that couples matter and entropy continua
    and provides a mechanism for quantum-to-classical transitions. The tensor has
    the form:
    J^μν = ∇^μ∇^νρ - γρ^μν
    
    where ρ is the information density scalar and ρ^μν is the entropy distribution
    tensor.
    
    Attributes:
        d (int): Number of spacetime dimensions (typically 4)
        gamma (float): Information processing rate γ
        hubble_parameter (float): Hubble parameter H
        density_grid (np.ndarray): Grid for information density computation
    """
    
    def __init__(self, d: int = 4, gamma: Optional[float] = None, 
                hubble_parameter: Optional[float] = None):
        """
        Initialize the information manifestation tensor.
        
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
        
        # Initialize grid for computation
        self.density_grid = None
        
        logger.debug(f"InformationManifestationTensor initialized with d={d}")
    
    def set_density_grid(self, density_grid: np.ndarray) -> None:
        """
        Set the grid for information density computation.
        
        Args:
            density_grid (np.ndarray): Grid for information density computation
        """
        self.density_grid = density_grid
        logger.debug(f"Density grid set with shape {density_grid.shape}")
    
    def compute_density_gradient(self, density: np.ndarray, 
                                dx: Union[float, np.ndarray]) -> np.ndarray:
        """
        Compute the gradient of the information density.
        
        Args:
            density (np.ndarray): Information density field
            dx (float or np.ndarray): Grid spacing
            
        Returns:
            np.ndarray: Gradient of the information density
        """
        # For a uniform grid
        if isinstance(dx, float):
            gradient = np.gradient(density, dx)
        else:
            # For non-uniform grid
            gradient = np.gradient(density, *dx)
        
        return np.array(gradient)
    
    def compute_density_hessian(self, density: np.ndarray, 
                               dx: Union[float, np.ndarray]) -> np.ndarray:
        """
        Compute the Hessian of the information density.
        
        Args:
            density (np.ndarray): Information density field
            dx (float or np.ndarray): Grid spacing
            
        Returns:
            np.ndarray: Hessian of the information density
        """
        # Compute the gradient
        gradient = self.compute_density_gradient(density, dx)
        
        # Compute the Hessian (second derivatives)
        dim = len(gradient)
        hessian = np.zeros((dim, dim) + density.shape)
        
        for i in range(dim):
            hessian_row = self.compute_density_gradient(gradient[i], dx)
            for j in range(dim):
                hessian[i, j] = hessian_row[j]
        
        return hessian
    
    def compute_entropy_distribution(self, density: np.ndarray, 
                                    dx: Union[float, np.ndarray]) -> np.ndarray:
        """
        Compute the entropy distribution tensor.
        
        The entropy distribution tensor ρ^μν is derived from the information density
        and involves its derivatives and quantum state properties.
        
        Args:
            density (np.ndarray): Information density field
            dx (float or np.ndarray): Grid spacing
            
        Returns:
            np.ndarray: Entropy distribution tensor
        """
        # Compute the gradient
        gradient = self.compute_density_gradient(density, dx)
        
        # Number of dimensions
        dim = len(gradient)
        
        # Initialize entropy tensor
        entropy_tensor = np.zeros((dim, dim) + density.shape)
        
        # Compute entropy tensor components
        for i in range(dim):
            for j in range(dim):
                # Basic contribution from gradient products
                entropy_tensor[i, j] = gradient[i] * gradient[j]
                
                # Add contribution from density directly
                if i == j:
                    entropy_tensor[i, j] += density
        
        # Normalize by total information content
        total_info = np.sum(density) * (dx if isinstance(dx, float) else np.prod(dx))
        if total_info > 0:
            entropy_tensor = entropy_tensor / total_info
        
        return entropy_tensor
    
    def compute_manifestation_tensor(self, density: np.ndarray, 
                                    dx: Union[float, np.ndarray]) -> np.ndarray:
        """
        Compute the information manifestation tensor.
        
        The information manifestation tensor has the form:
        J^μν = ∇^μ∇^νρ - γρ^μν
        
        Args:
            density (np.ndarray): Information density field
            dx (float or np.ndarray): Grid spacing
            
        Returns:
            np.ndarray: Information manifestation tensor
        """
        # Compute the Hessian (second derivatives)
        hessian = self.compute_density_hessian(density, dx)
        
        # Compute the entropy distribution tensor
        entropy_tensor = self.compute_entropy_distribution(density, dx)
        
        # Compute the information manifestation tensor
        # J^μν = ∇^μ∇^νρ - γρ^μν
        manifestation_tensor = hessian - self.gamma * entropy_tensor
        
        return manifestation_tensor
    
    def compute_divergence(self, tensor: np.ndarray, 
                          dx: Union[float, np.ndarray]) -> np.ndarray:
        """
        Compute the divergence of the information manifestation tensor.
        
        The divergence ∇_μJ^μν represents the flow of information and should
        satisfy a modified conservation law when boundary information manifests.
        
        Args:
            tensor (np.ndarray): Tensor field
            dx (float or np.ndarray): Grid spacing
            
        Returns:
            np.ndarray: Divergence of the tensor
        """
        # Number of dimensions
        dim = tensor.shape[0]
        
        # Initialize divergence
        divergence = np.zeros((dim,) + tensor.shape[2:])
        
        # Compute divergence
        for i in range(dim):
            if isinstance(dx, float):
                div_i = np.gradient(tensor[i, 0], dx, axis=0)
                for j in range(1, dim):
                    div_i += np.gradient(tensor[i, j], dx, axis=j)
            else:
                div_i = np.gradient(tensor[i, 0], dx[0], axis=0)
                for j in range(1, dim):
                    div_i += np.gradient(tensor[i, j], dx[j], axis=j)
            
            divergence[i] = div_i
        
        return divergence
    
    def check_conservation_law(self, density: np.ndarray, 
                             dx: Union[float, np.ndarray]) -> float:
        """
        Check if the information manifestation tensor satisfies the modified
        conservation law.
        
        The modified conservation law has the form:
        ∇_μJ^μν = γ · ρ^ν + (γ^2/c^4) · ℋ^ν(ρ,J) + 𝒪(γ^3)
        
        Args:
            density (np.ndarray): Information density field
            dx (float or np.ndarray): Grid spacing
            
        Returns:
            float: Maximum relative error in the conservation law
        """
        # Compute the manifestation tensor
        manifestation_tensor = self.compute_manifestation_tensor(density, dx)
        
        # Compute the divergence
        divergence = self.compute_divergence(manifestation_tensor, dx)
        
        # Compute the information current
        info_current = self.compute_density_gradient(density, dx)
        
        # Right side of the conservation law: γ · ρ^ν
        right_side = self.gamma * info_current
        
        # Compute relative error
        max_error = 0.0
        for i in range(len(right_side)):
            # Skip regions where right_side is very small
            mask = np.abs(right_side[i]) > 1e-10
            if np.any(mask):
                error = np.max(np.abs((divergence[i][mask] - right_side[i][mask]) / right_side[i][mask]))
                max_error = max(max_error, error)
        
        return max_error
    
    def compute_higher_order_functional(self, density: np.ndarray, 
                                      manifestion_tensor: np.ndarray,
                                      dx: Union[float, np.ndarray]) -> np.ndarray:
        """
        Compute the higher-order functional in the conservation law.
        
        The higher-order functional ℋ^ν is a complex functional of the density
        and manifestation tensor that appears in the full conservation law.
        
        Args:
            density (np.ndarray): Information density field
            manifestion_tensor (np.ndarray): Information manifestation tensor
            dx (float or np.ndarray): Grid spacing
            
        Returns:
            np.ndarray: Higher-order functional
        """
        # Number of dimensions
        dim = manifestion_tensor.shape[0]
        
        # Initialize higher-order functional
        functional = np.zeros((dim,) + density.shape)
        
        # Compute higher-order functional
        # This is a simplified approximation
        for i in range(dim):
            # Contribution from trace of manifestation tensor
            trace = np.zeros(density.shape)
            for j in range(dim):
                trace += manifestion_tensor[j, j]
            
            # Contribution from density gradient
            gradient = self.compute_density_gradient(density, dx)
            gradient_squared = np.zeros(density.shape)
            for j in range(dim):
                gradient_squared += gradient[j]**2
            
            # Combine contributions
            functional[i] = trace * gradient[i] + gradient_squared * manifestion_tensor[i, i]
        
        return functional
    
    def compute_energy_momentum_tensor(self, density: np.ndarray, 
                                     dx: Union[float, np.ndarray]) -> np.ndarray:
        """
        Compute the information energy-momentum tensor derived from the
        manifestation tensor.
        
        The information energy-momentum tensor T^μν_info is derived from the
        manifestation tensor and contributes to the right side of Einstein's
        field equations.
        
        Args:
            density (np.ndarray): Information density field
            dx (float or np.ndarray): Grid spacing
            
        Returns:
            np.ndarray: Information energy-momentum tensor
        """
        # Compute the manifestation tensor
        manifestation_tensor = self.compute_manifestation_tensor(density, dx)
        
        # Number of dimensions
        dim = manifestation_tensor.shape[0]
        
        # Initialize energy-momentum tensor
        energy_momentum = np.zeros((dim, dim) + density.shape)
        
        # Compute trace of the manifestation tensor
        trace = np.zeros(density.shape)
        for i in range(dim):
            trace += manifestation_tensor[i, i]
        
        # Compute the energy-momentum tensor
        for i in range(dim):
            for j in range(dim):
                # Contract the manifestation tensor
                contraction = np.zeros(density.shape)
                for k in range(dim):
                    for l in range(dim):
                        contraction += manifestation_tensor[i, k] * manifestation_tensor[j, l]
                
                # Set the energy-momentum tensor
                # T^μν_info = J^μα J^ν_α - (1/2) g^μν J^αβ J_αβ
                energy_momentum[i, j] = contraction
                if i == j:
                    energy_momentum[i, j] -= 0.5 * trace**2
        
        return energy_momentum
    
    def solve_1d_manifestation_equation(self, x_grid: np.ndarray, 
                                      initial_density: np.ndarray,
                                      time_span: Tuple[float, float],
                                      num_time_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the 1D manifestation equation for the information density.
        
        The manifestation equation describes how the information density evolves
        due to the information processing constraints.
        
        Args:
            x_grid (np.ndarray): Spatial grid for computation
            initial_density (np.ndarray): Initial information density
            time_span (Tuple[float, float]): Start and end times
            num_time_points (int, optional): Number of time points (default: 100)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Time points and density evolution
        """
        from scipy.integrate import solve_ivp
        
        # Grid spacing
        dx = x_grid[1] - x_grid[0]
        
        # Define the ODE for manifestation
        def manifestation_ode(t, density_vec):
            # Convert to 1D array
            density = density_vec
            
            # Compute the manifestation tensor
            manifestation_tensor = self.compute_manifestation_tensor(density, dx)
            
            # Compute the divergence
            divergence = self.compute_divergence(manifestation_tensor, dx)
            
            # Compute the right-hand side of the ODE
            # ∂ρ/∂t = -∇_μJ^μ0
            dpsi_dt = -divergence[0]
            
            return dpsi_dt
        
        # Time points
        t_eval = np.linspace(time_span[0], time_span[1], num_time_points)
        
        # Solve the ODE
        solution = solve_ivp(
            manifestation_ode,
            time_span,
            initial_density,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-8
        )
        
        # Extract solution
        times = solution.t
        density_evolution = solution.y
        
        return times, density_evolution


class MatterEntropyCoupling:
    """
    Coupling between matter and entropy continua through the information
    manifestation tensor.
    
    This class implements mechanisms for the coupling between matter and entropy
    continua, providing a framework for understanding quantum-to-classical
    transitions and the emergence of observable reality from quantum information.
    
    Attributes:
        d (int): Number of spacetime dimensions (typically 4)
        gamma (float): Information processing rate γ
        hubble_parameter (float): Hubble parameter H
        manifestation_tensor (InformationManifestationTensor): Information manifestation tensor
    """
    
    def __init__(self, d: int = 4, gamma: Optional[float] = None, 
                hubble_parameter: Optional[float] = None):
        """
        Initialize matter-entropy coupling mechanism.
        
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
        
        # Initialize the information manifestation tensor
        self.manifestation_tensor = InformationManifestationTensor(
            d=self.d, gamma=self.gamma, hubble_parameter=self.hubble_parameter
        )
        
        logger.debug(f"MatterEntropyCoupling initialized with d={d}")
    
    def matter_to_entropy_conversion(self, matter_density: np.ndarray, 
                                   dx: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate entropy production from matter density.
        
        The coupling between matter and entropy results in entropy production
        proportional to the spatial complexity of the matter distribution.
        
        Args:
            matter_density (np.ndarray): Matter density field
            dx (float or np.ndarray): Grid spacing
            
        Returns:
            np.ndarray: Entropy production rate field
        """
        # Compute the manifestation tensor
        manifestation_tensor = self.manifestation_tensor.compute_manifestation_tensor(
            matter_density, dx
        )
        
        # Compute the divergence
        divergence = self.manifestation_tensor.compute_divergence(manifestation_tensor, dx)
        
        # Entropy production rate = -∇_μJ^μ0
        entropy_production = -divergence[0]
        
        return entropy_production
    
    def compute_quantum_classical_boundary(self, matter_density: np.ndarray, 
                                         dx: Union[float, np.ndarray]) -> np.ndarray:
        """
        Compute the quantum-classical boundary from matter distribution.
        
        The quantum-classical boundary separates regions where quantum effects
        dominate from regions where classical behavior emerges due to information
        processing constraints.
        
        Args:
            matter_density (np.ndarray): Matter density field
            dx (float or np.ndarray): Grid spacing
            
        Returns:
            np.ndarray: Classicality parameter field (0 = quantum, 1 = classical)
        """
        # Compute the gradient of the matter density
        gradient = self.manifestation_tensor.compute_density_gradient(matter_density, dx)
        
        # Compute spatial complexity as |∇ρ|^2
        complexity = np.zeros_like(matter_density)
        for i in range(len(gradient)):
            complexity += gradient[i]**2
        
        # Compute manifestation timescale
        dc = DSQFTConstants()
        manifestation_timescale = np.zeros_like(complexity)
        for idx in np.ndindex(complexity.shape):
            manifestation_timescale[idx] = dc.get_manifestation_timescale(complexity[idx])
        
        # Compute the classicality parameter
        # c = 1 - exp(-γ/τ_manifestation)
        # where τ_manifestation = 1/(γ |∇ρ|^2)
        classicality = 1.0 - np.exp(-1.0 / manifestation_timescale)
        
        return classicality
    
    def compute_modified_gravity(self, matter_density: np.ndarray, 
                               dx: Union[float, np.ndarray]) -> np.ndarray:
        """
        Compute the modification to the gravitational field due to information
        processing constraints.
        
        The information manifestation tensor contributes to the right side of
        Einstein's field equations, resulting in a modified gravitational field.
        
        Args:
            matter_density (np.ndarray): Matter density field
            dx (float or np.ndarray): Grid spacing
            
        Returns:
            np.ndarray: Modification to the gravitational field
        """
        # Compute the information energy-momentum tensor
        info_energy_momentum = self.manifestation_tensor.compute_energy_momentum_tensor(
            matter_density, dx
        )
        
        # Compute the trace of the energy-momentum tensor
        trace = np.zeros_like(matter_density)
        for i in range(self.d):
            trace += info_energy_momentum[i, i]
        
        # Compute the contribution to the gravitational field
        # In the weak field approximation, the Newtonian potential is given by:
        # ∇^2Φ = 4πG(ρ + ρ_info)
        # where ρ_info = T^00_info - (1/2)T^μ_μ is the effective energy density
        
        # Compute the effective energy density
        effective_density = info_energy_momentum[0, 0] - 0.5 * trace
        
        # The modification to the gravitational field is proportional to:
        # ∇Φ_mod = 4πG ∫ ρ_info(x') (x-x')/|x-x'|^3 d^3x'
        # This would require a convolution with the Green's function for Poisson's equation
        # For simplicity, we just return the effective density
        
        return effective_density
    
    def compute_spacetime_metric_correction(self, matter_density: np.ndarray, 
                                          dx: Union[float, np.ndarray]) -> np.ndarray:
        """
        Compute the correction to the spacetime metric due to information
        processing constraints.
        
        The information manifestation tensor contributes to the right side of
        Einstein's field equations, resulting in a modified spacetime metric.
        
        Args:
            matter_density (np.ndarray): Matter density field
            dx (float or np.ndarray): Grid spacing
            
        Returns:
            np.ndarray: Correction to the spacetime metric
        """
        # Compute the information energy-momentum tensor
        info_energy_momentum = self.manifestation_tensor.compute_energy_momentum_tensor(
            matter_density, dx
        )
        
        # In the weak field approximation, the metric perturbation is given by:
        # h_μν = -16πG ∫ T_μν(x') / |x-x'| d^3x'
        # For simplicity, we just return the energy-momentum tensor
        
        return info_energy_momentum
    
    def compute_critical_density(self, radius: float) -> float:
        """
        Compute the critical matter density for quantum-classical transition.
        
        The critical density marks the threshold at which quantum behavior
        transitions to classical behavior due to information processing constraints.
        
        Args:
            radius (float): Characteristic length scale in meters
            
        Returns:
            float: Critical matter density in kg/m^3
        """
        # In the holographic framework, the critical density depends on the
        # information processing rate γ and the characteristic length scale
        
        pc = PhysicalConstants()
        dc = DSQFTConstants()
        
        # Critical density scales as:
        # ρ_crit = (1/γ) ⋅ (1/r^2) ⋅ (c^2/G)
        critical_density = (1.0 / self.gamma) * (1.0 / radius**2) * (pc.c**2 / pc.G)
        
        return critical_density
    
    def verify_coupling_properties(self, test_points: int = 50) -> Dict[str, bool]:
        """
        Verify that matter-entropy coupling mechanisms satisfy key mathematical properties.
        
        Args:
            test_points (int, optional): Number of test points to use
            
        Returns:
            Dict[str, bool]: Results of verification tests
        """
        results = {}
        
        # 1. Test conservation law for the information manifestation tensor
        
        np.random.seed(42)  # For reproducibility
        
        # Create a 1D grid
        x_grid = np.linspace(-10.0, 10.0, test_points)
        dx = x_grid[1] - x_grid[0]
        
        # Create a test density profile (Gaussian)
        sigma = 2.0
        density = np.exp(-0.5 * (x_grid / sigma)**2)
        
        # Check conservation law
        max_error = self.manifestation_tensor.check_conservation_law(density, dx)
        results['conservation_law'] = max_error < 0.1
        
        # 2. Test quantum-classical boundary calculation
        
        # Compute the quantum-classical boundary
        classicality = self.compute_quantum_classical_boundary(density, dx)
        
        # The classicality should be high where the density changes rapidly
        # and low where the density is flat
        gradient = np.gradient(density, dx)
        gradient_squared = gradient**2
        
        # The classicality should be correlated with the gradient squared
        correlation = np.corrcoef(classicality, gradient_squared)[0, 1]
        results['qc_boundary_correlation'] = correlation > 0.9
        
        # 3. Test modified gravity calculation
        
        # Compute the modified gravity
        mod_gravity = self.compute_modified_gravity(density, dx)
        
        # The modification should be real-valued
        results['mod_gravity_real'] = np.all(np.isreal(mod_gravity))
        
        # The modification should be non-zero
        results['mod_gravity_nonzero'] = np.any(np.abs(mod_gravity) > 1e-10)
        
        return results 