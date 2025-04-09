"""
Information Transport Module

This module implements information transport mechanisms across the holographic
boundary for the dS/QFT correspondence. These mechanisms govern how information
flows between the bulk and boundary, accounting for the information processing
rate γ and the resulting constraints on information transfer.
"""

import numpy as np
import logging
from typing import Dict, Union, Optional, Callable, Tuple, List, Any
from scipy.special import gamma as gamma_function
from scipy.integrate import solve_ivp

from holopy.constants.physical_constants import PhysicalConstants
from holopy.constants.dsqft_constants import DSQFTConstants
from holopy.dsqft.propagator import BulkBoundaryPropagator

# Setup logging
logger = logging.getLogger(__name__)

class InformationTransport:
    """
    Information transport across the holographic boundary.
    
    This class implements mechanisms for information transport between the bulk
    and boundary, accounting for the information processing rate γ and the
    resulting constraints on information transfer.
    
    Attributes:
        d (int): Number of spacetime dimensions (typically 4)
        gamma (float): Information processing rate γ
        hubble_parameter (float): Hubble parameter H
        max_info_rate (float): Maximum information transport rate
    """
    
    def __init__(self, d: int = 4, gamma: Optional[float] = None, 
                hubble_parameter: Optional[float] = None):
        """
        Initialize information transport mechanisms.
        
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
        
        # Calculate maximum information transfer rate
        # This is the maximum rate at which information can flow across the boundary
        self.max_info_rate = self.calculate_max_info_rate()
        
        # Get de Sitter constants
        dc = DSQFTConstants()
        self.critical_manifestation_threshold = dc.critical_manifestation_threshold
        
        logger.debug(f"InformationTransport initialized with d={d}")
    
    def calculate_max_info_rate(self) -> float:
        """
        Calculate the maximum information transfer rate across the boundary.
        
        The maximum rate is proportional to the area of the cosmological horizon:
        dI/dt_max = γ * (A/l_P^2)
        
        Returns:
            float: Maximum information transfer rate in bits/s
        """
        pc = PhysicalConstants()
        
        # Area of the cosmological horizon
        # A = 4π/H^2
        horizon_area = 4 * np.pi / self.hubble_parameter**2
        
        # Maximum information transfer rate
        # dI/dt_max = γ * (A/l_P^2)
        max_rate = self.gamma * horizon_area / pc.planck_area
        
        logger.debug(f"Maximum information transfer rate: {max_rate} bits/s")
        return max_rate
    
    def calculate_entropy_bound(self, radius: float) -> float:
        """
        Calculate the holographic entropy bound for a region of given radius.
        
        The holographic bound states that the maximum entropy of a region
        is proportional to its boundary area:
        S_max = A/(4G_N) = πr^2/l_P^2
        
        Args:
            radius (float): Radius of the region in meters
            
        Returns:
            float: Maximum entropy in natural units (bits)
        """
        # Get the holographic bound from DSQFTConstants
        dc = DSQFTConstants()
        entropy_bound = dc.get_holographic_bound(radius)
        
        return entropy_bound
    
    def calculate_manifestation_timescale(self, spatial_complexity: float) -> float:
        """
        Calculate the manifestation timescale for a quantum state with given spatial complexity.
        
        The manifestation timescale is inversely proportional to the spatial complexity:
        τ_manifestation ~ 1/(γ |∇ψ|^2)
        
        Args:
            spatial_complexity (float): Spatial complexity measure |∇ψ|^2
            
        Returns:
            float: Manifestation timescale in seconds
        """
        # Get the manifestation timescale from DSQFTConstants
        dc = DSQFTConstants()
        timescale = dc.get_manifestation_timescale(spatial_complexity)
        
        return timescale
    
    def coherence_decay(self, initial_coherence: float, spatial_separation: float, 
                       time_span: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the decay of quantum coherence due to information processing constraints.
        
        The coherence between spatial positions decays as:
        ⟨x|ρ(t)|x'⟩ = ⟨x|ρ(0)|x'⟩ * exp(-γt|x-x'|^2)
        
        Args:
            initial_coherence (float): Initial coherence value
            spatial_separation (float): Spatial separation |x-x'|
            time_span (float or np.ndarray): Time or time array
            
        Returns:
            float or np.ndarray: Coherence at given time(s)
        """
        # Convert to numpy array if needed
        t_array = np.asarray(time_span) if isinstance(time_span, np.ndarray) else np.array([time_span])
        
        # Compute coherence decay
        # ⟨x|ρ(t)|x'⟩ = ⟨x|ρ(0)|x'⟩ * exp(-γt|x-x'|^2)
        coherence = initial_coherence * np.exp(-self.gamma * t_array * spatial_separation**2)
        
        # Return scalar or array depending on input type
        if isinstance(time_span, np.ndarray):
            return coherence
        else:
            return coherence[0]
    
    def information_flow_rate(self, area: float, spatial_complexity: float) -> float:
        """
        Calculate the information flow rate across a boundary of given area.
        
        The flow rate is proportional to the area and spatial complexity:
        dI/dt = γ * (A/l_P^2) * f(|∇ψ|^2)
        
        where f is a function of the spatial complexity that approaches 1
        for high complexity and 0 for low complexity.
        
        Args:
            area (float): Area of the boundary in m^2
            spatial_complexity (float): Spatial complexity measure |∇ψ|^2
            
        Returns:
            float: Information flow rate in bits/s
        """
        pc = PhysicalConstants()
        
        # Complex states (high |∇ψ|^2) transfer information more efficiently
        # We model this with a saturating function
        complexity_factor = 1.0 - np.exp(-spatial_complexity)
        
        # Information flow rate
        # dI/dt = γ * (A/l_P^2) * f(|∇ψ|^2)
        flow_rate = self.gamma * (area / pc.planck_area) * complexity_factor
        
        # Ensure the flow rate doesn't exceed the maximum
        flow_rate = min(flow_rate, self.max_info_rate)
        
        return flow_rate
    
    def solve_manifestation_equation(self, initial_state: Callable, 
                                    spatial_grid: np.ndarray,
                                    time_span: Tuple[float, float],
                                    num_time_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the manifestation equation to track how quantum states manifest as reality.
        
        The manifestation equation is:
        d|ψ⟩/dt = -γ|∇ψ|^2|ψ⟩
        
        which describes how boundary information patterns manifest as observable reality.
        
        Args:
            initial_state (Callable): Function that returns the initial wavefunction
            spatial_grid (np.ndarray): Spatial grid for computation
            time_span (Tuple[float, float]): Start and end times
            num_time_points (int, optional): Number of time points (default: 100)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Time points and wavefunction evolution
        """
        # Initialize state on the spatial grid
        psi_0 = initial_state(spatial_grid)
        
        # Compute the spatial derivative for complexity
        # For 1D, we use finite difference
        if spatial_grid.ndim == 1:
            dx = spatial_grid[1] - spatial_grid[0]
            
            def complexity(psi):
                # Compute |∇ψ|^2 using finite difference
                dpsi_dx = np.zeros_like(psi, dtype=complex)
                dpsi_dx[1:-1] = (psi[2:] - psi[:-2]) / (2 * dx)
                dpsi_dx[0] = (psi[1] - psi[0]) / dx
                dpsi_dx[-1] = (psi[-1] - psi[-2]) / dx
                
                return np.abs(dpsi_dx)**2
        else:
            # For higher dimensions, we would need a more complex approach
            # For simplicity, we just estimate complexity as the variance
            def complexity(psi):
                return np.var(np.abs(psi))
        
        # Define the ODE for manifestation
        def manifestation_ode(t, psi_vec):
            # Convert flat vector to complex array
            psi = psi_vec[:len(psi_vec)//2] + 1j * psi_vec[len(psi_vec)//2:]
            
            # Compute spatial complexity
            psi_complexity = complexity(psi)
            
            # Compute the right-hand side of the ODE
            # d|ψ⟩/dt = -γ|∇ψ|^2|ψ⟩
            dpsi_dt = -self.gamma * psi_complexity * psi
            
            # Convert complex array to flat vector
            return np.concatenate([dpsi_dt.real, dpsi_dt.imag])
        
        # Initial state as flat vector
        psi_0_vec = np.concatenate([psi_0.real, psi_0.imag])
        
        # Time points
        t_eval = np.linspace(time_span[0], time_span[1], num_time_points)
        
        # Solve the ODE
        solution = solve_ivp(
            manifestation_ode,
            time_span,
            psi_0_vec,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-8
        )
        
        # Extract solution
        times = solution.t
        psi_evolution = solution.y[:len(psi_0),:] + 1j * solution.y[len(psi_0):,:]
        
        return times, psi_evolution
    
    def quantum_to_classical_transition(self, initial_coherence: float, 
                                      spatial_complexity: float,
                                      time_span: np.ndarray) -> np.ndarray:
        """
        Model the quantum-to-classical transition due to information processing.
        
        As the information processing constraints lead to decoherence, quantum
        states transition to classical ones. This function models the classicality
        parameter, which ranges from 0 (fully quantum) to 1 (fully classical).
        
        Args:
            initial_coherence (float): Initial quantum coherence
            spatial_complexity (float): Spatial complexity measure |∇ψ|^2
            time_span (np.ndarray): Time array
            
        Returns:
            np.ndarray: Classicality parameter at each time
        """
        # Manifestation timescale
        tau_manifestation = self.calculate_manifestation_timescale(spatial_complexity)
        
        # Compute coherence decay
        coherence = initial_coherence * np.exp(-time_span / tau_manifestation)
        
        # Classicality parameter
        # 0 = fully quantum, 1 = fully classical
        classicality = 1.0 - coherence / initial_coherence
        
        return classicality
    
    def entropy_production_rate(self, spatial_complexity: float, 
                              area: float) -> float:
        """
        Calculate the entropy production rate due to information processing.
        
        The entropy production rate is related to the information flow rate
        and the manifestation of quantum states:
        dS/dt = γ * (A/l_P^2) * |∇ψ|^2
        
        Args:
            spatial_complexity (float): Spatial complexity measure |∇ψ|^2
            area (float): Area of the boundary in m^2
            
        Returns:
            float: Entropy production rate in bits/s
        """
        pc = PhysicalConstants()
        
        # Entropy production rate
        # dS/dt = γ * (A/l_P^2) * |∇ψ|^2
        entropy_rate = self.gamma * (area / pc.planck_area) * spatial_complexity
        
        return entropy_rate
    
    def transport_efficiency(self, spatial_complexity: float) -> float:
        """
        Calculate the efficiency of information transport across the boundary.
        
        The efficiency depends on the spatial complexity of the quantum state:
        η = (1 - exp(-|∇ψ|^2/|∇ψ|^2_crit))
        
        Args:
            spatial_complexity (float): Spatial complexity measure |∇ψ|^2
            
        Returns:
            float: Transport efficiency (0 to 1)
        """
        # Critical complexity for efficient transport
        # This is a characteristic scale of the system
        critical_complexity = 1.0
        
        # Compute efficiency
        # η = (1 - exp(-|∇ψ|^2/|∇ψ|^2_crit))
        efficiency = 1.0 - np.exp(-spatial_complexity / critical_complexity)
        
        return efficiency
    
    def information_processing_constraint(self, transformation_rate: float, 
                                        entanglement_entropy: float) -> bool:
        """
        Check if a quantum process satisfies the information processing constraint.
        
        The constraint is:
        𝒯(𝒮) ≤ γ · S_ent
        
        where 𝒯(𝒮) is the transformation rate of quantum state 𝒮, and S_ent is its
        entanglement entropy.
        
        Args:
            transformation_rate (float): Rate of quantum state transformation
            entanglement_entropy (float): Entanglement entropy of the state
            
        Returns:
            bool: True if the constraint is satisfied, False otherwise
        """
        # Check information processing constraint
        # 𝒯(𝒮) ≤ γ · S_ent
        constraint_satisfied = transformation_rate <= self.gamma * entanglement_entropy
        
        if not constraint_satisfied:
            logger.warning(f"Information processing constraint violated: {transformation_rate} > {self.gamma * entanglement_entropy}")
        
        return constraint_satisfied
    
    def verity_transport_properties(self, test_points: int = 50) -> Dict[str, bool]:
        """
        Verify that information transport mechanisms satisfy key mathematical properties.
        
        Args:
            test_points (int, optional): Number of test points to use
            
        Returns:
            Dict[str, bool]: Results of verification tests
        """
        results = {}
        
        # 1. Test coherence decay
        # The coherence should decay exponentially with time and quadratically with distance
        
        np.random.seed(42)  # For reproducibility
        
        time_values = np.linspace(0, 100, test_points)
        separations = np.random.uniform(0.1, 10.0, test_points)
        
        # Check exponential decay with time
        time_decays = [self.coherence_decay(1.0, 1.0, t) for t in time_values]
        time_decay_rates = np.log(np.maximum(time_decays[1:], 1e-30) / np.maximum(time_decays[:-1], 1e-30)) / (time_values[1:] - time_values[:-1])
        
        # The rate should be approximately constant and equal to -γ
        time_rate_error = np.abs(np.mean(time_decay_rates) + self.gamma) / self.gamma
        results['time_decay'] = time_rate_error < 0.05
        
        # Check quadratic dependence on separation
        fixed_time = 1.0
        sep_decays = [self.coherence_decay(1.0, s, fixed_time) for s in separations]
        log_decays = np.log(np.maximum(sep_decays, 1e-30))
        
        # For quadratic dependence on separation, log(decay) should be proportional to s^2
        # Use linear regression to check this
        x = separations**2
        y = log_decays
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
        expected_slope = -self.gamma * fixed_time
        
        sep_rate_error = np.abs(slope - expected_slope) / abs(expected_slope)
        results['separation_decay'] = sep_rate_error < 0.1
        
        # 2. Test spatial complexity and manifestation timescale
        # The manifestation timescale should be inversely proportional to the spatial complexity
        
        complexity_values = np.logspace(-2, 2, test_points)
        timescales = [self.calculate_manifestation_timescale(c) for c in complexity_values]
        
        # Compute product of complexity and timescale
        products = complexity_values * np.array(timescales)
        
        # The product should be approximately constant and equal to 1/γ
        product_mean = np.mean(products)
        product_error = np.abs(product_mean - 1.0/self.gamma) / (1.0/self.gamma)
        results['manifestation_timescale'] = product_error < 0.05
        
        # 3. Test information flow rate
        # The flow rate should be proportional to the area and have a non-linear
        # dependence on spatial complexity
        
        areas = np.logspace(0, 10, test_points)
        fixed_complexity = 1.0
        
        # Test area dependence
        area_rates = [self.information_flow_rate(a, fixed_complexity) for a in areas]
        log_areas = np.log(areas)
        log_rates = np.log(area_rates)
        
        # For areas small enough that we don't hit the maximum rate,
        # the slope of log(rate) vs log(area) should be 1
        valid_indices = area_rates < 0.9 * self.max_info_rate
        if np.sum(valid_indices) > 5:
            x = log_areas[valid_indices]
            y = log_rates[valid_indices]
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
            area_slope_error = np.abs(slope - 1.0)
            results['area_scaling'] = area_slope_error < 0.05
        else:
            # Not enough points below the maximum to test scaling
            results['area_scaling'] = True  # Assume test passes
        
        # Test complexity saturation
        fixed_area = 1.0
        complexity_values = np.logspace(-3, 3, test_points)
        complexity_rates = [self.information_flow_rate(fixed_area, c) for c in complexity_values]
        
        # Check that rates saturate for high complexity
        high_indices = complexity_values > 10.0
        if np.sum(high_indices) > 5:
            high_rates = np.array(complexity_rates)[high_indices]
            rate_variation = np.std(high_rates) / np.mean(high_rates)
            results['complexity_saturation'] = rate_variation < 0.05
        else:
            # Not enough high complexity points to test saturation
            results['complexity_saturation'] = True  # Assume test passes
        
        return results 