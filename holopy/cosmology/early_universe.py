"""
Implementation of Early Universe Evolution in Holographic Framework.

This module provides implementations for simulating the early universe
evolution in the holographic framework, including inflation, reheating,
and critical transitions.
"""

import numpy as np
import logging
from typing import Optional, Union, Dict, List, Tuple, Callable
from scipy.integrate import solve_ivp

from holopy.constants.physical_constants import PhysicalConstants
from holopy.constants.e8_constants import E8Constants

# Setup logging
logger = logging.getLogger(__name__)

def inflation_parameters(
    clustering_coefficient: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute inflation parameters constrained by the E8×E8 framework.
    
    In the holographic framework, inflation parameters like the
    scalar spectral index and tensor-to-scalar ratio are constrained
    by the E8×E8 structure.
    
    Args:
        clustering_coefficient (float, optional): If None, use theoretical value
        
    Returns:
        Dict[str, float]: Dictionary of inflation parameters
    """
    # Get constants
    constants = PhysicalConstants()
    e8_constants = E8Constants()
    
    # Get clustering coefficient
    if clustering_coefficient is None:
        C_G = e8_constants.get_clustering_coefficient()
    else:
        C_G = clustering_coefficient
    
    logger.debug(f"Computing inflation parameters with C_G = {C_G}")
    
    # In the holographic framework, inflation parameters are constrained
    # by the E8×E8 structure
    
    # This is a simplified model - in a real implementation, this would
    # involve more detailed calculations
    
    # Compute scalar spectral index
    # n_s ≈ 1 - 2/N_e, where N_e is the number of e-folds
    # In the holographic framework, N_e is related to the clustering coefficient
    
    # Approximate relation: N_e ≈ 50 + 10 * (C_G - 0.75)
    N_e = 50 + 10 * (C_G - 0.75)
    
    # Compute spectral index
    n_s = 1.0 - 2.0 / N_e
    
    # Compute tensor-to-scalar ratio
    # In standard slow-roll inflation: r ≈ 16ε
    # In the holographic framework, ε is constrained by the information processing rate
    
    # Approximate relation
    epsilon = 1.0 / (2.0 * N_e)
    r = 16.0 * epsilon
    
    # Compute amplitude of scalar perturbations
    # In the holographic framework, this is related to the information processing rate
    
    gamma = constants.get_gamma()
    A_s = 2.1e-9  # Baseline value from Planck
    
    # Correction factor from holographic constraints
    # This is a simplified model
    A_s_correction = 1.0 + 0.1 * (C_G - 0.75)
    A_s *= A_s_correction
    
    # Compute energy scale of inflation
    # V^(1/4) ≈ r^(1/4) * 10^16 GeV
    V_scale = (r / 0.1)**(1/4) * 1.0e16  # in GeV
    
    # Store results
    params = {
        'n_s': n_s,
        'r': r,
        'A_s': A_s,
        'N_e': N_e,
        'epsilon': epsilon,
        'V_scale': V_scale,
        'clustering_coefficient': C_G
    }
    
    logger.info(f"Computed inflation parameters: n_s={n_s:.4f}, r={r:.6f}, A_s={A_s:.2e}")
    return params

def simulate_inflation(
    num_e_folds: float = 60.0,
    scalar_field_potential: str = 'quadratic',
    time_steps: int = 1000,
    clustering_coefficient: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """
    Simulate the inflationary phase of the early universe.
    
    Args:
        num_e_folds (float, optional): Target number of e-folds
        scalar_field_potential (str, optional): Type of inflation potential
        time_steps (int, optional): Number of time steps for simulation
        clustering_coefficient (float, optional): If None, use theoretical value
        
    Returns:
        Dict[str, np.ndarray]: Simulation results
    """
    # Get constants
    constants = PhysicalConstants()
    e8_constants = E8Constants()
    
    # Get clustering coefficient
    if clustering_coefficient is None:
        C_G = e8_constants.get_clustering_coefficient()
    else:
        C_G = clustering_coefficient
    
    logger.info(f"Simulating inflation with {num_e_folds} e-folds")
    
    # Planck mass in natural units
    m_pl = 1.0
    
    # Initial conditions
    # In the holographic framework, these might be constrained by the
    # information processing rate
    
    # Initial scalar field value
    phi_initial = np.sqrt(2 * num_e_folds) * m_pl
    
    # Initial scalar field velocity (slow-roll approximation)
    phi_dot_initial = -potential_derivative(phi_initial, scalar_field_potential) / (3 * hubble_inflation(phi_initial, scalar_field_potential))
    
    # Initial scale factor (arbitrary)
    a_initial = 1.0
    
    # Define the differential equations
    def inflation_equations(t, y):
        phi, phi_dot, a = y
        
        H = hubble_inflation(phi, scalar_field_potential)
        
        # Apply holographic corrections
        # In the holographic framework, the equations of motion are modified
        # by information processing constraints
        
        # Information processing rate
        gamma = constants.get_gamma()
        
        # Correction factor based on clustering coefficient
        # This is a simplified model
        correction = 1.0 + 0.1 * (C_G - 0.75) * np.exp(-phi**2 / (2 * m_pl**2))
        
        # Modified equations
        dphi_dt = phi_dot
        dphi_dot_dt = -3 * H * phi_dot - potential_derivative(phi, scalar_field_potential) * correction
        da_dt = a * H
        
        return [dphi_dt, dphi_dot_dt, da_dt]
    
    # Time array (in Planck units)
    # The actual time scale will depend on the energy scale of inflation
    t_span = [0, 100]  # Should be enough for most inflation models
    
    # Solve the differential equations
    sol = solve_ivp(
        inflation_equations,
        t_span,
        [phi_initial, phi_dot_initial, a_initial],
        method='RK45',
        t_eval=np.linspace(t_span[0], t_span[1], time_steps)
    )
    
    if not sol.success:
        logger.error(f"Inflation simulation failed: {sol.message}")
        raise RuntimeError(f"Inflation simulation failed: {sol.message}")
    
    # Extract results
    t = sol.t
    phi = sol.y[0]
    phi_dot = sol.y[1]
    a = sol.y[2]
    
    # Calculate derived quantities
    H = np.array([hubble_inflation(phi_i, scalar_field_potential) for phi_i in phi])
    
    # Calculate e-folds
    N_e = np.log(a / a_initial)
    
    # Calculate slow-roll parameters
    epsilon = 0.5 * (phi_dot / H / m_pl)**2
    eta = m_pl**2 * second_derivative(phi, scalar_field_potential) / potential(phi, scalar_field_potential)
    
    # Store results
    results = {
        't': t,
        'phi': phi,
        'phi_dot': phi_dot,
        'a': a,
        'H': H,
        'N_e': N_e,
        'epsilon': epsilon,
        'eta': eta
    }
    
    # Add holographic information
    results['clustering_coefficient'] = C_G
    results['gamma'] = constants.get_gamma()
    
    logger.info(f"Inflation simulation completed with {N_e[-1]:.2f} e-folds")
    return results

def hubble_inflation(phi: float, potential_type: str) -> float:
    """
    Calculate the Hubble parameter during inflation.
    
    Args:
        phi (float): Scalar field value
        potential_type (str): Type of inflation potential
        
    Returns:
        float: Hubble parameter in Planck units
    """
    # Planck mass in natural units
    m_pl = 1.0
    
    # Calculate potential energy
    V = potential(phi, potential_type)
    
    # In the slow-roll approximation, H² ≈ V / (3m_pl²)
    H_squared = V / (3.0 * m_pl**2)
    
    return np.sqrt(H_squared)

def potential(phi: float, potential_type: str) -> float:
    """
    Calculate the scalar field potential.
    
    Args:
        phi (float): Scalar field value
        potential_type (str): Type of inflation potential
        
    Returns:
        float: Potential energy
    """
    # Planck mass in natural units
    m_pl = 1.0
    
    # Mass parameter (tuned to give the correct amplitude of fluctuations)
    m = 1.5e-6 * m_pl
    
    if potential_type.lower() == 'quadratic':
        # V(φ) = (1/2) m² φ²
        return 0.5 * m**2 * phi**2
    elif potential_type.lower() == 'quartic':
        # V(φ) = (λ/4) φ⁴
        lambda_param = 1.0e-13
        return 0.25 * lambda_param * phi**4
    elif potential_type.lower() == 'natural':
        # V(φ) = Λ⁴ [1 + cos(φ/f)]
        Lambda = 1.0e-3 * m_pl
        f = 10.0 * m_pl
        return Lambda**4 * (1.0 + np.cos(phi / f))
    elif potential_type.lower() == 'hilltop':
        # V(φ) = Λ⁴ [1 - (φ/μ)²]²
        Lambda = 1.0e-3 * m_pl
        mu = 10.0 * m_pl
        return Lambda**4 * (1.0 - (phi / mu)**2)**2
    else:
        raise ValueError(f"Unsupported potential type: {potential_type}")

def potential_derivative(phi: float, potential_type: str) -> float:
    """
    Calculate the derivative of the scalar field potential.
    
    Args:
        phi (float): Scalar field value
        potential_type (str): Type of inflation potential
        
    Returns:
        float: Potential derivative dV/dφ
    """
    # Planck mass in natural units
    m_pl = 1.0
    
    # Mass parameter
    m = 1.5e-6 * m_pl
    
    if potential_type.lower() == 'quadratic':
        # dV/dφ = m² φ
        return m**2 * phi
    elif potential_type.lower() == 'quartic':
        # dV/dφ = λ φ³
        lambda_param = 1.0e-13
        return lambda_param * phi**3
    elif potential_type.lower() == 'natural':
        # dV/dφ = -Λ⁴ sin(φ/f) / f
        Lambda = 1.0e-3 * m_pl
        f = 10.0 * m_pl
        return -Lambda**4 * np.sin(phi / f) / f
    elif potential_type.lower() == 'hilltop':
        # dV/dφ = -4 Λ⁴ (φ/μ) [1 - (φ/μ)²] / μ
        Lambda = 1.0e-3 * m_pl
        mu = 10.0 * m_pl
        return -4.0 * Lambda**4 * (phi / mu) * (1.0 - (phi / mu)**2) / mu
    else:
        raise ValueError(f"Unsupported potential type: {potential_type}")

def second_derivative(phi: float, potential_type: str) -> float:
    """
    Calculate the second derivative of the scalar field potential.
    
    Args:
        phi (float): Scalar field value
        potential_type (str): Type of inflation potential
        
    Returns:
        float: Second derivative d²V/dφ²
    """
    # Planck mass in natural units
    m_pl = 1.0
    
    # Mass parameter
    m = 1.5e-6 * m_pl
    
    if potential_type.lower() == 'quadratic':
        # d²V/dφ² = m²
        return m**2
    elif potential_type.lower() == 'quartic':
        # d²V/dφ² = 3λ φ²
        lambda_param = 1.0e-13
        return 3.0 * lambda_param * phi**2
    elif potential_type.lower() == 'natural':
        # d²V/dφ² = -Λ⁴ cos(φ/f) / f²
        Lambda = 1.0e-3 * m_pl
        f = 10.0 * m_pl
        return -Lambda**4 * np.cos(phi / f) / f**2
    elif potential_type.lower() == 'hilltop':
        # d²V/dφ² = -4 Λ⁴ [1 - 3(φ/μ)²] / μ²
        Lambda = 1.0e-3 * m_pl
        mu = 10.0 * m_pl
        return -4.0 * Lambda**4 * (1.0 - 3.0 * (phi / mu)**2) / mu**2
    else:
        raise ValueError(f"Unsupported potential type: {potential_type}")

def simulate_reheating(
    inflation_results: Dict[str, np.ndarray],
    reheating_efficiency: float = 0.9,
    time_steps: int = 500,
    clustering_coefficient: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """
    Simulate the reheating phase after inflation.
    
    Args:
        inflation_results (Dict[str, np.ndarray]): Results from simulate_inflation
        reheating_efficiency (float, optional): Efficiency of energy transfer
        time_steps (int, optional): Number of time steps for simulation
        clustering_coefficient (float, optional): If None, use theoretical value
        
    Returns:
        Dict[str, np.ndarray]: Simulation results
    """
    # Get constants
    constants = PhysicalConstants()
    e8_constants = E8Constants()
    
    # Get clustering coefficient
    if clustering_coefficient is None:
        C_G = e8_constants.get_clustering_coefficient()
    else:
        C_G = clustering_coefficient
    
    logger.info(f"Simulating reheating with efficiency {reheating_efficiency}")
    
    # Extract final state from inflation
    phi_end = inflation_results['phi'][-1]
    phi_dot_end = inflation_results['phi_dot'][-1]
    a_end = inflation_results['a'][-1]
    
    # Initial radiation energy density (start with zero)
    rho_r_end = 0.0
    
    # Define the differential equations for reheating
    def reheating_equations(t, y):
        phi, phi_dot, a, rho_r = y
        
        # Total energy density
        rho_phi = 0.5 * phi_dot**2 + potential(phi, 'quadratic')
        rho_total = rho_phi + rho_r
        
        # Hubble parameter
        H = np.sqrt(rho_total / 3.0)
        
        # Decay rate of the inflaton
        # This depends on the coupling to other fields
        # For simplicity, we use a phenomenological approach
        Gamma = 1.0e-7  # in Planck units
        
        # Apply holographic corrections
        # In the holographic framework, the energy transfer rate is constrained
        # by the information processing rate
        
        # Information processing rate
        gamma = constants.get_gamma()
        
        # Correction factor based on clustering coefficient
        # This is a simplified model
        correction = 1.0 + 0.1 * (C_G - 0.75) * np.exp(-t / 100.0)
        
        # Modified energy transfer rate
        Gamma_effective = Gamma * correction
        
        # Equations of motion
        dphi_dt = phi_dot
        dphi_dot_dt = -3 * H * phi_dot - potential_derivative(phi, 'quadratic') - Gamma_effective * phi_dot
        da_dt = a * H
        drho_r_dt = -4 * H * rho_r + Gamma_effective * phi_dot**2 * reheating_efficiency
        
        return [dphi_dt, dphi_dot_dt, da_dt, drho_r_dt]
    
    # Time array (in Planck units)
    t_span = [0, 1000]  # Should be enough for reheating
    
    # Solve the differential equations
    sol = solve_ivp(
        reheating_equations,
        t_span,
        [phi_end, phi_dot_end, a_end, rho_r_end],
        method='RK45',
        t_eval=np.linspace(t_span[0], t_span[1], time_steps)
    )
    
    if not sol.success:
        logger.error(f"Reheating simulation failed: {sol.message}")
        raise RuntimeError(f"Reheating simulation failed: {sol.message}")
    
    # Extract results
    t = sol.t
    phi = sol.y[0]
    phi_dot = sol.y[1]
    a = sol.y[2]
    rho_r = sol.y[3]
    
    # Calculate inflaton energy density
    rho_phi = 0.5 * phi_dot**2 + np.array([potential(phi_i, 'quadratic') for phi_i in phi])
    
    # Calculate total energy density
    rho_total = rho_phi + rho_r
    
    # Calculate Hubble parameter
    H = np.sqrt(rho_total / 3.0)
    
    # Calculate temperature
    # In natural units, T ∝ ρ_r^(1/4)
    # Convert to GeV
    T = (rho_r)**(1/4) * 1.22e19  # GeV (Planck mass in GeV)
    
    # Calculate e-folds since the end of inflation
    N_e = np.log(a / a_end)
    
    # Store results
    results = {
        't': t,
        'phi': phi,
        'phi_dot': phi_dot,
        'a': a,
        'rho_r': rho_r,
        'rho_phi': rho_phi,
        'rho_total': rho_total,
        'H': H,
        'T': T,
        'N_e': N_e
    }
    
    # Add holographic information
    results['clustering_coefficient'] = C_G
    results['gamma'] = constants.get_gamma()
    
    logger.info(f"Reheating simulation completed with final temperature {T[-1]:.2e} GeV")
    return results

def simulate_early_universe(
    num_e_folds: float = 60.0,
    scalar_field_potential: str = 'quadratic',
    reheating_efficiency: float = 0.9,
    clustering_coefficient: Optional[float] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Simulate the early universe evolution including inflation and reheating.
    
    Args:
        num_e_folds (float, optional): Target number of e-folds for inflation
        scalar_field_potential (str, optional): Type of inflation potential
        reheating_efficiency (float, optional): Efficiency of energy transfer
        clustering_coefficient (float, optional): If None, use theoretical value
        
    Returns:
        Dict[str, Dict[str, np.ndarray]]: Simulation results for different phases
    """
    logger.info("Simulating early universe evolution")
    
    # Get constants
    constants = PhysicalConstants()
    e8_constants = E8Constants()
    
    # Get clustering coefficient
    if clustering_coefficient is None:
        C_G = e8_constants.get_clustering_coefficient()
    else:
        C_G = clustering_coefficient
    
    # Step 1: Simulate inflation
    inflation_results = simulate_inflation(
        num_e_folds=num_e_folds,
        scalar_field_potential=scalar_field_potential,
        clustering_coefficient=C_G
    )
    
    # Step 2: Simulate reheating
    reheating_results = simulate_reheating(
        inflation_results=inflation_results,
        reheating_efficiency=reheating_efficiency,
        clustering_coefficient=C_G
    )
    
    # Combine results
    results = {
        'inflation': inflation_results,
        'reheating': reheating_results
    }
    
    logger.info("Early universe simulation completed")
    return results

def compute_critical_transitions(
    simulation_results: Dict[str, Dict[str, np.ndarray]]
) -> List[Dict[str, float]]:
    """
    Identify critical transitions in the early universe evolution.
    
    Args:
        simulation_results (Dict[str, Dict[str, np.ndarray]]): Results from simulate_early_universe
        
    Returns:
        List[Dict[str, float]]: List of critical transitions
    """
    logger.info("Analyzing early universe evolution for critical transitions")
    
    # Extract inflation results
    inflation = simulation_results['inflation']
    t_infl = inflation['t']
    phi_infl = inflation['phi']
    H_infl = inflation['H']
    epsilon_infl = inflation['epsilon']
    
    # Extract reheating results
    reheating = simulation_results['reheating']
    t_reh = reheating['t']
    rho_r_reh = reheating['rho_r']
    rho_phi_reh = reheating['rho_phi']
    T_reh = reheating['T']
    
    # Find end of inflation (when epsilon = 1)
    inflation_end_idx = np.argmin(np.abs(epsilon_infl - 1.0))
    
    # Find reheating completion (when rho_r > rho_phi)
    reheating_complete_idx = np.argmax(rho_r_reh > rho_phi_reh)
    if reheating_complete_idx == 0 and rho_r_reh[0] <= rho_phi_reh[0]:
        # If radiation never dominates, take the last point
        reheating_complete_idx = len(rho_r_reh) - 1
    
    # Create list of critical transitions
    transitions = [
        {
            'name': 'end_of_inflation',
            'time': t_infl[inflation_end_idx],
            'phi': phi_infl[inflation_end_idx],
            'H': H_infl[inflation_end_idx],
            'epsilon': epsilon_infl[inflation_end_idx]
        },
        {
            'name': 'reheating_complete',
            'time': t_reh[reheating_complete_idx],
            'temperature': T_reh[reheating_complete_idx],
            'rho_r': rho_r_reh[reheating_complete_idx],
            'rho_phi': rho_phi_reh[reheating_complete_idx]
        }
    ]
    
    # Look for other interesting transitions
    # For example, when reheating is 50% complete
    half_reheating_idx = np.argmin(np.abs(rho_r_reh / (rho_r_reh + rho_phi_reh) - 0.5))
    transitions.append({
        'name': 'half_reheating',
        'time': t_reh[half_reheating_idx],
        'temperature': T_reh[half_reheating_idx],
        'rho_r': rho_r_reh[half_reheating_idx],
        'rho_phi': rho_phi_reh[half_reheating_idx]
    })
    
    logger.info(f"Identified {len(transitions)} critical transitions")
    return transitions 