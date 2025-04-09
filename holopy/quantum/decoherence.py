"""
Decoherence Module for HoloPy.

This module implements quantum decoherence calculations in the holographic framework,
including the decoherence functional, coherence decay, and spatial complexity measures.
The module implements the equation ⟨x|ρ(t)|x'⟩ = ⟨x|ρ(0)|x'⟩ · exp(-γt|x-x'|²) and 
related decoherence phenomena.
"""

import numpy as np
from typing import Callable, List, Tuple, Dict, Optional, Union, Any
import scipy.sparse as sp
from scipy.integrate import solve_ivp
import logging

from holopy.constants.physical_constants import PHYSICAL_CONSTANTS
from holopy.quantum.modified_schrodinger import WaveFunction, DecoherenceFunctional
from holopy.constants.physical_constants import PhysicalConstants

# Configure logging
logger = logging.getLogger(__name__)

def coherence_decay(rho_0: float, 
                  t: Union[float, np.ndarray], 
                  x1: Union[float, np.ndarray], 
                  x2: Union[float, np.ndarray],
                  gamma: Optional[float] = None) -> Union[float, np.ndarray]:
    """
    Calculate the decay of quantum coherence between spatial positions over time.
    
    This implements the equation: ⟨x|ρ(t)|x'⟩ = ⟨x|ρ(0)|x'⟩ · exp(-γt|x-x'|²)
    
    Args:
        rho_0 (float): Initial coherence value ⟨x|ρ(0)|x'⟩
        t (float or np.ndarray): Time or array of times in seconds
        x1 (float or np.ndarray): First position(s)
        x2 (float or np.ndarray): Second position(s)
        gamma (float, optional): Information processing rate. If None, use default.
        
    Returns:
        float or np.ndarray: Coherence value(s) at time(s) t
    """
    # Get the fundamental information processing rate if not provided
    if gamma is None:
        gamma = PHYSICAL_CONSTANTS.get_gamma()
    
    # Calculate the squared distance
    if isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray):
        if x1.shape != x2.shape:
            raise ValueError(f"Positions x1 and x2 must have the same shape, got {x1.shape} and {x2.shape}")
        
        # For vector positions, compute Euclidean distance
        if len(x1.shape) > 0 and (len(x1.shape) > 1 and x1.shape[1] > 1):
            squared_distance = np.sum((x1 - x2)**2, axis=-1)
        else:
            squared_distance = (x1 - x2)**2
    else:
        # For scalar positions
        squared_distance = (x1 - x2)**2
    
    # Calculate the decay factor
    decay_factor = np.exp(-gamma * t * squared_distance)
    
    # Calculate the coherence
    coherence = rho_0 * decay_factor
    
    return coherence

def spatial_complexity(wavefunction: Union[WaveFunction, Callable[[np.ndarray], complex]], 
                      domain: Optional[List[Tuple[float, float]]] = None, 
                      grid_size: int = 100) -> float:
    """
    Calculate the spatial complexity of a quantum wavefunction, defined as ∫|∇ψ|² dx.
    
    In the holographic framework, this quantity drives decoherence through
    the modified Schrödinger equation.
    
    Args:
        wavefunction: Either a WaveFunction object or a callable that returns the wavefunction value at a point
        domain: Domain boundaries for each dimension (not needed if wavefunction is a WaveFunction object)
        grid_size: Number of grid points in each dimension (not needed if wavefunction is a WaveFunction object)
        
    Returns:
        float: Spatial complexity measure
    """
    # If wavefunction is already a WaveFunction object
    if isinstance(wavefunction, WaveFunction):
        # Use the DecoherenceFunctional to compute the total
        decoherence = DecoherenceFunctional(wavefunction)
        return decoherence.total()
    
    # If wavefunction is a callable, we need to create a WaveFunction
    if domain is None:
        raise ValueError("Domain must be provided if wavefunction is a callable")
    
    # Determine dimension from domain
    dimension = len(domain)
    
    # Create a grid
    grids = []
    for d in range(dimension):
        grids.append(np.linspace(domain[d][0], domain[d][1], grid_size))
    
    mesh_grids = np.meshgrid(*grids, indexing='ij')
    
    # Create a WaveFunction object
    wave_func = WaveFunction(
        initial_function=wavefunction,
        domain=domain,
        grid_size=grid_size
    )
    
    # Calculate spatial complexity
    decoherence = DecoherenceFunctional(wave_func)
    return decoherence.total()

def decoherence_rate(system_size: float) -> float:
    """
    Calculate the decoherence rate for a quantum system of a given size.
    
    In the holographic framework, decoherence rates scale inversely with
    the square of the system size: Rate ∝ L^-2
    
    Args:
        system_size (float): Characteristic size of the system in meters
        
    Returns:
        float: Decoherence rate in s^-1
    """
    # Get the fundamental information processing rate
    gamma = PHYSICAL_CONSTANTS.get_gamma()
    
    # Get physical constants
    pc = PhysicalConstants()
    
    # Calculate the Planck length
    planck_length = np.sqrt(pc.hbar * pc.G / pc.c**3)
    
    # Calculate the decoherence rate
    # The rate is γ for a system of Planck length,
    # and scales as L^-2 for larger systems
    rate = gamma * (planck_length / system_size)**2
    
    return rate

def decoherence_timescale(system_size: float) -> float:
    """
    Calculate the characteristic decoherence timescale for a quantum system.
    
    This is the inverse of the decoherence rate: τ = 1/Γ.
    
    Args:
        system_size (float): Characteristic size of the system in meters
        
    Returns:
        float: Decoherence timescale in seconds
    """
    rate = decoherence_rate(system_size)
    return 1.0 / rate if rate > 0 else float('inf')

def decoherence_length(time: float) -> float:
    """
    Calculate the characteristic decoherence length for a given timescale.
    
    This is the spatial scale at which decoherence occurs within the given time.
    
    Args:
        time (float): Time in seconds
        
    Returns:
        float: Decoherence length in meters
    """
    # Get the fundamental information processing rate
    gamma = PHYSICAL_CONSTANTS.get_gamma()
    
    # Get physical constants
    pc = PhysicalConstants()
    
    # Calculate the Planck length
    planck_length = np.sqrt(pc.hbar * pc.G / pc.c**3)
    
    # Calculate the decoherence length
    # Solve for L in: time = 1 / (gamma * (planck_length / L)^2)
    length = planck_length * np.sqrt(gamma * time)
    
    return length

def decoherence_evolution(wavefunction: WaveFunction, 
                        t_span: List[float], 
                        dt: Optional[float] = None,
                        gamma: Optional[float] = None) -> Dict[str, np.ndarray]:
    """
    Evolve a wavefunction under pure decoherence (no Hamiltonian).
    
    Args:
        wavefunction: Initial wavefunction
        t_span: Time span [t_start, t_end]
        dt: Time step (optional)
        gamma: Decoherence rate (defaults to constant.gamma)
        
    Returns:
        Dict with evolution results
    """
    # Get gamma from constants if not provided
    if gamma is None:
        gamma = PHYSICAL_CONSTANTS.gamma
    
    # Create time array
    if dt is None:
        # Create logarithmically spaced time steps to capture initial rapid evolution
        # and long-term behavior efficiently
        t_eval = np.logspace(np.log10(max(t_span[0], 1e-10)), np.log10(t_span[1]), 100)
        # Ensure t_span[0] is included
        if t_span[0] > 0:
            t_eval = np.unique(np.sort(np.concatenate([[t_span[0]], t_eval])))
    else:
        t_eval = np.arange(t_span[0], t_span[1] + dt/2, dt)
    
    # Make a copy of the initial wavefunction to avoid modifying the original
    evolving_wf = wavefunction.copy()
    
    # Compute the DecoherenceFunctional
    decoherence = DecoherenceFunctional(evolving_wf)
    
    # Store results including full wavefunction evolution
    results = {
        'times': t_eval,
        'norm': np.zeros_like(t_eval),
        'decoherence': np.zeros_like(t_eval),
        'wavefunctions': []  # Store wavefunctions at each time step
    }
    
    # Initialize with initial values
    results['norm'][0] = evolving_wf.get_norm()
    results['decoherence'][0] = decoherence.total()
    results['wavefunctions'].append(evolving_wf.copy())
    
    # Current time
    t_current = t_eval[0]
    
    # Evolve the wavefunction
    for i in range(1, len(t_eval)):
        # Time step
        dt_step = t_eval[i] - t_current
        t_current = t_eval[i]
        
        # Implementation of the stochastic Schrödinger equation with decoherence only
        # dψ/dt = -γ|∇ψ|² ψ
        
        # Split into smaller substeps for better numerical accuracy
        n_substeps = max(1, int(dt_step / 0.01))
        dt_sub = dt_step / n_substeps
        
        for _ in range(n_substeps):
            # Compute current decoherence functional
            d_func = decoherence.evaluate()
            
            # Update wavefunction using the local decoherence value at each point
            evolving_wf.psi *= np.exp(-gamma * dt_sub * d_func)
            
            # Renormalize to ensure proper quantum state
            evolving_wf.normalize()
            
            # Update decoherence functional with new wavefunction
            decoherence = DecoherenceFunctional(evolving_wf)
        
        # Record results
        results['norm'][i] = evolving_wf.get_norm()
        results['decoherence'][i] = decoherence.total()
        results['wavefunctions'].append(evolving_wf.copy())
    
    return results 