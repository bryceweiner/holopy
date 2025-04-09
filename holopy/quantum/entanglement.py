"""
Quantum Entanglement Module for HoloPy.

This module implements quantum entanglement calculations in the holographic framework,
including the holographic bound on entanglement entropy and information processing rates.
It implements the constraint 𝒯(𝒮) ≤ γ · S_ent where 𝒯(𝒮) is the transformation rate
of quantum state 𝒮, and S_ent is its entanglement entropy.
"""

import numpy as np
from typing import Callable, List, Tuple, Dict, Optional, Union, Any
import scipy.sparse as sp
import scipy.linalg as la
import logging
from holopy.constants.physical_constants import PhysicalConstants, PHYSICAL_CONSTANTS

# Configure logging
logger = logging.getLogger(__name__)

def entanglement_entropy(density_matrix: np.ndarray, subsystem_dims: Tuple[int, int]) -> float:
    """
    Calculate the entanglement entropy of a bipartite quantum system.
    
    The entanglement entropy is defined as S_ent = -Tr(ρ_A log ρ_A) where ρ_A is
    the reduced density matrix of subsystem A.
    
    Args:
        density_matrix: The density matrix of the full system
        subsystem_dims: Dimensions of the two subsystems (d_A, d_B)
        
    Returns:
        float: Entanglement entropy in units of nats (natural logarithm)
    """
    # Extract dimensions
    d_A, d_B = subsystem_dims
    
    # Reshape the density matrix to a tensor with 4 indices
    # ρ_{ij,kl} where i,k are indices for system A and j,l are indices for system B
    rho_tensor = density_matrix.reshape(d_A, d_B, d_A, d_B)
    
    # Take the partial trace over system B to get the reduced density matrix ρ_A
    rho_A = np.zeros((d_A, d_A), dtype=complex)
    for i in range(d_A):
        for k in range(d_A):
            for j in range(d_B):
                rho_A[i, k] += rho_tensor[i, j, k, j]
    
    # Compute eigenvalues of ρ_A
    eigenvalues = la.eigvalsh(rho_A)
    
    # Keep only positive eigenvalues (numerical errors might give small negative values)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    
    # Compute entropy: S = -∑λ_i log λ_i
    entropy = -np.sum(eigenvalues * np.log(eigenvalues))
    
    return entropy

def mutual_information(density_matrix: np.ndarray, subsystem_dims: Tuple[int, int]) -> float:
    """
    Calculate the quantum mutual information between two subsystems.
    
    The mutual information is defined as I(A:B) = S(A) + S(B) - S(A,B) where
    S is the von Neumann entropy.
    
    Args:
        density_matrix: The density matrix of the full system
        subsystem_dims: Dimensions of the two subsystems (d_A, d_B)
        
    Returns:
        float: Mutual information in units of nats
    """
    # Extract dimensions
    d_A, d_B = subsystem_dims
    
    # Reshape the density matrix to a tensor with 4 indices
    rho_tensor = density_matrix.reshape(d_A, d_B, d_A, d_B)
    
    # Compute the reduced density matrix ρ_A
    rho_A = np.zeros((d_A, d_A), dtype=complex)
    for i in range(d_A):
        for k in range(d_A):
            for j in range(d_B):
                rho_A[i, k] += rho_tensor[i, j, k, j]
    
    # Compute the reduced density matrix ρ_B
    rho_B = np.zeros((d_B, d_B), dtype=complex)
    for j in range(d_B):
        for l in range(d_B):
            for i in range(d_A):
                rho_B[j, l] += rho_tensor[i, j, i, l]
    
    # Compute eigenvalues
    eig_AB = la.eigvalsh(density_matrix)
    eig_A = la.eigvalsh(rho_A)
    eig_B = la.eigvalsh(rho_B)
    
    # Filter out zero eigenvalues (to avoid log(0))
    eig_AB = eig_AB[eig_AB > 1e-12]
    eig_A = eig_A[eig_A > 1e-12]
    eig_B = eig_B[eig_B > 1e-12]
    
    # Compute entropies
    S_AB = -np.sum(eig_AB * np.log(eig_AB))
    S_A = -np.sum(eig_A * np.log(eig_A))
    S_B = -np.sum(eig_B * np.log(eig_B))
    
    # Compute mutual information
    mutual_info = S_A + S_B - S_AB
    
    return mutual_info

def max_entanglement_rate(entropy: float) -> float:
    """
    Calculate the maximum entanglement generation rate based on the holographic bound.
    
    In the holographic framework, the transformation rate of a quantum state is bounded by
    𝒯(𝒮) ≤ γ · S_ent, where γ is the fundamental information processing rate.
    
    Args:
        entropy (float): Entanglement entropy in nats
        
    Returns:
        float: Maximum transformation/entanglement rate in transformations per second
    """
    # Get the fundamental information processing rate
    gamma = PHYSICAL_CONSTANTS.get_gamma()
    
    # Calculate the maximum rate
    max_rate = gamma * entropy
    
    return max_rate

def area_law_bound(area: float, gamma: Optional[float] = None) -> float:
    """
    Calculate the holographic bound on entanglement entropy based on area.
    
    According to the holographic principle, the maximum entropy of a region
    is proportional to the area of its boundary, not its volume.
    
    Args:
        area (float): Area of the boundary in square meters
        gamma (float, optional): Information processing rate. If None, use default.
        
    Returns:
        float: Maximum entropy in nats
    """
    # Get constants
    if gamma is None:
        gamma = PHYSICAL_CONSTANTS.get_gamma()
    
    pc = PhysicalConstants()
    
    # Calculate the Planck area
    planck_area = pc.planck_length**2
    
    # Calculate the maximum entropy
    # The maximum entropy is proportional to the area in Planck units
    max_entropy = area / (4 * planck_area)
    
    return max_entropy

def create_bell_state(dims: Tuple[int, int] = (2, 2)) -> np.ndarray:
    """
    Create a maximally entangled Bell state.
    
    Args:
        dims (Tuple[int, int]): Dimensions of the two subsystems
        
    Returns:
        np.ndarray: Density matrix of the Bell state
    """
    # For now, we'll implement this for qubits only
    if dims != (2, 2):
        raise NotImplementedError("Bell states for dimensions other than (2,2) not implemented yet")
    
    # Create the state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    psi = np.zeros(4)
    psi[0] = 1.0 / np.sqrt(2)  # |00⟩
    psi[3] = 1.0 / np.sqrt(2)  # |11⟩
    
    # Compute the density matrix ρ = |Φ⁺⟩⟨Φ⁺|
    rho = np.outer(psi, psi.conj())
    
    return rho

def max_information_flow(entropy: float, time: float) -> float:
    """
    Calculate the maximum amount of information that can flow between
    two entangled subsystems in a given time.
    
    Args:
        entropy (float): Entanglement entropy in nats
        time (float): Time interval in seconds
        
    Returns:
        float: Maximum information flow in nats
    """
    # Get the fundamental information processing rate
    gamma = PHYSICAL_CONSTANTS.get_gamma()
    
    # The maximum information flow is limited by the entanglement rate
    # and the available time
    max_flow = gamma * entropy * time
    
    return max_flow

def holographic_entanglement_constraint(transformation_rate: float, entropy: float) -> bool:
    """
    Check if a transformation rate satisfies the holographic constraint.
    
    The holographic constraint is: 𝒯(𝒮) ≤ γ · S_ent
    
    Args:
        transformation_rate (float): Rate of quantum state transformation in transformations per second
        entropy (float): Entanglement entropy in nats
        
    Returns:
        bool: True if the constraint is satisfied, False otherwise
    """
    # Get the fundamental information processing rate
    gamma = PHYSICAL_CONSTANTS.get_gamma()
    
    # Check if the constraint is satisfied
    return transformation_rate <= gamma * entropy 