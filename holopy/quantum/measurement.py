"""
Quantum Measurement Module for HoloPy.

This module implements quantum measurement operations in the holographic framework,
including measurement processes that incorporate the holographic decoherence effects.
"""

import numpy as np
from typing import Callable, List, Tuple, Dict, Optional, Union, Any
import scipy.sparse as sp
import logging
from holopy.constants.physical_constants import PHYSICAL_CONSTANTS
from holopy.quantum.modified_schrodinger import WaveFunction

# Configure logging
logger = logging.getLogger(__name__)

def quantum_measurement(wavefunction: WaveFunction, 
                      observable: Union[Callable[[WaveFunction], WaveFunction], np.ndarray],
                      num_samples: int = 1) -> np.ndarray:
    """
    Simulate a quantum measurement of an observable.
    
    In the holographic framework, measurement is a physical process that
    involves decoherence. This function simulates the outcome of measuring
    an observable on a quantum state.
    
    Args:
        wavefunction: The quantum state to measure
        observable: Either a function that applies the observable operator to the wavefunction,
                   or a matrix representation of the observable
        num_samples: Number of measurement samples to generate
        
    Returns:
        np.ndarray: Measurement outcome(s)
    """
    # For now, we'll implement the simplest case: measurement in the position basis
    
    # Calculate probability distribution
    prob_density = wavefunction.probability_density()
    
    # Normalize the probability distribution
    prob = prob_density / np.sum(prob_density)
    
    # Flatten for sampling
    prob_flat = prob.flatten()
    
    # Generate indices based on probability distribution
    indices = np.random.choice(len(prob_flat), size=num_samples, p=prob_flat)
    
    # Get the actual positions
    if isinstance(wavefunction.grid, list):
        # For multi-dimensional grid, we need to convert flattened index to multi-index
        result = []
        for idx in indices:
            # Convert flat index to multi-index
            multi_idx = np.unravel_index(idx, prob.shape)
            
            # Get the position
            pos = np.array([grid[multi_idx] for grid in wavefunction.grid])
            result.append(pos)
        
        return np.array(result)
    else:
        # 1D case is simpler
        return wavefunction.grid[indices]

def expectation_value(wavefunction: WaveFunction, 
                     observable: Union[Callable[[WaveFunction], WaveFunction], np.ndarray]) -> complex:
    """
    Calculate the expectation value of an observable.
    
    Args:
        wavefunction: The quantum state
        observable: Either a function that applies the observable operator to the wavefunction,
                   or a matrix representation of the observable
        
    Returns:
        complex: Expectation value ⟨ψ|A|ψ⟩
    """
    # If observable is a function
    if callable(observable):
        # Apply the observable to the wavefunction
        A_psi = observable(wavefunction)
        
        # Calculate ⟨ψ|A|ψ⟩
        if isinstance(wavefunction.grid, list):
            # Get the grid spacing in each dimension
            dx = []
            for d in range(wavefunction.dimension):
                x = wavefunction.grid[d]
                dx.append((x.max() - x.min()) / (x.shape[0] - 1))
            
            # Compute the volume element
            dV = np.prod(dx)
            
            # Integrate
            expectation = np.sum(np.conjugate(wavefunction.psi) * A_psi.psi) * dV
        else:
            # 1D case
            dx = (wavefunction.grid.max() - wavefunction.grid.min()) / (wavefunction.grid.size - 1)
            expectation = np.sum(np.conjugate(wavefunction.psi) * A_psi.psi) * dx
    
    # If observable is a matrix
    else:
        # Convert wavefunction to vector
        psi_vec = wavefunction.to_vector()
        
        # Apply the observable
        A_psi_vec = observable @ psi_vec
        
        # Calculate ⟨ψ|A|ψ⟩
        expectation = np.vdot(psi_vec, A_psi_vec)
    
    return expectation

def measurement_probability(wavefunction: WaveFunction, 
                          observable: Union[Callable[[WaveFunction], WaveFunction], np.ndarray],
                          value: Union[complex, np.ndarray]) -> float:
    """
    Calculate the probability of measuring a specific value of an observable.
    
    Args:
        wavefunction: The wavefunction to measure
        observable: A function that applies the observable operator to a wavefunction
                   or a matrix representation of the observable
        value: The eigenvalue to calculate the probability for
        
    Returns:
        float: Probability of measuring the specified value
    """
    # Full implementation for different types of observables
    
    # Case 1: Position measurement
    if isinstance(observable, Callable) and observable.__name__ == 'position_operator':
        # For position measurement, the probability is |ψ(x)|²
        prob_density = wavefunction.probability_density()
        
        # Find the closest grid point to the specified value
        if isinstance(wavefunction.grid, list):
            # Multi-dimensional case
            # Convert value to array if it's not already
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            
            # Find the closest grid point
            closest_idx = None
            min_distance = float('inf')
            
            # Iterate through all grid points
            grid_iterator = np.ndindex(prob_density.shape)
            for idx in grid_iterator:
                # Get the position of this grid point
                pos = np.array([grid[idx] for grid in wavefunction.grid])
                
                # Calculate distance to specified value
                distance = np.linalg.norm(pos - value)
                
                # Update if this is closer
                if distance < min_distance:
                    min_distance = distance
                    closest_idx = idx
            
            # Get the probability at the closest point
            probability = prob_density[closest_idx]
        else:
            # 1D case
            # Find the closest grid point
            closest_idx = np.abs(wavefunction.grid - value).argmin()
            
            # Get the probability at the closest point
            probability = prob_density[closest_idx]
        
        # Normalize to get a proper probability
        if isinstance(wavefunction.grid, list):
            # Get the grid spacing in each dimension
            dx = []
            for d in range(wavefunction.dimension):
                x = wavefunction.grid[d]
                dx.append((x.max() - x.min()) / (x.shape[0] - 1))
            
            # Compute the volume element
            dV = np.prod(dx)
            
            # Multiply by volume element to get probability
            probability *= dV
        else:
            # 1D case
            dx = (wavefunction.grid.max() - wavefunction.grid.min()) / (wavefunction.grid.size - 1)
            probability *= dx
    
    # Case 2: Observable with matrix representation
    elif isinstance(observable, np.ndarray):
        # Find eigenvalues and eigenvectors of the observable
        eigenvalues, eigenvectors = np.linalg.eigh(observable)
        
        # Find the eigenvector(s) corresponding to the specified eigenvalue
        # (within a small tolerance to account for numerical errors)
        tol = 1e-10
        mask = np.abs(eigenvalues - value) < tol
        
        if not np.any(mask):
            # No matching eigenvalue found
            return 0.0
        
        # Get the eigenvectors corresponding to the specified eigenvalue
        matching_eigenvectors = eigenvectors[:, mask]
        
        # Convert wavefunction to vector in the same basis
        if wavefunction.basis == 'position':
            psi_vec = wavefunction.to_vector()
        else:
            # If wavefunction is in a different basis, extra conversion might be needed
            # For now, assume it's already in the appropriate basis
            psi_vec = wavefunction.to_vector()
        
        # Calculate probability as sum of squared inner products
        probability = 0.0
        for i in range(matching_eigenvectors.shape[1]):
            eigenvec = matching_eigenvectors[:, i]
            inner_product = np.abs(np.vdot(eigenvec, psi_vec))**2
            probability += inner_product
    
    # Case 3: Observable with functional representation (other than position)
    else:
        # For a general quantum operator, we need to numerically find eigenstates
        # This is a more complex computation
        
        # First, convert the observable to matrix form by applying it to basis states
        if isinstance(wavefunction.grid, list):
            # Multi-dimensional case - get total grid size
            total_size = wavefunction.psi.size
        else:
            # 1D case
            total_size = wavefunction.grid.size
        
        # Create matrix representation of the observable
        observable_matrix = np.zeros((total_size, total_size), dtype=complex)
        
        # Function to create a basis state with 1 at position i
        def create_basis_state(i):
            state_vector = np.zeros(total_size, dtype=complex)
            state_vector[i] = 1.0
            return WaveFunction(
                grid=wavefunction.grid,
                values=state_vector.reshape(wavefunction.psi.shape),
                basis=wavefunction.basis
            )
        
        # Apply observable to each basis state to build the matrix
        for i in range(total_size):
            basis_state = create_basis_state(i)
            result_state = observable(basis_state)
            observable_matrix[:, i] = result_state.to_vector()
        
        # Now find eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(observable_matrix)
        
        # Find the eigenvector(s) corresponding to the specified eigenvalue
        tol = 1e-10
        mask = np.abs(eigenvalues - value) < tol
        
        if not np.any(mask):
            # No matching eigenvalue found within tolerance
            # Find the closest eigenvalue
            closest_idx = np.abs(eigenvalues - value).argmin()
            mask = np.abs(eigenvalues - eigenvalues[closest_idx]) < tol
        
        # Get the eigenvectors corresponding to the eigenvalue
        matching_eigenvectors = eigenvectors[:, mask]
        
        # Calculate the probability
        psi_vec = wavefunction.to_vector()
        probability = 0.0
        for i in range(matching_eigenvectors.shape[1]):
            eigenvec = matching_eigenvectors[:, i]
            inner_product = np.abs(np.vdot(eigenvec, psi_vec))**2
            probability += inner_product
    
    return float(probability)

def position_operator(wavefunction: WaveFunction, component: int = 0) -> WaveFunction:
    """
    Position operator: x̂ψ(x) = x·ψ(x)
    
    Args:
        wavefunction: The wavefunction to apply the operator to
        component: The position component to use (for multi-dimensional systems)
        
    Returns:
        WaveFunction: The result of applying the position operator
    """
    if isinstance(wavefunction.grid, list):
        # Multi-dimensional case
        if component >= len(wavefunction.grid):
            raise ValueError(f"Component {component} is out of bounds for {len(wavefunction.grid)}-dimensional grid")
        
        # Apply the position operator
        x_psi = wavefunction.grid[component] * wavefunction.psi
    else:
        # 1D case
        x_psi = wavefunction.grid * wavefunction.psi
    
    # Return as a new WaveFunction
    result = WaveFunction(
        grid=wavefunction.grid,
        values=x_psi,
        basis=wavefunction.basis
    )
    
    return result

def momentum_operator(wavefunction: WaveFunction, component: int = 0) -> WaveFunction:
    """
    Momentum operator: p̂ψ(x) = -iℏ∇ψ(x)
    
    Args:
        wavefunction: The wavefunction to apply the operator to
        component: The momentum component to use (for multi-dimensional systems)
        
    Returns:
        WaveFunction: The result of applying the momentum operator
    """
    # Get the reduced Planck constant
    h_bar = PHYSICAL_CONSTANTS.hbar
    
    # Compute the gradient
    gradient = wavefunction.compute_gradient()
    
    if isinstance(gradient, list):
        # Multi-dimensional case
        if component >= len(gradient):
            raise ValueError(f"Component {component} is out of bounds for {len(gradient)}-dimensional gradient")
        
        # Apply the momentum operator
        p_psi = -1j * h_bar * gradient[component]
    else:
        # 1D case
        p_psi = -1j * h_bar * gradient
    
    # Return as a new WaveFunction
    result = WaveFunction(
        grid=wavefunction.grid,
        values=p_psi,
        basis=wavefunction.basis
    )
    
    return result

def collapse_wavefunction(wavefunction: WaveFunction, 
                         outcome: np.ndarray, 
                         uncertainty: Optional[float] = None) -> WaveFunction:
    """
    Collapse a wavefunction after a measurement.
    
    Args:
        wavefunction: The wavefunction before measurement
        outcome: The measurement outcome (position)
        uncertainty: Measurement uncertainty (width of Gaussian collapse)
        
    Returns:
        WaveFunction: The collapsed wavefunction
    """
    # If uncertainty is not provided, use a default value
    if uncertainty is None:
        # For a position measurement, we can use the grid spacing as a reasonable default
        if isinstance(wavefunction.grid, list):
            # Use the minimum grid spacing
            dx_min = float('inf')
            for d in range(wavefunction.dimension):
                x = wavefunction.grid[d]
                dx = (x.max() - x.min()) / (x.shape[0] - 1)
                dx_min = min(dx_min, dx)
            
            uncertainty = dx_min
        else:
            # 1D case
            uncertainty = (wavefunction.grid.max() - wavefunction.grid.min()) / (wavefunction.grid.size - 1)
    
    # Create a new wavefunction centered at the measurement outcome
    if isinstance(wavefunction.grid, list):
        # Multi-dimensional case
        # Create a Gaussian centered at the outcome
        collapsed_psi = np.zeros_like(wavefunction.psi)
        
        # Iterate through all grid points
        grid_iterator = np.ndindex(collapsed_psi.shape)
        for idx in grid_iterator:
            # Get the position of this grid point
            pos = np.array([grid[idx] for grid in wavefunction.grid])
            
            # Calculate distance to outcome
            distance_squared = np.sum((pos - outcome)**2)
            
            # Gaussian with width determined by uncertainty
            collapsed_psi[idx] = np.exp(-distance_squared / (2 * uncertainty**2))
    else:
        # 1D case
        # Create a Gaussian centered at the outcome
        collapsed_psi = np.exp(-(wavefunction.grid - outcome)**2 / (2 * uncertainty**2))
    
    # Create a new wavefunction
    collapsed_wavefunction = WaveFunction(
        grid=wavefunction.grid,
        values=collapsed_psi,
        basis=wavefunction.basis
    )
    
    # Normalize the wavefunction
    collapsed_wavefunction.normalize()
    
    return collapsed_wavefunction 