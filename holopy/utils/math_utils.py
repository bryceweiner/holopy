"""
Mathematical Utility Functions for HoloPy.

This module provides mathematical utility functions for tensor operations,
numerical derivatives, and specialized calculations used throughout HoloPy.
"""

import numpy as np
import logging
from typing import Optional, Union, Callable, Tuple, List, Dict

# Setup logging
logger = logging.getLogger(__name__)

def tensor_contraction(tensor: np.ndarray, indices: Tuple[int, int]) -> np.ndarray:
    """
    Perform tensor contraction over specified indices.
    
    Args:
        tensor (np.ndarray): Input tensor
        indices (Tuple[int, int]): Pair of indices to contract
        
    Returns:
        np.ndarray: Contracted tensor
    """
    # Validate input
    if len(indices) != 2:
        raise ValueError("Exactly two indices must be specified for contraction")
    
    # Ensure indices are sorted in descending order to avoid shifting issues
    i, j = sorted(indices, reverse=True)
    
    # Check if indices are valid
    if i >= tensor.ndim or j >= tensor.ndim:
        raise ValueError(f"Contraction indices {indices} out of bounds for tensor with {tensor.ndim} dimensions")
    
    # Check if indices are different
    if i == j:
        raise ValueError(f"Contraction indices must be different, got {indices}")
    
    # Perform contraction using numpy's trace function with appropriate axis permutation
    # Move the contraction indices to the first two positions
    perm = list(range(tensor.ndim))
    perm.pop(i)
    perm.pop(j)
    perm = [i, j] + perm
    
    # Permute the tensor
    permuted = np.transpose(tensor, perm)
    
    # Take trace over the first two dimensions
    contracted = np.trace(permuted, axis1=0, axis2=1)
    
    logger.debug(f"Contracted tensor of shape {tensor.shape} over indices {indices} to shape {contracted.shape}")
    return contracted

def numerical_gradient(
    function: Callable[[np.ndarray], Union[float, np.ndarray]],
    point: np.ndarray,
    h: float = 1e-5
) -> np.ndarray:
    """
    Compute numerical gradient of a function at a point.
    
    This function supports both scalar-valued and vector-valued functions.
    For vector-valued functions, it returns the Jacobian matrix of first derivatives.
    
    Args:
        function (Callable): Function to differentiate
        point (np.ndarray): Point at which to compute the gradient
        h (float, optional): Step size for finite difference
        
    Returns:
        np.ndarray: Gradient vector or Jacobian matrix
    """
    # Get dimensionality of input
    n = len(point)
    
    # Evaluate the function at the point to determine output dimension
    f0 = function(point)
    
    # Check if the function returns a scalar or a vector
    if np.isscalar(f0) or (isinstance(f0, np.ndarray) and f0.shape == ()):
        # Scalar function case
        # Initialize gradient
        gradient = np.zeros(n)
        
        # Compute gradient components
        for i in range(n):
            # Forward point
            point_forward = point.copy()
            point_forward[i] += h
            f_forward = function(point_forward)
            
            # Backward point
            point_backward = point.copy()
            point_backward[i] -= h
            f_backward = function(point_backward)
            
            # Central difference
            gradient[i] = (f_forward - f_backward) / (2 * h)
        
        logger.debug(f"Computed numerical gradient at point {point}")
        return gradient
    else:
        # Vector function case - calculate Jacobian
        # Determine output dimension
        m = len(f0)
        
        # Initialize Jacobian matrix
        jacobian = np.zeros((m, n))
        
        # Compute Jacobian components using central differences
        for i in range(n):
            # Forward point
            point_forward = point.copy()
            point_forward[i] += h
            f_forward = function(point_forward)
            
            # Backward point
            point_backward = point.copy()
            point_backward[i] -= h
            f_backward = function(point_backward)
            
            # Central difference for each output component
            for j in range(m):
                jacobian[j, i] = (f_forward[j] - f_backward[j]) / (2 * h)
        
        logger.debug(f"Computed numerical Jacobian at point {point}")
        return jacobian

def numerical_hessian(
    function: Callable[[np.ndarray], float],
    point: np.ndarray,
    h: float = 1e-5
) -> np.ndarray:
    """
    Compute numerical Hessian matrix of a function at a point.
    
    Args:
        function (Callable): Function to differentiate
        point (np.ndarray): Point at which to compute the Hessian
        h (float, optional): Step size for finite difference
        
    Returns:
        np.ndarray: Hessian matrix
    """
    # Get dimensionality
    n = len(point)
    
    # Initialize Hessian
    hessian = np.zeros((n, n))
    
    # Original function value
    f0 = function(point)
    
    # Compute Hessian components
    for i in range(n):
        for j in range(i, n):  # Exploit symmetry
            # Diagonal case: second derivative
            if i == j:
                # Forward point
                point_forward = point.copy()
                point_forward[i] += h
                f_forward = function(point_forward)
                
                # Backward point
                point_backward = point.copy()
                point_backward[i] -= h
                f_backward = function(point_backward)
                
                # Second derivative approximation
                hessian[i, i] = (f_forward - 2 * f0 + f_backward) / h**2
            
            # Off-diagonal case: mixed partial derivatives
            else:
                # Forward-forward point
                point_ff = point.copy()
                point_ff[i] += h
                point_ff[j] += h
                f_ff = function(point_ff)
                
                # Forward-backward point
                point_fb = point.copy()
                point_fb[i] += h
                point_fb[j] -= h
                f_fb = function(point_fb)
                
                # Backward-forward point
                point_bf = point.copy()
                point_bf[i] -= h
                point_bf[j] += h
                f_bf = function(point_bf)
                
                # Backward-backward point
                point_bb = point.copy()
                point_bb[i] -= h
                point_bb[j] -= h
                f_bb = function(point_bb)
                
                # Mixed derivative approximation
                hessian[i, j] = (f_ff - f_fb - f_bf + f_bb) / (4 * h**2)
                hessian[j, i] = hessian[i, j]  # Symmetry
    
    logger.debug(f"Computed numerical Hessian at point {point}")
    return hessian

def spatial_complexity(
    wavefunction: Callable[[np.ndarray], complex],
    points: np.ndarray,
    h: float = 1e-5
) -> np.ndarray:
    """
    Compute spatial complexity |∇ψ|² of a wavefunction at given points.
    
    Args:
        wavefunction (Callable): Wavefunction to evaluate
        points (np.ndarray): Points at which to compute complexity, shape (n_points, dim)
        h (float, optional): Step size for finite difference
        
    Returns:
        np.ndarray: Spatial complexity at each point
    """
    # Get number of points and dimensionality
    n_points, dim = points.shape
    
    # Initialize complexity array
    complexity = np.zeros(n_points)
    
    # Compute complexity at each point
    for i, point in enumerate(points):
        # Initialize sum of squared gradients
        grad_squared_sum = 0.0
        
        # Compute gradient components
        for d in range(dim):
            # Forward point
            point_forward = point.copy()
            point_forward[d] += h
            psi_forward = wavefunction(point_forward)
            
            # Backward point
            point_backward = point.copy()
            point_backward[d] -= h
            psi_backward = wavefunction(point_backward)
            
            # Central difference
            grad_d = (psi_forward - psi_backward) / (2 * h)
            
            # Add squared magnitude to sum
            grad_squared_sum += abs(grad_d)**2
        
        # Store complexity
        complexity[i] = grad_squared_sum
    
    logger.debug(f"Computed spatial complexity at {n_points} points")
    return complexity

def metrics_equivalent(
    metric1: np.ndarray,
    metric2: np.ndarray,
    tolerance: float = 1e-6
) -> bool:
    """
    Check if two metric tensors are equivalent within a tolerance.
    
    Args:
        metric1 (np.ndarray): First metric tensor
        metric2 (np.ndarray): Second metric tensor
        tolerance (float, optional): Relative tolerance for comparison
        
    Returns:
        bool: True if metrics are equivalent
    """
    # Check shapes
    if metric1.shape != metric2.shape:
        logger.warning(f"Metrics have different shapes: {metric1.shape} vs {metric2.shape}")
        return False
    
    # Compute relative difference
    norm1 = np.linalg.norm(metric1)
    norm2 = np.linalg.norm(metric2)
    
    # Avoid division by zero
    if norm1 == 0 and norm2 == 0:
        return True
    elif norm1 == 0 or norm2 == 0:
        return False
    
    # Compute relative Frobenius norm of the difference
    rel_diff = np.linalg.norm(metric1 - metric2) / max(norm1, norm2)
    
    # Check if below tolerance
    equivalent = rel_diff <= tolerance
    
    if equivalent:
        logger.debug(f"Metrics are equivalent with relative difference {rel_diff}")
    else:
        logger.debug(f"Metrics differ with relative difference {rel_diff} > {tolerance}")
    
    return equivalent

def gamma_matrix(index: int, metric: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute gamma matrix for a given index.
    
    In the holographic framework, gamma matrices represent the structure
    of the E8×E8 heterotic algebra projected to spacetime.
    
    Args:
        index (int): Spacetime index (0-3)
        metric (np.ndarray, optional): Metric tensor, if None use Minkowski
        
    Returns:
        np.ndarray: Gamma matrix
    """
    # Default to Minkowski metric
    if metric is None:
        metric = np.diag([1.0, -1.0, -1.0, -1.0])
    
    # Standard Dirac gamma matrices in the chiral representation
    if index == 0:
        gamma = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=complex)
    elif index == 1:
        gamma = np.array([
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [-1, 0, 0, 0]
        ], dtype=complex)
    elif index == 2:
        gamma = np.array([
            [0, 0, 0, -1j],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [-1j, 0, 0, 0]
        ], dtype=complex)
    elif index == 3:
        gamma = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, -1],
            [-1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=complex)
    else:
        raise ValueError(f"Invalid gamma matrix index: {index}")
    
    # Adjust for curved spacetime if needed
    if not np.allclose(metric, np.diag([1.0, -1.0, -1.0, -1.0])):
        # This is a simplified model - in a real implementation, this would
        # involve a more sophisticated calculation
        # We just scale the gamma matrix by the corresponding metric component
        gamma *= np.sqrt(abs(metric[index, index]))
    
    return gamma

def killing_form_e8(roots: np.ndarray) -> np.ndarray:
    """
    Compute the Killing form for E8 algebra from root vectors.
    
    Args:
        roots (np.ndarray): Root vectors, shape (240, 8)
        
    Returns:
        np.ndarray: Killing form, shape (248, 248)
    """
    # Number of roots
    n_roots = len(roots)
    
    # E8 has 248-dimensional Lie algebra: 240 roots + 8 Cartan generators
    dim = 248
    
    # Initialize Killing form
    killing_form = np.zeros((dim, dim))
    
    # Set Cartan subalgebra (first 8x8 block)
    # The Killing form is proportional to the identity on the Cartan subalgebra
    killing_form[:8, :8] = 60 * np.eye(8)
    
    # Set root-root blocks based on the root system structure
    for i, alpha in enumerate(roots):
        for j, beta in enumerate(roots):
            # Check if the sum is a root
            sum_is_root = any(np.allclose(alpha + beta, gamma) for gamma in roots)
            
            # Set the Killing form value based on the structure constants
            if sum_is_root:
                killing_form[8 + i, 8 + j] = 1.0
    
    # The Killing form must be symmetric
    killing_form = 0.5 * (killing_form + killing_form.T)
    
    logger.debug(f"Computed Killing form for E8 with shape {killing_form.shape}")
    return killing_form 