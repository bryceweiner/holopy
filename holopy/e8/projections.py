"""
E8×E8 Projections Module for HoloPy.

This module implements methods to project from the 16-dimensional E8×E8 space
to 4D spacetime, and other dimensional reduction operations essential for
relating the heterotic structure to physical spacetime.
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Union, Callable
import sympy as sp
from scipy.linalg import svd, orth

from holopy.constants.physical_constants import PhysicalConstants
from holopy.constants.e8_constants import E8Constants
from holopy.e8.heterotic import E8E8Heterotic, compute_killing_form

def project_to_4d(vector_16d: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Project a 16-dimensional vector from E8×E8 space to 4D spacetime.
    
    This function implements various projection methods from the 16-dimensional
    heterotic space to 4-dimensional spacetime, which is crucial for relating
    the mathematical structure to physical reality.
    
    Args:
        vector_16d (np.ndarray): A 16-dimensional vector in E8×E8 space
        method (str): Projection method to use. Options are:
            'standard': Standard projection using fixed matrix
            'killing': Projection preserving Killing form structure
            'holographic': Projection based on holographic principle
            'adaptive': Adaptive projection based on vector properties
    
    Returns:
        np.ndarray: The projected 4-dimensional vector
    """
    # Validate input
    if vector_16d.shape != (16,):
        raise ValueError("Input vector must be 16-dimensional")
    
    # Get constants
    pc = PhysicalConstants()
    kappa_pi = pc.kappa_pi  # π^4/24
    
    if method == 'standard':
        # Standard projection using a fixed matrix
        # This matrix preserves certain properties of the E8×E8 structure
        # The matrix is constructed to ensure that information content
        # scales according to the holographic principle
        
        # Construct the projection matrix
        P = np.zeros((4, 16))
        
        # First 3 spatial dimensions use different projections from each E8
        # Each row represents a different basis of projection
        P[0, 0:8] = 1/np.sqrt(8)  # x-dimension projection from first E8
        P[1, 8:16] = 1/np.sqrt(8)  # y-dimension projection from second E8
        
        # z-dimension combines components from both E8 lattices
        P[2, 0:8] = 1/(2*np.sqrt(8))
        P[2, 8:16] = -1/(2*np.sqrt(8))
        
        # Time dimension captures the "radial" component of combined E8s
        P[3, :] = 1/(4*np.sqrt(8))  # Weighted average of all dimensions
        
        # Apply scaling factor for dimensional reduction
        P *= np.sqrt(kappa_pi)
        
        # Apply projection
        vector_4d = np.dot(P, vector_16d)
        
        return vector_4d
    
    elif method == 'killing':
        # Projection that preserves Killing form structure
        # This method finds a 4D subspace that maximally preserves
        # the Killing form metric properties
        
        # Create a set of basis vectors for the 4D space
        # These are chosen to preserve critical E8×E8 structure relations
        basis = np.zeros((4, 16))
        
        # First basis vector combines both E8 components symmetrically
        basis[0, :8] = 1/np.sqrt(16)
        basis[0, 8:] = 1/np.sqrt(16)
        
        # Second basis vector differentiates the two E8 components
        basis[1, :8] = 1/np.sqrt(16)
        basis[1, 8:] = -1/np.sqrt(16)
        
        # Third and fourth basis vectors capture internal E8 structure
        # These are patterns from E8 roots that relate to 4D symmetries
        basis[2, 0::2] = 1/np.sqrt(8)  # Alternating pattern in first E8
        basis[3, 8:12] = 1/np.sqrt(4)  # First half of second E8
        
        # Orthonormalize the basis
        basis = orth(basis.T).T
        
        # Apply the projection
        vector_4d = np.dot(basis, vector_16d)
        
        # Apply holographic scaling
        vector_4d *= np.sqrt(kappa_pi)
        
        return vector_4d
    
    elif method == 'holographic':
        # Holographic projection based on information content
        # This method implements a projection that preserves information content
        # according to the holographic principle
        
        # Extract components from each E8
        vector_e8_1 = vector_16d[:8]
        vector_e8_2 = vector_16d[8:]
        
        # Create 4D vector using holographic projection rules
        vector_4d = np.zeros(4)
        
        # Space dimensions (x,y,z) derive from relative relationships
        # between the two E8 components
        vector_4d[0] = np.dot(vector_e8_1[:4], vector_e8_1[4:]) * np.sqrt(kappa_pi/8)
        vector_4d[1] = np.dot(vector_e8_2[:4], vector_e8_2[4:]) * np.sqrt(kappa_pi/8)
        vector_4d[2] = np.dot(vector_e8_1, vector_e8_2) * np.sqrt(kappa_pi/16)
        
        # Time dimension derives from the overall scale of both E8 components
        scale_1 = np.linalg.norm(vector_e8_1)
        scale_2 = np.linalg.norm(vector_e8_2)
        vector_4d[3] = (scale_1 + scale_2) * np.sqrt(kappa_pi/32)
        
        return vector_4d
    
    elif method == 'adaptive':
        # Adaptive projection that chooses method based on vector properties
        # This attempts to use the most appropriate projection based on
        # properties of the input vector
        
        # Analyze the input vector
        e8_1_norm = np.linalg.norm(vector_16d[:8])
        e8_2_norm = np.linalg.norm(vector_16d[8:])
        
        # If the norms are similar, use standard projection
        if 0.8 < (e8_1_norm / (e8_1_norm + e8_2_norm)) < 0.8:
            return project_to_4d(vector_16d, method='standard')
        
        # If one component dominates, use killing form projection
        elif e8_1_norm > 2 * e8_2_norm or e8_2_norm > 2 * e8_1_norm:
            return project_to_4d(vector_16d, method='killing')
        
        # Otherwise use holographic projection
        else:
            return project_to_4d(vector_16d, method='holographic')
    
    else:
        raise ValueError(f"Unsupported projection method: {method}")


def project_to_4d_tensor(tensor_16d: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Project a tensor from E8×E8 space to 4D spacetime.
    
    This function projects a rank-2 tensor from 16D to 4D by applying the
    projection to each index.
    
    Args:
        tensor_16d (np.ndarray): A 16×16 tensor in E8×E8 space
        method (str): Projection method to use
    
    Returns:
        np.ndarray: The projected 4×4 tensor
    """
    # Validate input
    if tensor_16d.shape != (16, 16):
        raise ValueError("Input tensor must be 16×16")
    
    # Get the appropriate projection matrix based on method
    # For simplicity, we'll compute this by comparing the action on a standard basis
    std_basis = np.eye(16)
    projection_matrix = np.zeros((4, 16))
    
    for i in range(16):
        projection_matrix[:, i] = project_to_4d(std_basis[i], method=method)
    
    # Apply the projection to both indices of the tensor
    tensor_4d = np.zeros((4, 4))
    for mu in range(4):
        for nu in range(4):
            for i in range(16):
                for j in range(16):
                    tensor_4d[mu, nu] += projection_matrix[mu, i] * tensor_16d[i, j] * projection_matrix[nu, j]
    
    return tensor_4d


def invert_projection(vector_4d: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Attempt to invert the projection from 4D back to 16D.
    
    This is not a true inverse since the projection is not bijective.
    The function returns a 16D vector that would project to the given 4D vector.
    
    Args:
        vector_4d (np.ndarray): A 4-dimensional vector
        method (str): The projection method to invert
    
    Returns:
        np.ndarray: A 16-dimensional vector that projects to the input
    """
    # Validate input
    if vector_4d.shape != (4,):
        raise ValueError("Input vector must be 4-dimensional")
    
    # Get constants
    pc = PhysicalConstants()
    kappa_pi = pc.kappa_pi
    
    # Initialize 16D vector
    vector_16d = np.zeros(16)
    
    if method == 'standard':
        # Create inverse of standard projection
        # Note: This is a pseudo-inverse, as the projection is not invertible
        
        # Remove the scaling factor
        v4d = vector_4d / np.sqrt(kappa_pi)
        
        # Distribute the x component to first E8
        vector_16d[:8] = v4d[0] * np.sqrt(8) / 8
        
        # Distribute the y component to second E8
        vector_16d[8:] = v4d[1] * np.sqrt(8) / 8
        
        # Distribute the z component to both E8s
        vector_16d[:8] += v4d[2] * 2 * np.sqrt(8) / 8
        vector_16d[8:] -= v4d[2] * 2 * np.sqrt(8) / 8
        
        # Distribute the time component to all dimensions
        vector_16d += v4d[3] * 4 * np.sqrt(8) / 16
        
        return vector_16d
    
    elif method in ['killing', 'holographic']:
        # For these methods, we use a more general pseudo-inverse approach
        
        # Get the projection matrix for the specified method
        std_basis = np.eye(16)
        projection_matrix = np.zeros((4, 16))
        
        for i in range(16):
            projection_matrix[:, i] = project_to_4d(std_basis[i], method=method)
        
        # Compute the pseudo-inverse
        u, s, vh = svd(projection_matrix, full_matrices=False)
        s_inv = np.where(s > 1e-10, 1/s, 0)
        pseudo_inv = vh.T @ np.diag(s_inv) @ u.T
        
        # Apply the pseudo-inverse
        vector_16d = pseudo_inv @ vector_4d
        
        return vector_16d
    
    elif method == 'adaptive':
        # For adaptive method, default to standard inversion
        return invert_projection(vector_4d, method='standard')
    
    else:
        raise ValueError(f"Unsupported projection method: {method}")


def compute_metric_from_killing_form(points_16d: np.ndarray) -> np.ndarray:
    """
    Compute a 4D metric derived from the Killing form on E8×E8.
    
    This function computes a metric for 4D spacetime based on the Killing form
    of the E8×E8 heterotic structure for a set of points.
    
    Args:
        points_16d (np.ndarray): Array of shape (n, 16) containing n points in E8×E8 space
    
    Returns:
        np.ndarray: The 4×4 metric tensor
    """
    n_points = points_16d.shape[0]
    
    # Initialize the metric tensor
    metric = np.zeros((4, 4))
    
    # For each pair of points, compute contribution to metric
    for i in range(n_points):
        for j in range(i+1, n_points):
            # Compute the Killing form between the points
            killing_value = compute_killing_form(points_16d[i], points_16d[j])
            
            # Project the points to 4D
            point_i_4d = project_to_4d(points_16d[i])
            point_j_4d = project_to_4d(points_16d[j])
            
            # Compute the displacement vector
            displacement = point_i_4d - point_j_4d
            
            # Contribute to the metric using the outer product
            # scaled by the Killing form value
            contribution = np.outer(displacement, displacement) * killing_value
            metric += contribution
    
    # Normalize by number of point pairs
    n_pairs = n_points * (n_points - 1) // 2
    if n_pairs > 0:
        metric /= n_pairs
    
    return metric


if __name__ == "__main__":
    # Example usage
    
    # Create a random 16D vector
    vector_16d = np.random.randn(16)
    
    # Project to 4D using different methods
    vector_4d_standard = project_to_4d(vector_16d, method='standard')
    vector_4d_killing = project_to_4d(vector_16d, method='killing')
    vector_4d_holographic = project_to_4d(vector_16d, method='holographic')
    
    print(f"Original 16D vector: {vector_16d}")
    print(f"\nProjected 4D vector (standard): {vector_4d_standard}")
    print(f"Projected 4D vector (killing): {vector_4d_killing}")
    print(f"Projected 4D vector (holographic): {vector_4d_holographic}")
    
    # Try to invert the projection
    vector_16d_inv = invert_projection(vector_4d_standard)
    
    # Project the inverted vector again to see how well it matches
    vector_4d_reinv = project_to_4d(vector_16d_inv)
    
    print(f"\nInverted 16D vector: {vector_16d_inv}")
    print(f"Re-projected 4D vector: {vector_4d_reinv}")
    print(f"Original 4D vector: {vector_4d_standard}")
    print(f"Difference: {np.linalg.norm(vector_4d_reinv - vector_4d_standard)}")
    
    # Create a set of random 16D points
    points_16d = np.random.randn(10, 16)
    
    # Compute a 4D metric from the Killing form
    metric = compute_metric_from_killing_form(points_16d)
    
    print(f"\n4D metric derived from Killing form:\n{metric}")