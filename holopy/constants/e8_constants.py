"""
E8 Constants for HoloPy.

This module defines constants derived from the E8×E8 heterotic structure,
as described in the holographic gravity framework.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union

# Setup logging
logger = logging.getLogger(__name__)

class E8Constants:
    """
    Class for constants derived from the E8×E8 heterotic structure.
    
    This class provides access to constants and properties of the E8 Lie group
    and the E8×E8 heterotic string theory that are relevant for holographic
    physics.
    
    Attributes:
        root_count (int): Number of roots in E8 (240)
        e8e8_root_count (int): Number of roots in E8×E8 (480)
        dimension (int): Dimension of E8 Lie algebra (248)
        e8e8_dimension (int): Dimension of E8×E8 Lie algebra (496)
        rank (int): Rank of the E8 Lie algebra (8)
        clustering_coefficient (float): Clustering coefficient C(G)
    """
    
    def __init__(self):
        """Initialize E8 constants."""
        # E8 root system properties
        self.root_count = 240  # Number of roots in E8
        self.e8e8_root_count = 480  # Number of roots in E8×E8
        
        # E8 rank (number of simple roots)
        self.rank = 8  # Rank of the E8 Lie algebra
        
        # Lie algebra dimensions
        self.dimension = 248  # Dimension of E8 Lie algebra (240 roots + 8 Cartan generators)
        self.e8e8_dimension = 496  # Dimension of E8×E8 Lie algebra
        
        # Special angles in E8
        self.min_rotation_angle = np.pi / 120  # Minimal rotation angle in E8 root space (radians)
        
        # Clustering coefficient with physical significance
        # This is crucial for holographic framework and Hubble tension
        self.clustering_coefficient = 0.78125  # C(G) ≈ 0.78125 (exact value is 25/32)
        
        # E8×E8 derived constants
        
        # Information processing rate (γ)
        # γ = (2π / 240²) × (1/t_P)
        # where t_P is the Planck time
        # This will be computed when needed using get_gamma() method
        
        # Information-spacetime conversion factor
        # κ(π) = π^4/24
        self.kappa_pi = np.pi**4 / 24  # dimensionless
        
        # The 2/π ratio with physical significance
        self.two_pi_ratio = 2 / np.pi  # dimensionless
        
        logger.debug("E8Constants initialized")
    
    def get_clustering_coefficient(self) -> float:
        """
        Get the clustering coefficient C(G) of the E8×E8 heterotic structure.
        
        The clustering coefficient describes the connectivity structure of the
        E8×E8 framework and relates to the Hubble tension.
        
        Returns:
            float: Clustering coefficient C(G) ≈ 0.78125
        """
        return self.clustering_coefficient
    
    def get_kappa_pi(self) -> float:
        """
        Get the information-spacetime conversion factor κ(π).
        
        This dimensionless constant relates information content in
        16-dimensional and 4-dimensional Hilbert spaces.
        
        Returns:
            float: Information-spacetime conversion factor κ(π)
        """
        return self.kappa_pi
    
    def get_2pi_ratio(self) -> float:
        """
        Get the 2/π ratio with physical significance.
        
        This ratio represents the optimal balance between information
        locality and non-locality in quantum systems.
        
        Returns:
            float: The 2/π ratio (approximately 0.6366)
        """
        return self.two_pi_ratio
    
    def get_e8_roots(self) -> np.ndarray:
        """
        Generate the 240 root vectors of E8.
        
        This function constructs the 240 root vectors of the E8 Lie algebra
        using the standard construction.
        
        Returns:
            np.ndarray: Array of shape (240, 8) containing the root vectors
        """
        # Initialize an empty list to store root vectors
        roots = []
        
        # Construction 1: Permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
        # There are 8 choose 2 = 28 ways to place two non-zero entries,
        # and 2^2 = 4 ways to choose the signs, giving 28 * 4 = 112 roots
        for i in range(8):
            for j in range(i + 1, 8):
                for si in [-1, 1]:
                    for sj in [-1, 1]:
                        root = np.zeros(8)
                        root[i] = si
                        root[j] = sj
                        roots.append(root)
        
        # Construction 2: (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2)
        # with an even number of minus signs
        # There are 2^7 = 128 such vectors (the 8th sign is determined)
        for i in range(2**7):
            # Convert i to binary and count the number of 1's
            # This represents the positions where we place a -1/2
            binary = format(i, '07b')  # 7-bit binary representation
            signs = [int(bit) for bit in binary]
            
            # Determine the 8th sign to ensure an even number of minus signs
            if sum(signs) % 2 == 0:
                signs.append(0)  # Even number of 1's, add a 0
            else:
                signs.append(1)  # Odd number of 1's, add a 1
            
            # Create the root vector
            root = np.array([0.5 if sign == 0 else -0.5 for sign in signs])
            roots.append(root)
        
        # Convert list of roots to numpy array
        roots_array = np.array(roots)
        
        # Validate that we have 240 roots
        assert len(roots_array) == 240, f"Expected 240 roots, got {len(roots_array)}"
        
        logger.debug(f"Generated {len(roots_array)} E8 root vectors")
        return roots_array
    
    def compute_root_angles(self) -> Dict[str, int]:
        """
        Compute the distribution of angles between root vectors in E8.
        
        Returns:
            Dict[str, int]: Dictionary mapping angle descriptions to counts
        """
        # Get the root vectors
        roots = self.get_e8_roots()
        
        # Initialize dictionary to store angle counts
        angle_counts = {
            "0_degrees": 0,        # 0° (same root)
            "60_degrees": 0,       # 60°
            "90_degrees": 0,       # 90°
            "120_degrees": 0,      # 120°
            "180_degrees": 0,      # 180° (negative of a root)
        }
        
        # Compute angles between all pairs of roots
        n_roots = len(roots)
        for i in range(n_roots):
            for j in range(i, n_roots):
                # Compute the dot product
                dot_product = np.dot(roots[i], roots[j])
                
                # Classify the angle based on the dot product
                if i == j:
                    angle_counts["0_degrees"] += 1  # Same root
                elif np.isclose(dot_product, 1.0):
                    angle_counts["0_degrees"] += 1  # Same direction
                elif np.isclose(dot_product, 0.5):
                    angle_counts["60_degrees"] += 1  # 60°
                elif np.isclose(dot_product, 0.0):
                    angle_counts["90_degrees"] += 1  # 90°
                elif np.isclose(dot_product, -0.5):
                    angle_counts["120_degrees"] += 1  # 120°
                elif np.isclose(dot_product, -1.0):
                    angle_counts["180_degrees"] += 1  # 180°
        
        logger.debug(f"Computed E8 root angle distribution: {angle_counts}")
        return angle_counts
    
    def compute_root_system_projections(self, projection_dims: int = 2) -> np.ndarray:
        """
        Compute projections of the E8 root system for visualization.
        
        Args:
            projection_dims (int, optional): Number of dimensions to project to
            
        Returns:
            np.ndarray: Projected root vectors, shape (240, projection_dims)
        """
        # Ensure projection dimensions are valid
        if projection_dims < 2 or projection_dims > 8:
            raise ValueError("Projection dimensions must be between 2 and 8")
        
        # Get the root vectors
        roots = self.get_e8_roots()
        
        # Create projection matrix - use first projection_dims components
        projection_matrix = np.eye(8, projection_dims)
        
        # Project roots
        projected_roots = roots @ projection_matrix
        
        logger.debug(f"Computed {projection_dims}-dimensional projections of E8 root system")
        return projected_roots
    
    def get_cartan_matrix(self) -> np.ndarray:
        """
        Get the Cartan matrix of E8.
        
        The Cartan matrix encodes the structure of a Lie algebra in terms of
        its simple roots.
        
        Returns:
            np.ndarray: 8x8 Cartan matrix of E8
        """
        # Cartan matrix of E8 in the standard presentation with Dynkin diagram:
        #    1
        #    |
        # 0--2--3--4--5--6--7
        
        # Initialize with zeros
        cartan = np.zeros((8, 8), dtype=int)
        
        # Diagonal elements (always 2)
        np.fill_diagonal(cartan, 2)
        
        # Off-diagonal elements from Dynkin diagram
        # Node connections: (0,2), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7)
        connections = [(0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
        
        for i, j in connections:
            cartan[i, j] = -1
            cartan[j, i] = -1
        
        logger.debug("Constructed E8 Cartan matrix")
        return cartan
    
    def get_distance_matrix(self) -> np.ndarray:
        """
        Compute the distance matrix between E8 roots.
        
        The distance here is defined as the number of simple reflections needed
        to transform one root into another.
        
        Returns:
            np.ndarray: 240x240 distance matrix
        """
        # Get the 240 roots
        roots = self.get_e8_roots()
        n_roots = len(roots)
        
        # Initialize distance matrix
        distance_matrix = np.zeros((n_roots, n_roots), dtype=int)
        
        # Compute distances based on simple dot products and norms
        # This is a simplified approximation
        for i in range(n_roots):
            for j in range(i, n_roots):
                # Compute dot product
                dot_product = np.dot(roots[i], roots[j])
                
                # Define a distance measure
                # Roots with smaller dot products are "farther apart"
                # This is a heuristic and not a rigorous distance in the Lie algebra
                if np.isclose(dot_product, 1.0):  # Same or parallel roots
                    d = 0
                elif np.isclose(dot_product, 0.5):  # 60° angle
                    d = 1
                elif np.isclose(dot_product, 0.0):  # 90° angle
                    d = 2
                elif np.isclose(dot_product, -0.5):  # 120° angle
                    d = 3
                elif np.isclose(dot_product, -1.0):  # 180° angle (opposite roots)
                    d = 4
                else:
                    # This shouldn't happen for E8 roots
                    d = 5
                
                # Set the distance
                distance_matrix[i, j] = d
                distance_matrix[j, i] = d  # Symmetry
        
        logger.debug("Computed E8 root distance matrix")
        return distance_matrix
    
    def compute_correction(self, k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the E8×E8 correction factor for the power spectrum.
        
        This method computes the correction to the power spectrum due to 
        E8×E8 heterotic structure effects, which modifies the standard 
        power spectrum according to holographic principles.
        
        Args:
            k (Union[float, np.ndarray]): Wavenumber(s) in h/Mpc
            
        Returns:
            Union[float, np.ndarray]: Correction factor(s) for the power spectrum
        """
        # Get the clustering coefficient
        C = self.get_clustering_coefficient()
        
        # Get the 2/π ratio
        two_pi = self.get_2pi_ratio()
        
        # Get the information processing rate (γ)
        gamma = 1.89e-29  # s^-1, from observed CMB E-mode transitions
        
        # Characteristic scale related to holographic information bounds
        k_holo = 0.2  # h/Mpc, scale where holographic effects become significant
        
        # Compute correction factor
        # The correction scales with k according to a function that preserves
        # large-scale behavior but suppresses power at small scales
        if isinstance(k, np.ndarray):
            # Vectorized computation for arrays
            correction = np.ones_like(k)
            
            # Apply scale-dependent correction based on clustering coefficient
            scale_factor = np.exp(-C * (k / k_holo)**two_pi)
            
            # Apply transition effect near characteristic scales from holographic theory
            transition = 1.0 - gamma * np.log(1.0 + k / k_holo)
            transition = np.clip(transition, 0.8, 1.0)  # Limit correction strength
            
            correction = scale_factor * transition
        else:
            # Scalar computation
            scale_factor = np.exp(-C * (k / k_holo)**two_pi)
            transition = 1.0 - gamma * np.log(1.0 + k / k_holo)
            transition = max(0.8, min(1.0, transition))  # Limit correction strength
            correction = scale_factor * transition
        
        logger.debug(f"Computed E8×E8 correction factor for power spectrum at k={k}: {correction}")
        return correction

# Single global instance for convenience
E8_CONSTANTS = E8Constants()

# E8×E8 heterotic structure constants (global instance)
# This is the combined structure used in the holographic framework
E8E8_CONSTANTS = E8Constants() 