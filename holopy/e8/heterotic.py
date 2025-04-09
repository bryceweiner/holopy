"""
E8×E8 Heterotic Structure Module for HoloPy.

This module implements the E8×E8 heterotic structure, which combines two E8 lattices
to form the foundational mathematical framework of the holographic universe theory.
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Union
import networkx as nx

from holopy.constants.e8_constants import E8Constants
from holopy.e8.root_system import RootSystem
from holopy.e8.lattice import E8Lattice

class E8E8Heterotic:
    """
    A class representing the E8×E8 heterotic structure.
    
    The E8×E8 heterotic structure combines two E8 lattices into a 16-dimensional
    structure that forms the mathematical foundation of the holographic universe theory.
    It has deep connections to string theory, where it appears as the gauge group
    of one of the heterotic string theories.
    
    Attributes:
        dimension (int): Dimension of the structure (16)
        lattice1 (E8Lattice): First E8 lattice component
        lattice2 (E8Lattice): Second E8 lattice component
        root_count (int): Total number of roots (480)
        heterotic_dimension (int): Dimension of the E8×E8 Lie algebra (496)
    """
    
    def __init__(self, max_points: int = 500):
        """
        Initialize the E8×E8 heterotic structure.
        
        Args:
            max_points (int): Maximum number of lattice points to generate for each E8 component
        """
        self.e8_constants = E8Constants()
        self.dimension = 16  # 8 + 8
        self.root_count = self.e8_constants.e8e8_root_count  # 240 + 240
        self.heterotic_dimension = self.e8_constants.e8e8_dimension  # 248 + 248
        
        # Create two E8 lattices
        self.lattice1 = E8Lattice(max_points=max_points)
        self.lattice2 = E8Lattice(max_points=max_points)
        
        # Generate combined lattice points
        self.generate_heterotic_points(max_points)
    
    def generate_heterotic_points(self, max_points: int):
        """
        Generate a set of points in the E8×E8 heterotic structure.
        
        This method generates points by combining points from the two E8 lattices,
        prioritizing points closest to the origin.
        
        Args:
            max_points (int): Maximum number of heterotic points to generate
        """
        # Get points from each E8 lattice
        points1 = self.lattice1.get_lattice_points()
        points2 = self.lattice2.get_lattice_points()
        
        # Create combined points by taking the Cartesian product
        # but limit the total number to max_points
        combined_points = []
        
        # Sort points by distance from origin
        dist1 = [np.linalg.norm(p) for p in points1]
        dist2 = [np.linalg.norm(p) for p in points2]
        idx1 = np.argsort(dist1)
        idx2 = np.argsort(dist2)
        
        # Create combined points, prioritizing those closest to the origin
        count = 0
        for i in idx1:
            if count >= max_points:
                break
            for j in idx2:
                if count >= max_points:
                    break
                combined_point = np.concatenate((points1[i], points2[j]))
                combined_points.append(combined_point)
                count += 1
        
        self.heterotic_points = np.array(combined_points)
    
    def get_heterotic_points(self) -> np.ndarray:
        """
        Get the generated heterotic points.
        
        Returns:
            np.ndarray: Array of generated points in the E8×E8 structure
        """
        return self.heterotic_points
    
    def project_to_e8(self, point: np.ndarray, component: int = 0) -> np.ndarray:
        """
        Project a 16-dimensional point onto one of the E8 components.
        
        Args:
            point (np.ndarray): A 16-dimensional point
            component (int): Which E8 component to project onto (0 or 1)
            
        Returns:
            np.ndarray: The 8-dimensional projection
        """
        if component == 0:
            return point[:8]
        else:
            return point[8:]
    
    def lift_from_e8(self, point1: np.ndarray, point2: np.ndarray) -> np.ndarray:
        """
        Combine two 8-dimensional points from E8 lattices into a heterotic point.
        
        Args:
            point1 (np.ndarray): Point from the first E8 lattice
            point2 (np.ndarray): Point from the second E8 lattice
            
        Returns:
            np.ndarray: The combined 16-dimensional point
        """
        return np.concatenate((point1, point2))
    
    def contains_point(self, point: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Check if a point belongs to the E8×E8 heterotic structure.
        
        A point is in the E8×E8 structure if its first 8 coordinates form a point
        in the first E8 lattice and its last 8 coordinates form a point in the
        second E8 lattice.
        
        Args:
            point (np.ndarray): The 16-dimensional point to check
            tolerance (float): Tolerance for floating-point comparisons
            
        Returns:
            bool: True if the point is in the E8×E8 structure, False otherwise
        """
        point1 = point[:8]
        point2 = point[8:]
        
        return self.lattice1.contains_point(point1, tolerance) and \
               self.lattice2.contains_point(point2, tolerance)
    
    def nearest_heterotic_point(self, point: np.ndarray) -> np.ndarray:
        """
        Find the nearest E8×E8 heterotic point to a given point.
        
        This is done by finding the nearest point in each E8 lattice
        and combining them.
        
        Args:
            point (np.ndarray): The 16-dimensional query point
            
        Returns:
            np.ndarray: The nearest heterotic point
        """
        point1 = point[:8]
        point2 = point[8:]
        
        nearest1 = self.lattice1.nearest_lattice_point(point1)
        nearest2 = self.lattice2.nearest_lattice_point(point2)
        
        return np.concatenate((nearest1, nearest2))
    
    def get_e8_components(self) -> Tuple[E8Lattice, E8Lattice]:
        """
        Get the two E8 lattice components.
        
        Returns:
            Tuple[E8Lattice, E8Lattice]: The two E8 lattices
        """
        return (self.lattice1, self.lattice2)
    
    def get_root_count(self) -> int:
        """
        Get the total number of roots in the E8×E8 structure.
        
        Returns:
            int: Total number of roots (480)
        """
        return self.root_count
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the E8×E8 heterotic structure.
        
        Returns:
            int: Dimension (16)
        """
        return self.dimension
    
    def get_lie_algebra_dimension(self) -> int:
        """
        Get the dimension of the E8×E8 Lie algebra.
        
        Returns:
            int: Dimension of the Lie algebra (496)
        """
        return self.heterotic_dimension
    
    def get_roots(self) -> np.ndarray:
        """
        Get all roots of the E8×E8 heterotic structure.
        
        This is an alias for get_all_roots() for API consistency.
        
        Returns:
            np.ndarray: Array of shape (480, 16) containing all root vectors
        """
        return self.get_all_roots()
    
    def get_all_roots(self) -> np.ndarray:
        """
        Get all roots of the E8×E8 heterotic structure.
        
        This combines the roots from the two E8 components, embedding them
        in the 16-dimensional space.
        
        Returns:
            np.ndarray: Array of shape (480, 16) containing all root vectors
        """
        roots1 = self.lattice1.root_system.get_roots()
        roots2 = self.lattice2.root_system.get_roots()
        
        # Embed E8 roots in 16-dimensional space
        roots = np.zeros((self.root_count, self.dimension))
        
        # First 240 roots: embed first E8 roots
        for i, root in enumerate(roots1):
            roots[i, :8] = root
        
        # Next 240 roots: embed second E8 roots
        for i, root in enumerate(roots2):
            roots[i + 240, 8:] = root
        
        return roots
    
    def compute_killing_form(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute the Killing form between two elements of the E8×E8 Lie algebra.
        
        The Killing form is a bilinear form on the Lie algebra that is invariant
        under the adjoint action of the Lie group.
        
        Args:
            X (np.ndarray): First Lie algebra element
            Y (np.ndarray): Second Lie algebra element
            
        Returns:
            float: Killing form value
        """
        # For E8×E8, the Killing form is block diagonal
        X1, X2 = X[:8], X[8:]
        Y1, Y2 = Y[:8], Y[8:]
        
        # Compute the Killing form for each E8 component
        killing1 = np.dot(X1, Y1)
        killing2 = np.dot(X2, Y2)
        
        # The Killing form for E8×E8 is the sum
        return killing1 + killing2


def compute_killing_form(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute the Killing form between two elements of the E8×E8 Lie algebra.
    
    The Killing form is a bilinear form on the Lie algebra that is invariant
    under the adjoint action of the Lie group.
    
    Args:
        X (np.ndarray): First Lie algebra element
        Y (np.ndarray): Second Lie algebra element
        
    Returns:
        float: Killing form value
    """
    heterotic = E8E8Heterotic()
    return heterotic.compute_killing_form(X, Y)


if __name__ == "__main__":
    # Example usage
    heterotic = E8E8Heterotic(max_points=100)
    
    # Get some heterotic points
    points = heterotic.get_heterotic_points()
    print(f"Generated {len(points)} heterotic points")
    print(f"First heterotic point: {points[0]}")
    
    # Test if a point is in the heterotic structure
    test_point = np.concatenate((
        np.array([1, 1, 0, 0, 0, 0, 0, 0]),
        np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    ))
    is_in_heterotic = heterotic.contains_point(test_point)
    print(f"\nIs the test point in the heterotic structure? {is_in_heterotic}")
    
    # Get dimensions
    print(f"\nDimension of E8×E8 space: {heterotic.get_dimension()}")
    print(f"Dimension of E8×E8 Lie algebra: {heterotic.get_lie_algebra_dimension()}")
    print(f"Number of roots: {heterotic.get_root_count()}")
    
    # Get the nearest heterotic point to a random point
    random_point = np.random.randn(16)
    nearest = heterotic.nearest_heterotic_point(random_point)
    print(f"\nRandom point: {random_point}")
    print(f"Nearest heterotic point: {nearest}") 