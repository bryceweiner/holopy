"""
E8 Lattice Module for HoloPy.

This module implements the E8 lattice, which is the exceptional lattice in 8 dimensions
with remarkable properties relevant to the heterotic framework.
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Union, Generator
import networkx as nx
from scipy.spatial import cKDTree

from holopy.constants.e8_constants import E8Constants
from holopy.e8.root_system import RootSystem

class E8Lattice:
    """
    A class representing the E8 lattice structure.
    
    The E8 lattice is a set of points in 8-dimensional space, defined as either:
    1. The root lattice of E8 (integer linear combinations of the simple roots)
    2. The set of points with integer or half-integer coordinates with the sum being even
    
    The E8 lattice has the densest packing of spheres in 8 dimensions and has 
    profound connections to many areas of mathematics and physics.
    
    Attributes:
        dimension (int): Dimension of the space (8)
        root_system (RootSystem): Associated root system
        kdtree (scipy.spatial.cKDTree): KD-tree for efficient nearest point lookup
        kissing_number (int): Number of nearest neighbors of any lattice point (240)
        packing_density (float): Density of the sphere packing (π^4/384)
    """
    
    def __init__(self, max_points: int = 1000):
        """
        Initialize the E8 lattice.
        
        Args:
            max_points (int): Maximum number of lattice points to generate
        """
        self.e8_constants = E8Constants()
        self.dimension = 8  # E8 lattice is always 8-dimensional
        self.root_system = RootSystem()
        self.kissing_number = 240  # Number of nearest neighbors in E8
        self.packing_density = np.pi**4 / 384  # Optimal sphere packing density in 8D
        
        # Generate lattice points up to a certain distance from the origin
        self.generate_lattice_points(max_points)
        
        # Build KD-tree for efficient nearest neighbor search
        self.kdtree = cKDTree(self.lattice_points)
    
    def generate_lattice_points(self, max_points: int):
        """
        Generate a set of E8 lattice points.
        
        This generates points in the E8 lattice up to a maximum number, 
        prioritizing points closest to the origin.
        
        Args:
            max_points (int): Maximum number of lattice points to generate
        """
        # Start with the origin
        self.lattice_points = [np.zeros(self.dimension)]
        
        # Get all roots from the root system
        roots = self.root_system.get_roots()
        
        # Define a function to generate shells of lattice points
        def generate_shell(current_points, roots, shell_number):
            """Generate the next shell of lattice points."""
            new_points = set()
            for point in current_points:
                for root in roots:
                    new_point = tuple(point + shell_number * root)
                    new_points.add(new_point)
            return [np.array(p) for p in new_points]
        
        # Generate shells until we reach the maximum number of points
        shell_number = 1
        while len(self.lattice_points) < max_points:
            current_points = [np.array(self.lattice_points[0])]  # Start from origin
            new_shell = generate_shell(current_points, roots, shell_number)
            
            # Sort by distance from origin and add until max_points is reached
            new_shell.sort(key=lambda p: np.linalg.norm(p))
            
            remaining = max_points - len(self.lattice_points)
            self.lattice_points.extend(new_shell[:remaining])
            
            if len(new_shell) <= remaining:
                shell_number += 1
            else:
                break
        
        # Convert to numpy array
        self.lattice_points = np.array(self.lattice_points)
    
    def contains_point(self, point: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Check if a point belongs to the E8 lattice.
        
        A point is in the E8 lattice if:
        1. All coordinates are integers and their sum is even, or
        2. All coordinates are half-integers and their sum is even
        
        Args:
            point (np.ndarray): The point to check
            tolerance (float): Tolerance for floating-point comparisons
            
        Returns:
            bool: True if the point is in the E8 lattice, False otherwise
        """
        # Check if all coordinates are integers or half-integers
        is_integer = all(abs(p - round(p)) < tolerance for p in point)
        is_half_integer = all(abs(p - (round(p * 2) / 2)) < tolerance for p in point)
        
        if not (is_integer or is_half_integer):
            return False
        
        # Check if the sum is even
        if is_integer:
            return round(sum(point)) % 2 == 0
        else:  # half-integer
            return round(sum(point) * 2) % 2 == 0
    
    def nearest_lattice_point(self, point: np.ndarray) -> np.ndarray:
        """
        Find the nearest E8 lattice point to a given point.
        
        Args:
            point (np.ndarray): The query point
            
        Returns:
            np.ndarray: The nearest E8 lattice point
        """
        distance, index = self.kdtree.query(point, k=1)
        return self.lattice_points[index]
    
    def get_lattice_points(self) -> np.ndarray:
        """
        Get the generated lattice points.
        
        Returns:
            np.ndarray: Array of generated lattice points
        """
        return self.lattice_points
    
    def get_kissing_number(self) -> int:
        """
        Get the kissing number of the E8 lattice.
        
        The kissing number is the number of nearest neighbors of any lattice point.
        
        Returns:
            int: Kissing number (240)
        """
        return self.kissing_number
    
    def get_packing_density(self) -> float:
        """
        Get the packing density of the E8 lattice.
        
        The packing density is the fraction of space filled by non-overlapping
        spheres centered at the lattice points.
        
        Returns:
            float: Packing density (π^4/384)
        """
        return self.packing_density
    
    def get_neighbors(self, lattice_point: np.ndarray, radius: float = 2.0) -> List[np.ndarray]:
        """
        Get all lattice points within a given radius of a lattice point.
        
        Args:
            lattice_point (np.ndarray): The center lattice point
            radius (float): The search radius
            
        Returns:
            List[np.ndarray]: List of neighboring lattice points
        """
        indices = self.kdtree.query_ball_point(lattice_point, radius)
        return [self.lattice_points[i] for i in indices]
    
    def get_voronoi_cell_vertices(self, lattice_point: np.ndarray = None) -> List[np.ndarray]:
        """
        Get the vertices of the Voronoi cell around a lattice point.
        
        The Voronoi cell is the region of space closest to the given lattice point.
        Note: This is an approximate implementation using a finite number of lattice points.
        
        Args:
            lattice_point (np.ndarray): The lattice point (defaults to origin)
            
        Returns:
            List[np.ndarray]: List of vertex coordinates
        """
        if lattice_point is None:
            lattice_point = np.zeros(self.dimension)
        
        # In the E8 lattice, the Voronoi cell is a specific polytope (the Gosset polytope)
        # A full implementation would construct it from the 240 minimal vectors
        # Here we return a simplified representation based on the minimal vectors
        
        # The vertices of the Voronoi cell at the origin are at distance √2 from the origin
        # and their coordinates are permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
        vertices = []
        for i in range(self.dimension):
            for j in range(i+1, self.dimension):
                for si in [-1, 1]:
                    for sj in [-1, 1]:
                        vertex = np.zeros(self.dimension)
                        vertex[i] = si
                        vertex[j] = sj
                        # Adjust to be centered at the specified lattice point
                        vertices.append(vertex / 2 + lattice_point)
        
        return vertices
    
    def fundamental_weights(self) -> np.ndarray:
        """
        Get the fundamental weights of the E8 lattice.
        
        The fundamental weights are the dual basis to the simple roots
        with respect to the coroot inner product.
        
        Returns:
            np.ndarray: Array of shape (8, 8) containing the fundamental weights
        """
        simple_roots = self.root_system.get_simple_roots()
        cartan_matrix = self.root_system.get_cartan_matrix()
        
        # The fundamental weights are given by (simple_roots)^T * (cartan_matrix)^-1
        cartan_inv = np.linalg.inv(cartan_matrix)
        return np.dot(simple_roots.T, cartan_inv)
    
    def distance_to_nearest_point(self, point: np.ndarray) -> float:
        """
        Compute the distance from a point to the nearest lattice point.
        
        Args:
            point (np.ndarray): The query point
            
        Returns:
            float: Distance to the nearest lattice point
        """
        distance, _ = self.kdtree.query(point, k=1)
        return distance


if __name__ == "__main__":
    # Example usage
    lattice = E8Lattice(max_points=1000)
    
    # Get some lattice points
    points = lattice.get_lattice_points()
    print(f"Generated {len(points)} lattice points")
    print(f"First 5 lattice points:")
    for i in range(min(5, len(points))):
        print(f"  {points[i]}")
    
    # Test if a point is in the lattice
    test_point1 = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    test_point2 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    test_point3 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    
    print(f"\nIs {test_point1} in the E8 lattice? {lattice.contains_point(test_point1)}")
    print(f"Is {test_point2} in the E8 lattice? {lattice.contains_point(test_point2)}")
    print(f"Is {test_point3} in the E8 lattice? {lattice.contains_point(test_point3)}")
    
    # Get packing density
    print(f"\nPacking density: {lattice.get_packing_density()}")
    print(f"Kissing number: {lattice.get_kissing_number()}") 