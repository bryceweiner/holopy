"""
E8 Root System Module for HoloPy.

This module implements the 240 roots of the E8 root system, providing tools
for working with and analyzing the root structure that underlies the
heterotic framework.
"""

import numpy as np
import scipy.spatial.distance as dist
import networkx as nx
from itertools import combinations, product
from typing import List, Tuple, Dict, Set, Optional, Union, Generator

from holopy.constants.e8_constants import E8Constants

class RootSystem:
    """
    A class representing the root system of the E8 Lie algebra.
    
    The E8 root system consists of 240 roots in an 8-dimensional space:
    - 112 roots of the form (±1, ±1, 0, 0, 0, 0, 0, 0) and all permutations
    - 128 roots of the form (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2)
      with an even number of positive signs
    
    Attributes:
        roots (np.ndarray): Array of shape (240, 8) containing all root vectors
        rank (int): Rank of the E8 Lie algebra (8)
        dimension (int): Dimension of the E8 Lie algebra (248)
        root_count (int): Number of roots (240)
        simple_roots (np.ndarray): Array of shape (8, 8) containing the simple roots
        graph (nx.Graph): Graph representation of the root system
    """
    
    def __init__(self):
        """Initialize the E8 root system."""
        self.e8_constants = E8Constants()
        self.rank = self.e8_constants.rank
        self.dimension = self.e8_constants.dimension
        self.root_count = self.e8_constants.root_count
        
        # Generate the 240 roots of E8
        self.generate_roots()
        
        # Generate simple roots
        self.generate_simple_roots()
        
        # Build the root graph
        self.build_root_graph()
    
    def generate_roots(self):
        """
        Generate all 240 roots of the E8 root system.
        
        This method fills the self.roots array with all 240 root vectors,
        which come in two types:
        - Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0) and permutations (112 roots)
        - Type 2: (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2) with 
          even number of positive signs (128 roots)
        """
        # Initialize array to store all 240 roots
        self.roots = np.zeros((self.root_count, self.rank))
        
        # Generate Type 1 roots: (±1, ±1, 0, 0, 0, 0, 0, 0) and permutations
        # This gives 8 choose 2 = 28 positions × 4 sign combinations = 112 roots
        index = 0
        for i, j in combinations(range(self.rank), 2):
            for si, sj in product([-1, 1], repeat=2):
                root = np.zeros(self.rank)
                root[i] = si
                root[j] = sj
                self.roots[index] = root
                index += 1
        
        # Generate Type 2 roots: (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2)
        # with even number of positive signs
        # This gives 2^7 = 128 roots
        for signs in self._generate_even_sign_patterns():
            root = np.array(signs) / 2
            self.roots[index] = root
            index += 1
    
    def _generate_even_sign_patterns(self) -> Generator[List[int], None, None]:
        """
        Generate all sign patterns with an even number of positive signs.
        
        This method generates all 128 sign patterns (±1, ±1, ..., ±1) with an 
        even number of positive signs.
        
        Yields:
            List[int]: Sign pattern [±1, ±1, ..., ±1] with even number of +1s
        """
        # Helper function to count number of positive signs in a pattern
        def count_positives(pattern):
            return sum(1 for sign in pattern if sign > 0)
        
        # Generate all possible sign patterns
        for signs in product([-1, 1], repeat=self.rank):
            # Keep only patterns with an even number of positive signs
            if count_positives(signs) % 2 == 0:
                yield list(signs)
    
    def generate_simple_roots(self):
        """
        Generate the simple roots of the E8 root system.
        
        The simple roots form a basis of the root space and can generate
        all other roots through linear combinations with integer coefficients.
        """
        # Standard simple roots for E8 in the α-basis
        self.simple_roots = np.array([
            [1, -1, 0, 0, 0, 0, 0, 0],  # α₁
            [0, 1, -1, 0, 0, 0, 0, 0],  # α₂
            [0, 0, 1, -1, 0, 0, 0, 0],  # α₃
            [0, 0, 0, 1, -1, 0, 0, 0],  # α₄
            [0, 0, 0, 0, 1, -1, 0, 0],  # α₅
            [0, 0, 0, 0, 0, 1, -1, 0],  # α₆
            [0, 0, 0, 0, 0, 0, 1, 1],   # α₇
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5]  # α₈
        ])
    
    def build_root_graph(self):
        """
        Build a graph representation of the root system.
        
        This creates a graph where each node is a root and edges connect
        roots that differ by a simple root.
        """
        # Create a graph
        self.graph = nx.Graph()
        
        # Add roots as nodes
        for i, root in enumerate(self.roots):
            self.graph.add_node(i, vector=root)
        
        # Add edges between roots that differ by a simple root
        for i in range(self.root_count):
            for j in range(i + 1, self.root_count):
                diff = self.roots[i] - self.roots[j]
                # Check if the difference is a simple root
                if any(np.allclose(diff, sr) or np.allclose(diff, -sr) for sr in self.simple_roots):
                    self.graph.add_edge(i, j)
    
    def get_roots(self) -> np.ndarray:
        """
        Get all roots of the E8 root system.
        
        Returns:
            np.ndarray: Array of shape (240, 8) containing all 240 root vectors
        """
        return self.roots
    
    def get_simple_roots(self) -> np.ndarray:
        """
        Get the simple roots of the E8 root system.
        
        Returns:
            np.ndarray: Array of shape (8, 8) containing the 8 simple roots
        """
        return self.simple_roots
    
    def get_root_graph(self) -> nx.Graph:
        """
        Get the graph representation of the root system.
        
        Returns:
            nx.Graph: Graph where nodes are roots and edges connect roots
                     that differ by a simple root
        """
        return self.graph
    
    def get_positive_roots(self) -> np.ndarray:
        """
        Get the positive roots of the E8 root system.
        
        Positive roots are those that can be expressed as positive linear
        combinations of the simple roots.
        
        Returns:
            np.ndarray: Array containing the positive roots
        """
        # A root is positive if its first non-zero component is positive
        positive_roots = []
        for root in self.roots:
            for component in root:
                if component != 0:
                    if component > 0:
                        positive_roots.append(root)
                    break
        return np.array(positive_roots)
    
    def get_highest_root(self) -> np.ndarray:
        """
        Get the highest root of the E8 root system.
        
        Returns:
            np.ndarray: The highest root vector
        """
        # The highest root is the one with the largest height
        # (sum of coefficients when expressed in terms of simple roots)
        # For E8, it is known to be [2, 3, 4, 6, 5, 4, 3, 2] in the α-basis
        # Here we return it in the standard basis
        return np.array([1, 1, 0, 0, 0, 0, 0, 0])
    
    def compute_inner_product(self, root1: np.ndarray, root2: np.ndarray) -> float:
        """
        Compute the inner product between two roots.
        
        Args:
            root1 (np.ndarray): First root vector
            root2 (np.ndarray): Second root vector
            
        Returns:
            float: Inner product between the roots
        """
        return np.dot(root1, root2)
    
    def compute_angle(self, root1: np.ndarray, root2: np.ndarray) -> float:
        """
        Compute the angle between two roots in radians.
        
        Args:
            root1 (np.ndarray): First root vector
            root2 (np.ndarray): Second root vector
            
        Returns:
            float: Angle between the roots in radians
        """
        dot_product = self.compute_inner_product(root1, root2)
        norm1 = np.linalg.norm(root1)
        norm2 = np.linalg.norm(root2)
        
        # Handle numerical precision issues
        cos_angle = dot_product / (norm1 * norm2)
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        
        return np.arccos(cos_angle)
    
    def find_nearest_root(self, vector: np.ndarray) -> np.ndarray:
        """
        Find the root nearest to a given vector.
        
        Args:
            vector (np.ndarray): Input vector
            
        Returns:
            np.ndarray: The nearest root vector
        """
        distances = np.array([np.linalg.norm(vector - root) for root in self.roots])
        nearest_index = np.argmin(distances)
        return self.roots[nearest_index]
    
    def get_cartan_matrix(self) -> np.ndarray:
        """
        Get the Cartan matrix of the E8 root system.
        
        The Cartan matrix A_ij = 2(α_i,α_j)/(α_j,α_j) where α_i are simple roots.
        
        Returns:
            np.ndarray: The 8×8 Cartan matrix
        """
        n = len(self.simple_roots)
        cartan_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                alpha_i = self.simple_roots[i]
                alpha_j = self.simple_roots[j]
                cartan_matrix[i, j] = 2 * np.dot(alpha_i, alpha_j) / np.dot(alpha_j, alpha_j)
        
        return cartan_matrix


def get_root_vectors() -> np.ndarray:
    """
    Get the 240 root vectors of the E8 root system.
    
    Returns:
        np.ndarray: Array of shape (240, 8) containing all root vectors
    """
    root_system = RootSystem()
    return root_system.get_roots()


if __name__ == "__main__":
    # Example usage
    rs = RootSystem()
    roots = rs.get_roots()
    
    print(f"Number of roots: {len(roots)}")
    print(f"First 5 roots:")
    for i in range(5):
        print(f"  {roots[i]}")
    
    # Verify some properties
    simple_roots = rs.get_simple_roots()
    cartan_matrix = rs.get_cartan_matrix()
    print(f"\nCartan matrix shape: {cartan_matrix.shape}")
    print(f"Cartan matrix:\n{cartan_matrix}") 