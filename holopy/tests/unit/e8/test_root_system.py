"""
Unit tests for the root_system module.
"""

import unittest
import numpy as np
from holopy.e8.root_system import RootSystem, get_root_vectors

class TestRootSystem(unittest.TestCase):
    """Tests for the RootSystem class and related functions."""
    
    def setUp(self):
        """Set up the test case."""
        self.root_system = RootSystem()
        self.roots = self.root_system.get_roots()
        self.simple_roots = self.root_system.get_simple_roots()
    
    def test_root_count(self):
        """Test that the E8 root system contains exactly 240 roots."""
        self.assertEqual(len(self.roots), 240, "E8 root system should have exactly 240 roots")
    
    def test_root_norm(self):
        """Test that all roots have the correct norm (squared length)."""
        # All roots in the standard normalization should have squared length 2
        for root in self.roots:
            norm_squared = np.dot(root, root)
            self.assertAlmostEqual(norm_squared, 2.0, delta=1e-10, 
                                  msg="All roots should have squared length 2")
    
    def test_simple_roots_count(self):
        """Test that there are 8 simple roots."""
        self.assertEqual(len(self.simple_roots), 8, "There should be 8 simple roots")
    
    def test_root_integrality(self):
        """Test that roots have the correct form (integer or half-integer coordinates)."""
        for root in self.roots:
            # Check that each root has either all integer coordinates or all half-integer coordinates
            is_integer = all(abs(x - round(x)) < 1e-10 for x in root)
            is_half_integer = all(abs(x - round(x * 2) / 2) < 1e-10 for x in root)
            
            self.assertTrue(is_integer or is_half_integer, 
                           f"Root {root} should have integer or half-integer coordinates")
    
    def test_root_patterns(self):
        """Test that roots follow the expected patterns."""
        # Count roots of type (±1, ±1, 0, 0, 0, 0, 0, 0) and permutations
        type1_count = 0
        
        # Count roots of type (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2)
        type2_count = 0
        
        for root in self.roots:
            # Check if it's a type 1 root
            if all(abs(x) < 1e-10 or abs(abs(x) - 1) < 1e-10 for x in root) and sum(abs(x) > 1e-10 for x in root) == 2:
                type1_count += 1
            
            # Check if it's a type 2 root
            elif all(abs(abs(x) - 0.5) < 1e-10 for x in root):
                type2_count += 1
        
        self.assertEqual(type1_count, 112, "There should be 112 roots of type (±1, ±1, 0, ..., 0)")
        self.assertEqual(type2_count, 128, "There should be 128 roots of type (±1/2, ±1/2, ..., ±1/2)")
    
    def test_positive_roots(self):
        """Test that the number of positive roots is half the total number of roots."""
        positive_roots = self.root_system.get_positive_roots()
        self.assertEqual(len(positive_roots), 120, "There should be 120 positive roots")
    
    def test_highest_root(self):
        """Test the highest root."""
        highest_root = self.root_system.get_highest_root()
        
        # The highest root should have a specific form known for E8
        # In our case, we're returning it in the standard basis as [1, 1, 0, 0, 0, 0, 0, 0]
        expected_highest_root = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        
        self.assertTrue(np.allclose(highest_root, expected_highest_root), 
                       f"Highest root {highest_root} does not match expected {expected_highest_root}")
    
    def test_inner_product(self):
        """Test the inner product computation."""
        # Take two roots
        root1 = self.roots[0]
        root2 = self.roots[1]
        
        # Compute inner product directly
        expected_inner_product = np.dot(root1, root2)
        
        # Compute using the method
        actual_inner_product = self.root_system.compute_inner_product(root1, root2)
        
        self.assertAlmostEqual(actual_inner_product, expected_inner_product, delta=1e-10,
                              msg="Inner product computation is incorrect")
    
    def test_angle_between_roots(self):
        """Test the computation of angles between roots."""
        # Find two roots with a known angle (e.g., perpendicular roots)
        for i, root1 in enumerate(self.roots):
            for j, root2 in enumerate(self.roots[i+1:], i+1):
                if abs(np.dot(root1, root2)) < 1e-10:  # Perpendicular roots
                    angle = self.root_system.compute_angle(root1, root2)
                    self.assertAlmostEqual(angle, np.pi/2, delta=1e-10,
                                          msg="Angle computation is incorrect")
                    return
    
    def test_cartan_matrix(self):
        """Test that the Cartan matrix has the expected properties."""
        cartan_matrix = self.root_system.get_cartan_matrix()
        
        # Cartan matrix should be 8x8
        self.assertEqual(cartan_matrix.shape, (8, 8), "Cartan matrix should be 8x8")
        
        # Cartan matrix should be symmetric in this case (since all roots have same length)
        self.assertTrue(np.allclose(cartan_matrix, cartan_matrix.T), "Cartan matrix should be symmetric")
        
        # Diagonal elements should be 2
        for i in range(8):
            self.assertAlmostEqual(cartan_matrix[i, i], 2.0, delta=1e-10,
                                  msg="Diagonal elements of Cartan matrix should be 2")
    
    def test_nearest_root(self):
        """Test finding the nearest root to a given vector."""
        # Create a vector close to a known root
        known_root = self.roots[0]
        test_vector = known_root + 0.1 * np.random.randn(8)
        
        nearest_root = self.root_system.find_nearest_root(test_vector)
        
        self.assertTrue(np.allclose(nearest_root, known_root), 
                       "Nearest root finder failed to find the expected root")
    
    def test_get_root_vectors_function(self):
        """Test the get_root_vectors function."""
        roots = get_root_vectors()
        self.assertEqual(len(roots), 240, "get_root_vectors should return 240 roots")


if __name__ == '__main__':
    unittest.main() 