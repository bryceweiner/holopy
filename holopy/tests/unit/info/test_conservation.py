"""
Unit tests for the conservation module.
"""

import unittest
import numpy as np
from holopy.info.conservation import (
    information_conservation,
    maximum_information_processing_rate,
    information_processing_constraint,
    black_hole_information_processing,
    check_information_bound
)
from holopy.info.current import InfoCurrentTensor
from holopy.constants.physical_constants import get_planck_area, get_planck_mass

class TestConservationModule(unittest.TestCase):
    """Tests for the conservation module functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create a simple test tensor (identity matrix in 4D)
        dimension = 4
        self.tensor = np.eye(dimension)
        self.density = np.ones(dimension)
        
        # Create an information current tensor that should satisfy conservation
        self.current = InfoCurrentTensor(self.tensor, self.density)
        
        # Create test values for various functions
        self.test_area = get_planck_area() * 1000  # 1000 Planck areas
        self.test_entropy = 1.0  # 1 bit of entropy
        self.test_mass = get_planck_mass() * 10  # 10 Planck masses
        self.test_system_size = 1e-35  # m
        self.test_information_content = 1e20  # bits
    
    def test_information_conservation(self):
        """Test the information_conservation function."""
        # Check if the tensor satisfies conservation law to first order
        conserved, deviation = information_conservation(self.current, order=1)
        
        # The simple test tensor should approximately satisfy conservation
        self.assertTrue(conserved)
        self.assertLess(deviation, 1e-10)
        
        # Test with higher order
        conserved2, deviation2 = information_conservation(self.current, order=2)
        
        # The function should return meaningful values
        self.assertIsInstance(conserved2, bool)
        self.assertIsInstance(deviation2, float)
    
    def test_maximum_information_processing_rate(self):
        """Test the maximum_information_processing_rate function."""
        # Calculate the maximum rate
        rate = maximum_information_processing_rate(self.test_area)
        
        # Check that the rate is proportional to the area
        rate2 = maximum_information_processing_rate(self.test_area * 2)
        self.assertAlmostEqual(rate2 / rate, 2.0)
        
        # Check that the rate is positive
        self.assertGreater(rate, 0)
    
    def test_information_processing_constraint(self):
        """Test the information_processing_constraint function."""
        # Calculate the constraint for our test entropy
        constraint = information_processing_constraint(self.test_entropy)
        
        # Check that the constraint is positive
        self.assertGreater(constraint, 0)
        
        # Test with zero entropy
        zero_constraint = information_processing_constraint(0)
        self.assertEqual(zero_constraint, 0)
    
    def test_black_hole_information_processing(self):
        """Test the black_hole_information_processing function."""
        # Calculate the processing capacity
        capacity = black_hole_information_processing(self.test_mass)
        
        # Check that the capacity is positive
        self.assertGreater(capacity, 0)
        
        # Check that the capacity scales with mass
        capacity2 = black_hole_information_processing(self.test_mass * 2)
        self.assertGreater(capacity2, capacity)
    
    def test_check_information_bound(self):
        """Test the check_information_bound function."""
        # Test a case that should be within the bound
        small_system = check_information_bound(self.test_system_size, 1)
        
        # Test a case that should exceed the bound
        large_system = check_information_bound(self.test_system_size, self.test_information_content)
        
        # The small system should be within bounds, the large one should not
        self.assertTrue(small_system)
        self.assertFalse(large_system)


if __name__ == '__main__':
    unittest.main() 