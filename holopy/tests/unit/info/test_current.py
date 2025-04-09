"""
Unit tests for the current module.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from holopy.constants.physical_constants import PHYSICAL_CONSTANTS
from holopy.info.current import InfoCurrentTensor, compute_divergence

class TestInfoCurrentTensor(unittest.TestCase):
    """Tests for the InfoCurrentTensor class and related functions."""
    
    def setUp(self):
        """Set up the test case."""
        # Define a simple test tensor and density
        dimension = 4
        self.tensor = np.eye(dimension)  # Simple identity matrix for testing
        self.density = np.ones(dimension)
        
        # Create a current tensor
        self.current = InfoCurrentTensor(self.tensor, self.density)
        
        # Set up a Gaussian density function for testing
        def gaussian_density(x):
            return np.exp(-np.sum(x**2) / 2)
        
        self.density_function = gaussian_density
    
    def test_initialization(self):
        """Test the initialization of an InfoCurrentTensor."""
        # Check that tensor and density are stored correctly
        np.testing.assert_array_equal(self.current.get_tensor(), self.tensor)
        np.testing.assert_array_equal(self.current.get_density(), self.density)
        
        # Check that dimension is correct
        self.assertEqual(self.current.dimension, 4)
        
        # Check that coordinates default to 'cartesian'
        self.assertEqual(self.current.coordinates, 'cartesian')
    
    def test_from_density_function(self):
        """Test creating a current tensor from a density function."""
        # Create a current tensor from the density function
        grid_size = 10
        current = InfoCurrentTensor.from_density(self.density_function, grid_size=grid_size)
        
        # Check that the result has the right shape
        self.assertEqual(current.get_tensor().shape, (4, 4))
        self.assertEqual(current.get_density().shape, (4,))
    
    def test_get_component(self):
        """Test retrieving a specific component of the tensor."""
        # Check that components match the original tensor
        for mu in range(4):
            for nu in range(4):
                self.assertEqual(self.current.get_component(mu, nu), self.tensor[mu, nu])
    
    def test_trace(self):
        """Test computing the trace of the tensor."""
        # The trace of an identity matrix is its dimension
        self.assertEqual(self.current.trace(), 4)
    
    def test_compute_divergence(self):
        """Test computing the divergence of the tensor."""
        # Get the divergence
        divergence = self.current.compute_divergence()
        
        # For this simple test case, the divergence should be gamma * density
        expected_divergence = PHYSICAL_CONSTANTS.get_gamma() * self.density
        
        # Check that the divergence is computed correctly
        np.testing.assert_array_almost_equal(divergence, expected_divergence)
    
    def test_compute_divergence_function(self):
        """Test the standalone compute_divergence function."""
        # Compute the divergence using the function
        divergence = compute_divergence(self.current)
        
        # This should be the same as calling the method directly
        expected_divergence = self.current.compute_divergence()
        
        # Check that they match
        np.testing.assert_array_equal(divergence, expected_divergence)
    
    def test_conservation_law(self):
        """Test if the tensor satisfies the conservation law."""
        # The conservation law states that ∇_μ J^μν = γ · ρ^ν
        # Get the divergence
        divergence = self.current.compute_divergence()
        
        # Get the expected divergence from the conservation law
        gamma = PHYSICAL_CONSTANTS.get_gamma()
        expected_divergence = gamma * self.density
        
        # Check that they match within a small tolerance
        np.testing.assert_array_almost_equal(divergence, expected_divergence)
    
    def test_invalid_dimensions(self):
        """Test that initialization with invalid dimensions raises an error."""
        # Create tensor and density with mismatched dimensions
        invalid_tensor = np.eye(3)
        invalid_density = np.ones(4)
        
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            InfoCurrentTensor(invalid_tensor, invalid_density)
    
    def test_from_density_with_domain(self):
        """Test creating a current tensor with a custom domain."""
        # Create a domain centered at the origin
        domain = [(-2, 2), (-2, 2), (-2, 2), (-2, 2)]
        
        # Create a current tensor from the density function with this domain
        current = InfoCurrentTensor.from_density(self.density_function, domain=domain)
        
        # Check that the result has the right shape
        self.assertEqual(current.get_tensor().shape, (4, 4))
        self.assertEqual(current.get_density().shape, (4,))


if __name__ == '__main__':
    unittest.main() 