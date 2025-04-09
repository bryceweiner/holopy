"""
Unit tests for the tensor module.
"""

import unittest
import numpy as np
from holopy.info.tensor import (
    compute_higher_order_functional,
    higher_rank_tensor,
    compute_scalar_curvature,
    compute_k_tensor,
    information_to_energy_tensor
)
from holopy.info.current import InfoCurrentTensor

class TestTensorModule(unittest.TestCase):
    """Tests for the tensor module functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create a simple test tensor
        dimension = 4
        self.tensor = np.eye(dimension)  # Simple identity matrix
        self.density = np.ones(dimension)
        
        # Create a test metric (flat Minkowski space for simplicity)
        self.metric = np.diag([-1, 1, 1, 1])
        
        # Create an information current tensor
        self.info_tensor = InfoCurrentTensor(self.tensor, self.density)
        
        # Get the numpy array representation
        self.J = self.info_tensor.get_tensor()
    
    def test_compute_higher_order_functional(self):
        """Test the compute_higher_order_functional function."""
        # Compute the higher-order functional
        H = compute_higher_order_functional(self.density, self.J)
        
        # For an identity tensor and constant density, this should have specific properties
        self.assertEqual(H.shape, (4,))  # Should have same dimension as density
        
        # Check that it's not all zeros (the function should do something)
        self.assertFalse(np.all(H == 0))
    
    def test_higher_rank_tensor(self):
        """Test the higher_rank_tensor function."""
        # Compute a higher-rank tensor without metric
        higher_tensor = higher_rank_tensor(self.J)
        
        # Check the shape (should be 4x4x4)
        self.assertEqual(higher_tensor.shape, (4, 4, 4))
        
        # Create a different metric (strongly curved spacetime)
        curved_metric = np.diag([-2.0, 0.5, 0.5, 0.5])
        
        # Compute with curved metric
        higher_tensor_curved = higher_rank_tensor(self.J, curved_metric)
        
        # Check the shape
        self.assertEqual(higher_tensor_curved.shape, (4, 4, 4))
        
        # Both tensors should be valid - check they're not all zeros
        self.assertFalse(np.all(higher_tensor == 0))
        self.assertFalse(np.all(higher_tensor_curved == 0))
    
    def test_compute_scalar_curvature(self):
        """Test the compute_scalar_curvature function."""
        # Compute the scalar curvature
        R = compute_scalar_curvature(self.J, self.metric)
        
        # Check that it's a scalar (float)
        self.assertIsInstance(R, float)
    
    def test_compute_k_tensor(self):
        """Test the compute_k_tensor function."""
        # Compute the K tensor
        K = compute_k_tensor(self.J)
        
        # Check the shape (should be 4x4)
        self.assertEqual(K.shape, (4, 4))
        
        # Compute with metric
        K_with_metric = compute_k_tensor(self.J, self.metric)
        
        # Check the shape
        self.assertEqual(K_with_metric.shape, (4, 4))
    
    def test_information_to_energy_tensor(self):
        """Test the information_to_energy_tensor function."""
        # Convert to energy-momentum tensor
        T = information_to_energy_tensor(self.J)
        
        # Check the shape (should be 4x4)
        self.assertEqual(T.shape, (4, 4))
        
        # Check that the energy-momentum tensor is symmetric
        # T_μν should equal T_νμ
        np.testing.assert_array_almost_equal(T, T.T)
        
        # Compute with metric
        T_with_metric = information_to_energy_tensor(self.J, self.metric)
        
        # Check the shape
        self.assertEqual(T_with_metric.shape, (4, 4))
        
        # Check symmetry with metric version
        np.testing.assert_array_almost_equal(T_with_metric, T_with_metric.T)


if __name__ == '__main__':
    unittest.main() 