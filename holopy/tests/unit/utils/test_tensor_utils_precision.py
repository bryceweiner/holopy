"""
Tests for numerical precision in tensor utilities.

This module contains high-precision tests of tensor calculations,
ensuring that our numerical methods provide accurate results
for common test cases with known analytical solutions.
"""

import unittest
import numpy as np
from holopy.utils.tensor_utils import (
    compute_christoffel_symbols,
    compute_riemann_tensor
)


class TestTensorUtilsPrecision(unittest.TestCase):
    """Tests for ensuring high precision in tensor calculations."""
    
    def setUp(self):
        """Set up test data for precision testing."""
        # Create a Minkowski metric
        self.minkowski_metric = np.diag([-1.0, 1.0, 1.0, 1.0])
        
        # Create a Schwarzschild metric
        r = 10.0  # Large radius to avoid strong curvature effects
        M = 1.0   # Mass in geometric units
        self.schwarzschild_metric = np.array([
            [-(1-2*M/r), 0, 0, 0],
            [0, 1/(1-2*M/r), 0, 0],
            [0, 0, r**2, 0],
            [0, 0, 0, r**2 * np.sin(np.pi/4)**2]
        ])
    
    def test_minkowski_exact_zero(self):
        """Test that Minkowski metric produces exactly zero Christoffel symbols and Riemann tensor."""
        # Compute Christoffel symbols
        christoffel = compute_christoffel_symbols(self.minkowski_metric)
        
        # Check that they are exactly zero (not just close to zero)
        self.assertTrue(np.all(christoffel == 0.0), 
                     "Christoffel symbols should be exactly zero for Minkowski metric")
        
        # Compute Riemann tensor
        riemann = compute_riemann_tensor(christoffel)
        
        # Check that it is exactly zero
        self.assertTrue(np.all(riemann == 0.0),
                     "Riemann tensor should be exactly zero for Minkowski metric")
    
    def test_schwarzschild_key_components(self):
        """Test the accuracy of key Schwarzschild metric tensor components."""
        # Parameters used for Schwarzschild metric
        r = 10.0
        M = 1.0
        
        # Compute Christoffel symbols
        christoffel = compute_christoffel_symbols(self.schwarzschild_metric)
        
        # Check key Schwarzschild Christoffel symbols - high precision checks
        # Γ^t_tr = Γ^t_rt = M/r^2/(1-2M/r)
        expected_gamma_t_tr = M / (r**2 - 2*M*r)  # = M/(r^2(1-2M/r))
        self.assertAlmostEqual(christoffel[0, 0, 1], expected_gamma_t_tr, places=10)
        self.assertAlmostEqual(christoffel[0, 1, 0], expected_gamma_t_tr, places=10)
        
        # Γ^r_tt = M/r^2 * (1-2M/r)
        expected_gamma_r_tt = M/r**2 * (1-2*M/r)
        self.assertAlmostEqual(christoffel[1, 0, 0], expected_gamma_r_tt, places=10)
        
        # Compute Riemann tensor
        riemann = compute_riemann_tensor(christoffel)
        
        # Check key Schwarzschild Riemann tensor components - high precision checks
        # R^t_rtr = 2M/r^3
        expected_R_t_rtr = 2*M/r**3
        self.assertAlmostEqual(riemann[0, 1, 0, 1], expected_R_t_rtr, places=10)
        
        # R^r_trt = -2M/r^3
        expected_R_r_trt = -2*M/r**3
        self.assertAlmostEqual(riemann[1, 0, 1, 0], expected_R_r_trt, places=10)
        
        # Verify antisymmetry: R^t_rtr = -R^t_rrt
        self.assertAlmostEqual(riemann[0, 1, 0, 1], -riemann[0, 1, 1, 0], places=10)


if __name__ == '__main__':
    unittest.main() 