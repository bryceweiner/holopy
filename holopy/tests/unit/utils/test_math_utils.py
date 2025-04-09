"""
Tests for the math_utils module.

This module tests the mathematical utility functions for tensor operations,
numerical derivatives, and specialized calculations used throughout HoloPy.
"""

import unittest
import numpy as np
from typing import Callable

from holopy.utils.math_utils import (
    tensor_contraction,
    numerical_gradient,
    numerical_hessian,
    spatial_complexity,
    metrics_equivalent,
    gamma_matrix,
    killing_form_e8
)


class TestMathUtils(unittest.TestCase):
    """Test suite for the math_utils module."""

    def setUp(self):
        """Set up test fixtures."""
        # Simple 3D tensor for contraction tests
        self.tensor_3d = np.arange(27).reshape(3, 3, 3)
        
        # Define a simple function for gradient and Hessian tests
        # f(x, y) = x^2 + 2xy + y^2
        self.test_function = lambda p: p[0]**2 + 2*p[0]*p[1] + p[1]**2
        
        # Known gradient of test_function at [1, 1]: [4, 4]
        self.expected_gradient = np.array([4.0, 4.0])
        
        # Known Hessian of test_function: [[2, 2], [2, 2]]
        self.expected_hessian = np.array([[2.0, 2.0], [2.0, 2.0]])
        
        # Simple wavefunction for complexity test
        # ψ(x, y) = e^(-(x^2 + y^2)/2) (complex Gaussian)
        self.test_wavefunction = lambda p: np.exp(-(p[0]**2 + p[1]**2)/2)
        
        # Test metrics
        self.metric1 = np.diag([1.0, -1.0, -1.0, -1.0])  # Minkowski
        self.metric2 = np.diag([1.0, -1.0, -1.0, -1.0]) * 1.0000001  # Slightly different
        self.metric3 = np.diag([1.1, -1.0, -1.0, -1.0])  # Notably different
        
        # Random root vectors for E8 tests (simplified)
        np.random.seed(42)  # For reproducibility
        self.root_vectors = np.random.randn(240, 8)
        
    def test_tensor_contraction_validation(self):
        """Test tensor contraction input validation."""
        # Test invalid indices length
        with self.assertRaises(ValueError):
            tensor_contraction(self.tensor_3d, (0, 1, 2))
        
        # Test out of bounds indices
        with self.assertRaises(ValueError):
            tensor_contraction(self.tensor_3d, (0, 3))
        
        # Test equal indices
        with self.assertRaises(ValueError):
            tensor_contraction(self.tensor_3d, (1, 1))
            
    def test_tensor_contraction_correctness(self):
        """Test tensor contraction correctness."""
        # Contract first two dimensions (trace)
        result = tensor_contraction(self.tensor_3d, (0, 1))
        
        # Expected: sum of diagonal elements for each 3rd index
        expected = np.array([
            self.tensor_3d[0, 0, 0] + self.tensor_3d[1, 1, 0] + self.tensor_3d[2, 2, 0],
            self.tensor_3d[0, 0, 1] + self.tensor_3d[1, 1, 1] + self.tensor_3d[2, 2, 1],
            self.tensor_3d[0, 0, 2] + self.tensor_3d[1, 1, 2] + self.tensor_3d[2, 2, 2]
        ])
        
        np.testing.assert_array_equal(result, expected)
        
        # Contract first and last dimensions
        result = tensor_contraction(self.tensor_3d, (0, 2))
        
        # Expected: sum of elements where first and last indices are equal
        expected = np.array([
            self.tensor_3d[0, 0, 0] + self.tensor_3d[1, 0, 1] + self.tensor_3d[2, 0, 2],
            self.tensor_3d[0, 1, 0] + self.tensor_3d[1, 1, 1] + self.tensor_3d[2, 1, 2],
            self.tensor_3d[0, 2, 0] + self.tensor_3d[1, 2, 1] + self.tensor_3d[2, 2, 2]
        ])
        
        np.testing.assert_array_equal(result, expected)
        
    def test_numerical_gradient(self):
        """Test numerical gradient calculation."""
        # Test point
        point = np.array([1.0, 1.0])
        
        # Calculate gradient
        gradient = numerical_gradient(self.test_function, point)
        
        # Check result
        np.testing.assert_allclose(gradient, self.expected_gradient, rtol=1e-4)
        
        # Test with a vector-valued function
        vector_function = lambda p: np.array([p[0]**2, p[0]*p[1]])
        jacobian = numerical_gradient(vector_function, point)
        
        # Expected result: Jacobian matrix
        # For [x^2, xy] at [1,1], the Jacobian is [[2x, 0], [y, x]] = [[2, 0], [1, 1]]
        expected_jacobian = np.array([
            [2.0, 0.0],  # Derivatives of x^2
            [1.0, 1.0]   # Derivatives of xy
        ])
        
        np.testing.assert_allclose(jacobian, expected_jacobian, rtol=1e-4)
        
    def test_numerical_hessian(self):
        """Test numerical Hessian matrix calculation."""
        # Test point
        point = np.array([1.0, 1.0])
        
        # Calculate Hessian
        hessian = numerical_hessian(self.test_function, point)
        
        # Check result
        np.testing.assert_allclose(hessian, self.expected_hessian, rtol=1e-4)
        
        # Test with another function
        # f(x, y) = x^3 + y^2
        another_function = lambda p: p[0]**3 + p[1]**2
        another_hessian = numerical_hessian(another_function, point)
        
        # Expected Hessian at [1, 1]: [[6, 0], [0, 2]]
        expected_another_hessian = np.array([[6.0, 0.0], [0.0, 2.0]])
        
        # Only check diagonal elements (function derivatives) with strict tolerance
        np.testing.assert_allclose(np.diag(another_hessian), np.diag(expected_another_hessian), rtol=1e-2)
        
        # Check off-diagonal elements are near zero (within numerical precision)
        self.assertAlmostEqual(another_hessian[0, 1], 0.0, places=5)
        self.assertAlmostEqual(another_hessian[1, 0], 0.0, places=5)
        
    def test_spatial_complexity(self):
        """Test spatial complexity calculation."""
        # Test points
        points = np.array([
            [0.0, 0.0],  # Origin
            [1.0, 0.0],  # On x-axis
            [0.0, 1.0]   # On y-axis
        ])
        
        # Calculate complexity
        complexity = spatial_complexity(self.test_wavefunction, points)
        
        # Expected complexity for Gaussian:
        # |∇ψ|² = |xψ|² + |yψ|² = |x·e^(-(x^2+y^2)/2)|² + |y·e^(-(x^2+y^2)/2)|²
        # At origin: 0
        # At [1, 0]: |1·e^(-1/2)|² = |e^(-1/2)|² = e^(-1)
        # At [0, 1]: |1·e^(-1/2)|² = |e^(-1/2)|² = e^(-1)
        expected_complexity = np.array([
            0.0,            # At origin: 0
            np.exp(-1.0),   # At [1, 0]: e^(-1)
            np.exp(-1.0)    # At [0, 1]: e^(-1)
        ])
        
        np.testing.assert_allclose(complexity, expected_complexity, rtol=1e-4)
        
    def test_metrics_equivalent(self):
        """Test metrics equivalence checking."""
        # Test identical metrics
        self.assertTrue(metrics_equivalent(self.metric1, self.metric1))
        
        # Test nearly identical metrics
        self.assertTrue(metrics_equivalent(self.metric1, self.metric2, tolerance=1e-5))
        
        # Test different metrics
        self.assertFalse(metrics_equivalent(self.metric1, self.metric3, tolerance=1e-5))
        
        # Test zero metrics
        zero_metric = np.zeros((4, 4))
        self.assertTrue(metrics_equivalent(zero_metric, zero_metric))
        
        # Test one zero and one non-zero metric
        self.assertFalse(metrics_equivalent(zero_metric, self.metric1))
        
        # Test metrics with different shapes
        small_metric = np.eye(3)
        self.assertFalse(metrics_equivalent(small_metric, self.metric1))
        
    def test_gamma_matrix(self):
        """Test gamma matrix construction."""
        # Test standard gamma matrices with Minkowski metric
        gamma0 = gamma_matrix(0)
        gamma1 = gamma_matrix(1)
        gamma2 = gamma_matrix(2)
        gamma3 = gamma_matrix(3)
        
        # Check dimensions
        self.assertEqual(gamma0.shape, (4, 4))
        self.assertEqual(gamma1.shape, (4, 4))
        self.assertEqual(gamma2.shape, (4, 4))
        self.assertEqual(gamma3.shape, (4, 4))
        
        # Check Clifford algebra relation: {γᵐ, γⁿ} = 2gᵐⁿI
        g_mn = np.diag([1.0, -1.0, -1.0, -1.0])  # Minkowski metric
        
        def commutator(a, b):
            return a @ b + b @ a
        
        # Check {γ₀, γ₀} = 2g₀₀I = 2I
        np.testing.assert_allclose(commutator(gamma0, gamma0), 2 * g_mn[0, 0] * np.eye(4), rtol=1e-10)
        
        # Check {γ₁, γ₁} = 2g₁₁I = -2I
        np.testing.assert_allclose(commutator(gamma1, gamma1), 2 * g_mn[1, 1] * np.eye(4), rtol=1e-10)
        
        # Check {γ₀, γ₁} = 0
        np.testing.assert_allclose(commutator(gamma0, gamma1), np.zeros((4, 4)), rtol=1e-10)
        
        # Test with invalid index
        with self.assertRaises(ValueError):
            gamma_matrix(4)
        
        # Test with custom metric
        custom_metric = np.diag([2.0, -2.0, -2.0, -2.0])  # Scaled Minkowski
        custom_gamma0 = gamma_matrix(0, custom_metric)
        
        # Check shape
        self.assertEqual(custom_gamma0.shape, (4, 4))
        
        # Custom gamma should be scaled by sqrt(g₀₀)
        expected_scaling = np.sqrt(custom_metric[0, 0] / g_mn[0, 0])
        np.testing.assert_allclose(custom_gamma0, gamma0 * expected_scaling, rtol=1e-10)
        
    def test_killing_form_e8(self):
        """Test Killing form calculation for E8."""
        # Calculate Killing form
        killing_form = killing_form_e8(self.root_vectors)
        
        # Check dimensions
        self.assertEqual(killing_form.shape, (248, 248))
        
        # Check symmetry
        np.testing.assert_allclose(killing_form, killing_form.T, rtol=1e-10)
        
        # Check Cartan subalgebra part
        np.testing.assert_allclose(killing_form[:8, :8], 60 * np.eye(8), rtol=1e-10)
        
        # Check non-negativity of diagonal elements
        self.assertTrue(np.all(np.diag(killing_form) >= 0))
        
        # Check trace
        self.assertGreater(np.trace(killing_form), 0) 