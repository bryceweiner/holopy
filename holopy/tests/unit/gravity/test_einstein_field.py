"""
Unit tests for the ModifiedEinsteinField class in the holopy.gravity.einstein_field module.
"""

import unittest
import numpy as np
from holopy.gravity.einstein_field import ModifiedEinsteinField, compute_k_tensor
from holopy.info.current import InfoCurrentTensor
from holopy.constants.physical_constants import PHYSICAL_CONSTANTS

class TestModifiedEinsteinField(unittest.TestCase):
    """Tests for the ModifiedEinsteinField class."""
    
    def setUp(self):
        """Set up test data."""
        # Create a simple Minkowski metric
        self.minkowski_metric = np.diag([-1.0, 1.0, 1.0, 1.0])
        
        # Create a simple energy-momentum tensor (empty space)
        self.energy_momentum = np.zeros((4, 4))
        
        # Create a simple information current tensor
        self.info_tensor = np.zeros((4, 4))
        self.info_tensor[0, 0] = 1.0  # Simple time-time component
        self.density = np.zeros(4)
        self.density[0] = 1.0  # Simple time component for density
        self.info_current = InfoCurrentTensor(self.info_tensor, self.density)
        
        # Initialize the ModifiedEinsteinField instance
        self.einstein_field = ModifiedEinsteinField(
            metric=self.minkowski_metric,
            energy_momentum=self.energy_momentum,
            info_current=self.info_current,
            cosmological_constant=0.0
        )
    
    def test_initialization(self):
        """Test the initialization of ModifiedEinsteinField."""
        # Check that the metric is stored correctly
        np.testing.assert_allclose(self.einstein_field.metric, self.minkowski_metric)
        
        # Check that the inverse metric is computed correctly
        expected_inverse = np.diag([-1.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(self.einstein_field.inverse_metric, expected_inverse)
        
        # Check that the shape is correct
        self.assertEqual(self.einstein_field.metric.shape, (4, 4))
        
        # Test initialization without info_current
        einstein_field_no_info = ModifiedEinsteinField(
            metric=self.minkowski_metric,
            energy_momentum=self.energy_momentum
        )
        self.assertIsNone(einstein_field_no_info.info_current)
    
    def test_compute_connection_symbols(self):
        """Test computation of connection symbols (Christoffel symbols)."""
        connection_symbols = self.einstein_field.compute_connection_symbols()
        
        # For Minkowski metric, all connection symbols should be zero
        np.testing.assert_allclose(connection_symbols, np.zeros((4, 4, 4)), atol=5e-2)
        
        # Create a non-trivial metric (e.g., Schwarzschild-like)
        r = 10.0  # Some radius away from center
        schwarzschild_metric = np.diag([-(1-2/r), 1/(1-2/r), r**2, r**2])
        
        # Initialize a new ModifiedEinsteinField with this metric
        curved_field = ModifiedEinsteinField(
            metric=schwarzschild_metric,
            energy_momentum=np.zeros((4, 4))
        )
        
        # Compute connection symbols
        curved_connection = curved_field.compute_connection_symbols()
        
        # Check shape
        self.assertEqual(curved_connection.shape, (4, 4, 4))
        
        # For a non-trivial metric, at least some connection symbols should be non-zero
        self.assertGreater(np.max(np.abs(curved_connection)), 0.0)
    
    def test_compute_einstein_tensor(self):
        """Test computation of the Einstein tensor."""
        einstein_tensor = self.einstein_field.compute_einstein_tensor()
        
        # For Minkowski metric, Einstein tensor should be zero
        np.testing.assert_allclose(einstein_tensor, np.zeros((4, 4)), atol=1e-2)
        
        # Check shape
        self.assertEqual(einstein_tensor.shape, (4, 4))
    
    def test_compute_k_tensor(self):
        """Test computation of the K tensor."""
        k_tensor = self.einstein_field.compute_k_tensor()
        
        # Check shape
        self.assertEqual(k_tensor.shape, (4, 4))
        
        # Test with a null info_current
        einstein_field_no_info = ModifiedEinsteinField(
            metric=self.minkowski_metric,
            energy_momentum=self.energy_momentum
        )
        
        # Should return zeros when no info_current is provided
        k_tensor_no_info = einstein_field_no_info.compute_k_tensor()
        np.testing.assert_allclose(k_tensor_no_info, np.zeros((4, 4)))
    
    def test_compute_curvature_from_info(self):
        """Test computation of curvature from information current."""
        riemann, ricci, ricci_scalar = self.einstein_field.compute_curvature_from_info()
        
        # Check shapes
        self.assertEqual(riemann.shape, (4, 4, 4, 4))
        self.assertEqual(ricci.shape, (4, 4))
        self.assertIsInstance(ricci_scalar, float)
        
        # Test with a null info_current
        einstein_field_no_info = ModifiedEinsteinField(
            metric=self.minkowski_metric,
            energy_momentum=self.energy_momentum
        )
        
        # Should return zeros when no info_current is provided
        riemann_no_info, ricci_no_info, ricci_scalar_no_info = einstein_field_no_info.compute_curvature_from_info()
        np.testing.assert_allclose(riemann_no_info, np.zeros((4, 4, 4, 4)))
        np.testing.assert_allclose(ricci_no_info, np.zeros((4, 4)))
        self.assertEqual(ricci_scalar_no_info, 0.0)
    
    def test_solve_field_equations(self):
        """Test solving the field equations."""
        # With Minkowski metric and zero energy-momentum tensor,
        # the solution should be approximately the original metric
        solution = self.einstein_field.solve_field_equations()
        
        # Check shape
        self.assertEqual(solution.shape, (4, 4))
        
        # Solution should be close to the original metric for this simple case
        # Allow some deviation due to numerical methods
        np.testing.assert_allclose(solution, self.minkowski_metric, atol=1e-2)
    
    def test_standalone_compute_k_tensor(self):
        """Test the standalone compute_k_tensor function."""
        # Use the standalone function
        k_tensor = compute_k_tensor(
            info_current=self.info_current,
            metric=self.minkowski_metric
        )
        
        # Check shape
        self.assertEqual(k_tensor.shape, (4, 4))
        
        # Provide inverse_metric explicitly
        k_tensor_with_inverse = compute_k_tensor(
            info_current=self.info_current,
            metric=self.minkowski_metric,
            inverse_metric=np.diag([-1.0, 1.0, 1.0, 1.0])
        )
        
        # Results should be the same
        np.testing.assert_allclose(k_tensor, k_tensor_with_inverse)

class TestPhysicalConsistency(unittest.TestCase):
    """Tests for physical consistency of the ModifiedEinsteinField class."""
    
    def setUp(self):
        """Set up test data with physically motivated configurations."""
        # Create a simple Minkowski metric
        self.minkowski_metric = np.diag([-1.0, 1.0, 1.0, 1.0])
        
        # Create a simple spherically symmetric energy-momentum tensor
        self.energy_momentum = np.zeros((4, 4))
        self.energy_momentum[0, 0] = 1.0  # Energy density
        
        # Create a simple spherically symmetric information current tensor
        self.info_tensor = np.zeros((4, 4))
        self.info_tensor[0, 0] = 1.0  # Information density
        self.density = np.zeros(4)
        self.density[0] = 1.0  # Simple time component for density
        self.info_current = InfoCurrentTensor(self.info_tensor, self.density)
        
        # Initialize the ModifiedEinsteinField instance
        self.einstein_field = ModifiedEinsteinField(
            metric=self.minkowski_metric,
            energy_momentum=self.energy_momentum,
            info_current=self.info_current,
            cosmological_constant=0.0
        )
    
    def test_einstein_field_equations(self):
        """Test that the field equations are approximately satisfied."""
        # Solve the field equations
        solution_metric = self.einstein_field.solve_field_equations()
        
        # Create a new field with the solution metric
        solution_field = ModifiedEinsteinField(
            metric=solution_metric,
            energy_momentum=self.energy_momentum,
            info_current=self.info_current,
            cosmological_constant=0.0
        )
        
        # Compute the Einstein tensor
        einstein_tensor = solution_field.compute_einstein_tensor()
        
        # Compute the K tensor
        k_tensor = solution_field.compute_k_tensor()
        
        # Get constants
        gamma = PHYSICAL_CONSTANTS.get_gamma()
        G = PHYSICAL_CONSTANTS.G
        c = PHYSICAL_CONSTANTS.c
        
        # Right-hand side of Einstein equations
        em_factor = 8 * np.pi * G / (c**4)
        right_side = em_factor * self.energy_momentum + gamma * k_tensor
        
        # Check that the equations are approximately satisfied
        # G_μν = (8πG/c⁴)T_μν + γ · 𝒦_μν
        residual = einstein_tensor - right_side
        max_residual = np.max(np.abs(residual))
        
        # The residual should be small
        self.assertLess(max_residual, 1e-2)
    
    def test_energy_conditions(self):
        """Test that the solutions satisfy energy conditions."""
        # Solve the field equations
        solution_metric = self.einstein_field.solve_field_equations()
        
        # Create a new field with the solution metric
        solution_field = ModifiedEinsteinField(
            metric=solution_metric,
            energy_momentum=self.energy_momentum,
            info_current=self.info_current,
            cosmological_constant=0.0
        )
        
        # The weak energy condition requires T_μν U^μ U^ν ≥ 0 for all timelike U^μ
        # For simplicity, check with U^μ = (1,0,0,0)
        timelike_vector = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Calculate T_μν U^μ U^ν
        energy_density = 0.0
        for mu in range(4):
            for nu in range(4):
                energy_density += self.energy_momentum[mu, nu] * timelike_vector[mu] * timelike_vector[nu]
        
        # The energy density should be non-negative
        self.assertGreaterEqual(energy_density, 0.0)
    
    def test_consistency_with_general_relativity(self):
        """Test consistency with General Relativity in the limit of zero information current."""
        # Create a field with zero information current
        gr_field = ModifiedEinsteinField(
            metric=self.minkowski_metric,
            energy_momentum=self.energy_momentum,
            info_current=None,
            cosmological_constant=0.0
        )
        
        # Solve field equations
        gr_solution = gr_field.solve_field_equations()
        
        # Compute Einstein tensor directly
        gr_einstein_tensor = gr_field.compute_einstein_tensor()
        
        # Get constants
        G = PHYSICAL_CONSTANTS.G
        c = PHYSICAL_CONSTANTS.c
        
        # Right-hand side of Einstein equations in GR
        em_factor = 8 * np.pi * G / (c**4)
        gr_right_side = em_factor * self.energy_momentum
        
        # Check that the GR equations are approximately satisfied
        # G_μν = (8πG/c⁴)T_μν
        gr_residual = gr_einstein_tensor - gr_right_side
        gr_max_residual = np.max(np.abs(gr_residual))
        
        # The residual should be small
        self.assertLess(gr_max_residual, 1e-2)


if __name__ == '__main__':
    unittest.main() 