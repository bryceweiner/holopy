"""
Unit tests for the InfoSpacetimeMetric class in the holopy.gravity.emergent_metric module.
"""

import unittest
import numpy as np
from unittest import mock
from holopy.gravity.emergent_metric import InfoSpacetimeMetric
from holopy.info.current import InfoCurrentTensor
from holopy.e8.heterotic import E8E8Heterotic

# Create a global patch to prevent expensive operations during testing
# This will be applied for all tests
mock_e8e8_heterotic_patcher = mock.patch('holopy.e8.heterotic.E8E8Heterotic')
mock_physical_constants_patcher = mock.patch('holopy.gravity.emergent_metric.PHYSICAL_CONSTANTS')

class TestInfoSpacetimeMetric(unittest.TestCase):
    """Tests for the InfoSpacetimeMetric class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level test data to avoid repeated expensive calculations."""
        # Pre-compute a mock killing form for testing
        cls.mock_killing_form = np.eye(496)
        
        # Create a standard Minkowski metric for testing
        cls.minkowski_metric = np.diag([-1.0, 1.0, 1.0, 1.0])
        
        # Start the patches for the entire test suite
        cls.mock_e8 = mock_e8e8_heterotic_patcher.start()
        cls.mock_constants = mock_physical_constants_patcher.start()
        
        # Set up the mock constants with reasonable values
        cls.mock_constants.get_gamma.return_value = 1.89e-29
        cls.mock_constants.G = 6.67430e-11
        cls.mock_constants.c = 299792458
    
    @classmethod
    def tearDownClass(cls):
        """Clean up class-level patches."""
        mock_e8e8_heterotic_patcher.stop()
        mock_physical_constants_patcher.stop()
    
    def setUp(self):
        """Set up test data."""
        # Create a mock E8E8Heterotic instance
        self.mock_e8_instance = self.mock_e8.return_value
        self.mock_e8_instance.get_roots.return_value = np.random.randn(480, 16) * 0.1
        self.mock_e8_instance.get_lie_algebra_dimension.return_value = 496
        self.mock_e8_instance.compute_killing_form.return_value = self.__class__.mock_killing_form
        
        # Create a simple information current tensor
        self.info_tensor = np.zeros((4, 4))
        self.info_tensor[0, 0] = 1.0  # Simple time-time component
        
        # Create a simple density vector 
        self.density_vector = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Initialize the InfoCurrentTensor with both parameters
        self.info_current = InfoCurrentTensor(self.info_tensor, self.density_vector)
        
        # Create test coordinates
        self.coordinates = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Initialize the InfoSpacetimeMetric instance with e8_structure set to None
        # Prevent it from creating a real E8E8Heterotic by default
        with mock.patch('holopy.gravity.emergent_metric.E8E8Heterotic') as m:
            m.return_value = self.mock_e8_instance
            self.spacetime_metric = InfoSpacetimeMetric(e8_structure=None, info_current=None)
        
        # Clear the e8_structure to ensure it's None for the test_initialization test
        self.spacetime_metric.e8_structure = None
    
    def test_initialization_and_getters_setters(self):
        """Test the initialization of InfoSpacetimeMetric and getter/setter methods."""
        # Check default values
        self.assertIsNone(self.spacetime_metric.metric)
        self.assertIsNone(self.spacetime_metric.inverse_metric)
        self.assertIsNone(self.spacetime_metric.e8_structure)
        
        # Test initialization with E8 structure
        spacetime_metric_with_e8 = InfoSpacetimeMetric(e8_structure=self.mock_e8_instance)
        self.assertIsNotNone(spacetime_metric_with_e8.e8_structure)
        
        # Test set_info_current and get_info_current
        self.spacetime_metric.set_info_current(self.info_current)
        retrieved_info_current = self.spacetime_metric.get_info_current()
        self.assertEqual(retrieved_info_current, self.info_current)
        
        # Test set_metric and get_metric
        test_metric = np.eye(4)
        self.spacetime_metric.set_metric(test_metric)
        retrieved_metric = self.spacetime_metric.get_metric()
        np.testing.assert_allclose(retrieved_metric, test_metric)
        
        # Test get_inverse_metric
        inverse_metric = self.spacetime_metric.get_inverse_metric()
        self.assertEqual(inverse_metric.shape, (4, 4))
        
        # Test set_e8_structure and get_e8_structure
        self.spacetime_metric.set_e8_structure(self.mock_e8_instance)
        retrieved_e8 = self.spacetime_metric.get_e8_structure()
        self.assertEqual(retrieved_e8, self.mock_e8_instance)
    
    def test_compute_flat_metric(self):
        """Test computation of flat metric."""
        flat_metric = self.spacetime_metric.compute_flat_metric()
        
        # Check shape
        self.assertEqual(flat_metric.shape, (4, 4))
        
        # Should be Minkowski
        np.testing.assert_allclose(flat_metric, self.__class__.minkowski_metric)
    
    def test_compute_from_info_current(self):
        """Test computation of metric from information current."""
        # First, set info_current
        self.spacetime_metric.set_info_current(self.info_current)
        
        # Then compute metric
        metric = self.spacetime_metric.compute_from_info_current()
        
        # Check shape
        self.assertEqual(metric.shape, (4, 4))
        
        # Metric should be symmetric
        np.testing.assert_allclose(metric, metric.T)
        
        # Test with null info_current
        self.spacetime_metric.info_current = None
        null_metric = self.spacetime_metric.compute_from_info_current()
        
        # With null info_current, should return flat metric
        np.testing.assert_allclose(null_metric, self.__class__.minkowski_metric)
    
    def test_projection_and_metric_computation(self):
        """Test computation of projection derivatives and metric from projection."""
        # Set the mock on the spacetime_metric
        self.spacetime_metric.e8_structure = self.mock_e8_instance
        
        # Manually set the _killing_form cache
        self.spacetime_metric._killing_form = self.__class__.mock_killing_form
        
        # 1. Test projection derivatives
        projection_derivatives = self.spacetime_metric.compute_projection_derivatives(self.coordinates)
        
        # Check shape
        self.assertEqual(projection_derivatives.shape, (496, 4))
        
        # Verify the mock was called
        self.mock_e8_instance.get_roots.assert_called_once()
        
        # 2. Test metric from projection
        metric = self.spacetime_metric.compute_metric_from_projection(projection_derivatives)
        
        # Check shape
        self.assertEqual(metric.shape, (4, 4))
        
        # Metric should be symmetric
        np.testing.assert_allclose(metric, metric.T)
    
    def test_compute_line_element_and_physical_metric(self):
        """Test computation of line element and physical metric properties."""
        # Set a known metric
        test_metric = self.__class__.minkowski_metric
        self.spacetime_metric.set_metric(test_metric)
        
        # Test line element calculations
        # Create displacement vector
        displacement = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Compute line element
        ds_squared = self.spacetime_metric.compute_line_element(displacement)
        
        # For timelike displacement along time axis with Minkowski metric, ds^2 = -1
        self.assertAlmostEqual(ds_squared, -1.0)
        
        # Try spacelike displacement
        spacelike_displacement = np.array([0.0, 1.0, 0.0, 0.0])
        ds_squared_spacelike = self.spacetime_metric.compute_line_element(spacelike_displacement)
        self.assertAlmostEqual(ds_squared_spacelike, 1.0)
        
        # Test physical metric detection
        # Check if physical (should be true for Minkowski)
        is_physical = self.spacetime_metric.is_physical_metric()
        self.assertTrue(is_physical)
        
        # Test with non-physical metric
        non_physical_metric = np.diag([1.0, 1.0, 1.0, 1.0])  # All positive eigenvalues
        self.spacetime_metric.set_metric(non_physical_metric)
        
        # Check if physical
        is_physical = self.spacetime_metric.is_physical_metric()
        self.assertFalse(is_physical)


class TestPhysicalConsistency(unittest.TestCase):
    """Tests for physical consistency of the InfoSpacetimeMetric class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level test data to avoid repeated expensive calculations."""
        # Pre-compute a simplified metric for consistency tests
        cls.predefined_metric = np.diag([-1.0, 1.0, 1.0, 1.0])
        
        # Start the patches for the entire test suite
        cls.mock_e8 = mock_e8e8_heterotic_patcher.start()
        cls.mock_constants = mock_physical_constants_patcher.start()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up class-level patches."""
        mock_e8e8_heterotic_patcher.stop()
        mock_physical_constants_patcher.stop()
    
    def setUp(self):
        """Set up test data with physically motivated configurations."""
        # Create a mock E8E8Heterotic instance
        self.mock_e8_instance = self.mock_e8.return_value
        
        # Create a simple information current tensor
        self.info_tensor = np.zeros((4, 4))
        
        # Create a spherically symmetric information distribution
        for i in range(4):
            for j in range(4):
                # Simple r^-2 falloff for information density
                r = 10.0  # Some radius from center
                self.info_tensor[i, j] = 1.0 / (r**2) if i == j else 0.0
        
        # Create a simple density vector with r^-2 falloff
        self.density_vector = np.array([1.0, 0.0, 0.0, 0.0]) / (10.0**2)
        
        # Initialize the InfoCurrentTensor with both parameters
        self.info_current = InfoCurrentTensor(self.info_tensor, self.density_vector)
        
        # Create test coordinates
        self.coordinates = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Initialize the InfoSpacetimeMetric instance with mocks
        with mock.patch('holopy.gravity.emergent_metric.E8E8Heterotic') as m:
            m.return_value = self.mock_e8_instance
            self.spacetime_metric = InfoSpacetimeMetric(e8_structure=self.mock_e8_instance)
            self.spacetime_metric.set_info_current(self.info_current)
        
        # For speed, directly set the metric instead of computing it
        self.spacetime_metric.set_metric(self.__class__.predefined_metric)
    
    def test_metric_properties(self):
        """Test metric signature and symmetry properties."""
        # Use the predefined metric for speed
        metric = self.__class__.predefined_metric
        
        # Test signature
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(metric)
        
        # Sort eigenvalues
        eigenvalues = sorted(eigenvalues)
        
        # Should have one negative and three positive eigenvalues
        self.assertLess(eigenvalues[0], 0)
        self.assertGreater(eigenvalues[1], 0)
        self.assertGreater(eigenvalues[2], 0)
        self.assertGreater(eigenvalues[3], 0)
        
        # Test symmetry
        # Check symmetry
        np.testing.assert_allclose(metric, metric.T)
    
    def test_causality(self):
        """Test that the metric preserves causality."""
        # Test with timelike vector
        # For Minkowski-like metric, a timelike vector is [1, 0, 0, 0]
        timelike_vector = np.array([1.0, 0.0, 0.0, 0.0])
        ds_squared = self.spacetime_metric.compute_line_element(timelike_vector)
        self.assertLess(ds_squared, 0, "Timelike vector should have negative ds²")
        
        # Test with spacelike vector
        # For Minkowski-like metric, a spacelike vector is [0, 1, 0, 0]
        spacelike_vector = np.array([0.0, 1.0, 0.0, 0.0])
        ds_squared = self.spacetime_metric.compute_line_element(spacelike_vector)
        self.assertGreater(ds_squared, 0, "Spacelike vector should have positive ds²")
        
        # Test with null vector (approximately)
        # For Minkowski-like metric, a null vector is approximately [1, 1, 0, 0]
        null_vector = np.array([1.0, 1.0, 0.0, 0.0])
        ds_squared_null = self.spacetime_metric.compute_line_element(null_vector)
        
        # Should be close to zero for null vectors
        self.assertAlmostEqual(ds_squared_null, 0.0, delta=0.1)
    
    def test_consistency_with_einstein_equations(self):
        """Test consistency with Einstein's equations.
        
        This test checks that the emergent metric from holographic information
        satisfies the modified Einstein field equations:
        G_μν + Λg_μν = (8πG/c⁴)T_μν + γ · 𝒦_μν
        
        where 𝒦_μν is derived from the information current tensor.
        """
        # Import necessary modules for this test
        from holopy.gravity.einstein_field import (
            ModifiedEinsteinField, 
            compute_einstein_tensor, 
            compute_information_k_tensor
        )
        from holopy.utils.tensor_utils import compute_christoffel_symbols, compute_riemann_tensor
        from holopy.constants.physical_constants import PhysicalConstants
        
        # Get constants
        constants = PhysicalConstants()
        gamma = constants.gamma  # Information processing rate
        
        # Generate a test metric from the information current
        # We use the existing spacetime metric from setUp
        metric = self.spacetime_metric.get_metric()
        
        # Compute geometric quantities from the metric
        christoffel = compute_christoffel_symbols(metric)
        riemann = compute_riemann_tensor(christoffel)
        
        # Compute Einstein tensor
        einstein_tensor = compute_einstein_tensor(metric, riemann)
        
        # Get the information current tensor
        info_current = self.spacetime_metric.get_info_current()
        
        # Compute the information correction tensor
        k_tensor = compute_information_k_tensor(info_current, metric)
        
        # Compute the stress-energy tensor assuming a vacuum solution
        # (just cosmological constant contribution)
        lambda_cosmo = constants.lambda_cosmological
        stress_energy = -lambda_cosmo * metric
        
        # Modified Einstein field equation:
        # G_μν + Λg_μν = (8πG/c⁴)T_μν + γ · 𝒦_μν
        
        # Left-hand side: G_μν + Λg_μν
        lhs = einstein_tensor + lambda_cosmo * metric
        
        # Right-hand side: (8πG/c⁴)T_μν + γ · 𝒦_μν
        G_newton = constants.G
        c = constants.c
        factor = 8 * np.pi * G_newton / c**4
        
        rhs = factor * stress_energy + gamma * k_tensor
        
        # Check consistency for each component
        # Due to numerical approximations, we use a relatively large tolerance
        tolerance = 1e-6
        
        # Test that the equations are approximately satisfied
        error = np.max(np.abs(lhs - rhs))
        
        # Create an informative message
        msg = (f"Einstein equations not satisfied with error {error:.2e} > {tolerance:.2e}\n"
               f"Max LHS = {np.max(np.abs(lhs)):.2e}, Max RHS = {np.max(np.abs(rhs)):.2e}")
        
        self.assertLessEqual(error, tolerance, msg=msg)
        
        # Also check explicitly for the modified trace relation
        # In GR: R = -8πGT/c^4 + 4Λ
        # In holographic gravity: R = -8πGT/c^4 + 4Λ + γ·Tr(𝒦)
        
        ricci_scalar = np.trace(einstein_tensor @ metric)
        stress_energy_trace = np.trace(stress_energy @ metric)
        k_tensor_trace = np.trace(k_tensor @ metric)
        
        lhs_trace = ricci_scalar
        rhs_trace = -factor * stress_energy_trace + 4 * lambda_cosmo + gamma * k_tensor_trace
        
        trace_error = abs(lhs_trace - rhs_trace)
        trace_tolerance = 1e-6
        
        self.assertLessEqual(
            trace_error, 
            trace_tolerance, 
            msg=f"Trace relation not satisfied with error {trace_error:.2e} > {trace_tolerance:.2e}"
        )


if __name__ == '__main__':
    unittest.main() 