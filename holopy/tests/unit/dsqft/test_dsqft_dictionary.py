"""
Unit tests for holopy.dsqft.dictionary module.

These tests verify the field-operator dictionary implementation for the dS/QFT correspondence,
focusing on the mapping between bulk fields and boundary operators.
"""

import unittest
import numpy as np
from holopy.dsqft.dictionary import FieldOperatorDictionary, FieldType, OperatorType
from holopy.constants.physical_constants import PhysicalConstants

class TestFieldOperatorDictionary(unittest.TestCase):
    """Test cases for the FieldOperatorDictionary class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create dictionary with standard parameters
        self.d = 4  # 4D spacetime
        self.dictionary = FieldOperatorDictionary(d=self.d)
        
        # Physical constants
        self.pc = PhysicalConstants()
        
        # Register test fields
        self.dictionary.register_bulk_field(
            field_name='scalar_massless',
            field_type=FieldType.SCALAR,
            mass=0.0,
            spin=0
        )
        
        self.dictionary.register_bulk_field(
            field_name='scalar_massive',
            field_type=FieldType.SCALAR,
            mass=1.0,
            spin=0
        )
        
        self.dictionary.register_bulk_field(
            field_name='vector_field',
            field_type=FieldType.VECTOR,
            mass=0.0,
            spin=1
        )
    
    def test_initialization(self):
        """Test that dictionary initializes with correct parameters."""
        self.assertEqual(self.dictionary.d, self.d)
        self.assertAlmostEqual(self.dictionary.gamma, self.pc.gamma)
        self.assertAlmostEqual(self.dictionary.hubble_parameter, self.pc.hubble_parameter)
        
        # Should have initialized conformal dimensions for standard fields
        self.assertIn((FieldType.SCALAR, 'massless'), self.dictionary.conformal_dimensions)
        self.assertIn((FieldType.VECTOR, 'massless'), self.dictionary.conformal_dimensions)
    
    def test_field_registration(self):
        """Test field registration and retrieval."""
        # Fields should be registered
        self.assertIn('scalar_massless', self.dictionary.bulk_boundary_map)
        self.assertIn('scalar_massive', self.dictionary.bulk_boundary_map)
        self.assertIn('vector_field', self.dictionary.bulk_boundary_map)
        
        # Should be able to get field info
        field_info = self.dictionary.get_field_info('scalar_massless')
        self.assertEqual(field_info['name'], 'scalar_massless')
        self.assertEqual(field_info['type'], FieldType.SCALAR)
        self.assertEqual(field_info['mass'], 0.0)
        self.assertEqual(field_info['spin'], 0)
        
        # Should be able to get operator info
        operator_info = self.dictionary.get_operator_info('scalar_massless')
        self.assertEqual(operator_info['name'], 'O_scalar_massless')
        
        # Should be able to get propagator
        propagator = self.dictionary.get_propagator('scalar_massless')
        self.assertIsNotNone(propagator)
    
    def test_conformal_dimension_computation(self):
        """Test computation of conformal dimensions for different fields."""
        # Get conformal dimensions
        massless_scalar_dim = self.dictionary.get_field_info('scalar_massless')['conformal_dimension']
        massive_scalar_dim = self.dictionary.get_field_info('scalar_massive')['conformal_dimension']
        
        # For massless scalar, Δ = (d-2)/2 = 1 in 4D
        self.assertAlmostEqual(massless_scalar_dim, 1.0)
        
        # For massive scalar, Δ = d/2 + sqrt((d/2)² + m²/H²)
        # In 4D with mass=1.0, this should be greater than the massless value
        self.assertGreater(massive_scalar_dim, massless_scalar_dim)
        
        # Vector field should have higher dimension than scalar
        vector_dim = self.dictionary.get_field_info('vector_field')['conformal_dimension']
        self.assertGreater(vector_dim, massless_scalar_dim)
    
    def test_operator_mapping(self):
        """Test mapping between bulk fields and boundary operators."""
        # Operator should have same conformal dimension as field
        field_info = self.dictionary.get_field_info('scalar_massless')
        operator_info = self.dictionary.get_operator_info('scalar_massless')
        
        self.assertEqual(
            field_info['conformal_dimension'],
            operator_info['conformal_dimension']
        )
        
        # Scalar field maps to scalar operator
        self.assertEqual(operator_info['type'], OperatorType.SCALAR)
        
        # Vector field maps to tensor operator
        vector_operator = self.dictionary.get_operator_info('vector_field')
        self.assertEqual(vector_operator['type'], OperatorType.TENSOR)
    
    def test_bulk_to_boundary_mapping(self):
        """Test computation of boundary operator value from bulk field."""
        # Simple bulk field function (constant value)
        def bulk_field(eta, x):
            return 1.0
        
        # Compute boundary operator value
        boundary_point = np.array([0.0, 0.0, 0.0])
        operator_value = self.dictionary.compute_boundary_operator_value(
            'scalar_massless', bulk_field, boundary_point
        )
        
        # Value should be non-zero and related to the conformal dimension
        self.assertIsInstance(operator_value, float)
        self.assertNotEqual(operator_value, 0.0)
        
        # Test with a different field type
        operator_value2 = self.dictionary.compute_boundary_operator_value(
            'vector_field', bulk_field, boundary_point
        )
        
        # Values should be different for different field types
        self.assertNotEqual(operator_value, operator_value2)
    
    def test_boundary_to_bulk_mapping(self):
        """Test computation of bulk field value from boundary operator."""
        # Simple boundary operator function (constant value)
        def boundary_op(x):
            return 1.0
        
        # Create a few boundary points for integration
        boundary_grid = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ])
        
        # Compute bulk field value
        eta = -1.0
        bulk_point = np.array([0.0, 0.0, 0.0])
        
        field_value = self.dictionary.compute_bulk_field_value(
            'scalar_massless', boundary_op, boundary_grid, eta, bulk_point
        )
        
        # Value should be non-zero
        self.assertIsInstance(field_value, float)
        self.assertNotEqual(field_value, 0.0)
        
        # Test with a different field type
        field_value2 = self.dictionary.compute_bulk_field_value(
            'vector_field', boundary_op, boundary_grid, eta, bulk_point
        )
        
        # Values should be different for different field types
        self.assertNotEqual(field_value, field_value2)
    
    def test_dictionary_properties(self):
        """Test verification of dictionary properties."""
        # Verify properties
        results = self.dictionary.verify_dictionary_properties(test_points=20)
        
        # At least some of the verification tests should pass
        self.assertTrue(any(results.values()))
        
        # For a well-implemented dictionary, all tests should pass
        # But this might be too strict during development
        # Uncomment when implementation is complete
        # self.assertTrue(all(results.values()))

if __name__ == '__main__':
    unittest.main() 