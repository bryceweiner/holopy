"""
Unit tests for the physical_constants module.
"""

import unittest
import numpy as np
from holopy.constants.physical_constants import (
    PhysicalConstants, 
    get_gamma,
    get_clustering_coefficient,
    PHYSICAL_CONSTANTS
)

class TestPhysicalConstants(unittest.TestCase):
    """Tests for the PhysicalConstants class and related functions."""
    
    def setUp(self):
        """Set up the test case."""
        self.pc = PhysicalConstants()
        
        # Expected values based on physical theory
        self.expected_gamma = 1.89e-29  # Information processing rate (s⁻¹)
        self.expected_kappa_pi = np.pi**4 / 24  # Information-spacetime conversion factor
        self.expected_clustering_coefficient = 0.78125  # E8×E8 network clustering coefficient
        
    def test_singleton_pattern(self):
        """Test that PhysicalConstants follows the singleton pattern."""
        pc1 = PHYSICAL_CONSTANTS
        pc2 = PHYSICAL_CONSTANTS
        self.assertIs(pc1, pc2, "PhysicalConstants should be a singleton")
    
    def test_gamma_value(self):
        """Test that the gamma constant has the correct value."""
        # Test the default gamma value from get_gamma()
        self.assertAlmostEqual(
            get_gamma(),
            self.expected_gamma,
            delta=1e-31,
            msg="Information processing rate γ should be approximately 1.89e-29 s⁻¹"
        )
        
        # Test that gamma from Hubble parameter matches the theoretical prediction
        H = self.pc.hubble_parameter  # Current Hubble parameter
        expected_gamma = H / (8 * np.pi)  # Theoretical prediction: γ = H/8π
        self.assertAlmostEqual(
            self.pc.gamma,
            expected_gamma,
            delta=1e-31,
            msg="γ should equal H/8π"
        )
    
    def test_kappa_pi_value(self):
        """Test that the kappa_pi constant has the correct value."""
        self.assertEqual(
            self.pc.kappa_pi,
            self.expected_kappa_pi,
            "Information-spacetime conversion factor κ(π) should be π^4/24"
        )
    
    def test_clustering_coefficient_value(self):
        """Test that the clustering coefficient has the correct value."""
        self.assertEqual(
            self.pc.clustering_coefficient,
            self.expected_clustering_coefficient,
            "Clustering coefficient C(G) should be exactly 0.78125"
        )
    
    def test_hubble_gamma_relation(self):
        """Test the relation between Hubble parameter and gamma."""
        expected_ratio = 8 * np.pi
        actual_ratio = self.pc.hubble_parameter / self.pc.gamma
        self.assertAlmostEqual(
            actual_ratio,
            expected_ratio,
            delta=1e-10,
            msg="Hubble parameter should be approximately 8π times γ"
        )
    
    def test_get_gamma_function(self):
        """Test the get_gamma function."""
        # Test default value
        self.assertEqual(
            get_gamma(),
            self.expected_gamma,
            "get_gamma() should return the standard value of γ"
        )
        
        # Test with custom Hubble parameter
        H = 70.0 * 1000 / (3.086e22)  # 70 km/s/Mpc in s⁻¹
        expected_gamma = H / (8 * np.pi)
        self.assertEqual(
            get_gamma(H),
            expected_gamma,
            "get_gamma(H) should return H/8π"
        )
    
    def test_get_clustering_coefficient_function(self):
        """Test the get_clustering_coefficient function."""
        # Test E8×E8 value
        self.assertEqual(
            get_clustering_coefficient('E8xE8'),
            0.78125,
            "get_clustering_coefficient('E8xE8') should return 0.78125"
        )
        
        # Test SO(32) value
        self.assertEqual(
            get_clustering_coefficient('SO32'),
            0.75000,
            "get_clustering_coefficient('SO32') should return 0.75000"
        )
        
        # Test default value
        self.assertEqual(
            get_clustering_coefficient(),
            0.78125,
            "get_clustering_coefficient() should default to E8×E8 value"
        )
    
    def test_manifestation_scale(self):
        """Test the information manifestation scale."""
        expected_scale = self.pc.c / self.pc.gamma
        self.assertEqual(
            self.pc.manifestation_scale,
            expected_scale,
            "Manifestation scale should be c/γ"
        )
    
    def test_bulk_boundary_factor(self):
        """Test the bulk-boundary correspondence factor."""
        expected_factor = np.pi * self.pc.gamma / self.pc.hubble_parameter
        self.assertEqual(
            self.pc.bulk_boundary_factor,
            expected_factor,
            "Bulk-boundary factor should be πγ/H"
        )


if __name__ == '__main__':
    unittest.main() 