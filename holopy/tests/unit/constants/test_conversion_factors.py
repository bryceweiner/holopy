"""
Tests for the conversion_factors module.

This module tests the functionality of conversion factors used throughout
the holographic framework, ensuring consistency between different unit systems
and physical quantities.
"""

import unittest
import numpy as np
from scipy import constants as scipy_constants

from holopy.constants.conversion_factors import (
    ConversionFactors,
    get_length_in_planck_units,
    get_time_in_planck_units,
    get_mass_in_planck_units,
    get_entropy_from_area,
    get_entropy_from_bits,
    get_information_from_mass
)
from holopy.constants.physical_constants import PhysicalConstants


class TestConversionFactors(unittest.TestCase):
    """Test suite for the conversion factors module."""

    def setUp(self):
        """Set up test fixtures."""
        self.cf = ConversionFactors()
        self.pc = PhysicalConstants()
        
        # Define small tolerance for floating point comparisons
        self.tolerance = 1e-10
        
    def test_singleton_pattern(self):
        """Test that ConversionFactors follows the singleton pattern."""
        cf1 = ConversionFactors()
        cf2 = ConversionFactors()
        self.assertIs(cf1, cf2, "ConversionFactors should use singleton pattern")
        
    def test_planck_units_consistency(self):
        """Test that Planck units are defined consistently."""
        # Test Planck length
        expected_length = np.sqrt(scipy_constants.hbar * scipy_constants.G / scipy_constants.c**3)
        self.assertAlmostEqual(self.cf.length_planck, expected_length, delta=self.tolerance * expected_length)
        
        # Test Planck time
        expected_time = np.sqrt(scipy_constants.hbar * scipy_constants.G / scipy_constants.c**5)
        self.assertAlmostEqual(self.cf.time_planck, expected_time, delta=self.tolerance * expected_time)
        
        # Test Planck mass
        expected_mass = np.sqrt(scipy_constants.hbar * scipy_constants.c / scipy_constants.G)
        self.assertAlmostEqual(self.cf.mass_planck, expected_mass, delta=self.tolerance * expected_mass)
        
        # Test Planck energy = Planck mass * c^2
        expected_energy = self.cf.mass_planck * scipy_constants.c**2
        self.assertAlmostEqual(self.cf.energy_planck, expected_energy, delta=self.tolerance * expected_energy)
        
    def test_info_related_factors(self):
        """Test information-related conversion factors."""
        # Test info_spacetime_factor = π^4/24
        self.assertAlmostEqual(self.cf.info_spacetime_factor, self.pc.kappa_pi, delta=self.tolerance)
        
        # Test entropy_area_factor = 1/4 in Planck units
        expected_factor = 1 / (4 * scipy_constants.hbar * scipy_constants.G / scipy_constants.c**3)
        self.assertAlmostEqual(self.cf.entropy_area_factor, expected_factor, delta=self.tolerance * expected_factor)
        
        # Test info_entropy_factor = ln(2)
        self.assertAlmostEqual(self.cf.info_entropy_factor, np.log(2), delta=self.tolerance)
        
    def test_holographic_gravity_factors(self):
        """Test holographic gravity conversion factors."""
        # Test info_mass_factor
        expected_factor = self.pc.gamma * scipy_constants.c**2 / scipy_constants.G
        self.assertAlmostEqual(self.cf.info_mass_factor, expected_factor, delta=self.tolerance * expected_factor)
        
        # Test info_curvature_factor
        expected_factor = 8 * np.pi * scipy_constants.G / scipy_constants.c**4
        self.assertAlmostEqual(self.cf.info_curvature_factor, expected_factor, delta=self.tolerance * expected_factor)
        
    def test_cosmology_factors(self):
        """Test cosmology conversion factors."""
        # Test hubble_gamma_ratio = 8π
        self.assertAlmostEqual(self.cf.hubble_gamma_ratio, 8 * np.pi, delta=self.tolerance)
        
        # Test redshift_distance_factor
        expected_factor = scipy_constants.c / self.pc.hubble_parameter
        self.assertAlmostEqual(self.cf.redshift_distance_factor, expected_factor, delta=self.tolerance * expected_factor)
        
    def test_e8_length_scale(self):
        """Test E8 length scale calculation."""
        expected_scale = self.cf.length_planck * np.sqrt(self.pc.kappa_pi)
        self.assertAlmostEqual(self.cf.e8_length_scale, expected_scale, delta=self.tolerance * expected_scale)
        
    def test_get_length_in_planck_units(self):
        """Test conversion of SI length to Planck units."""
        test_length_si = 1.0  # 1 meter
        expected_planck_units = test_length_si / self.cf.length_planck
        self.assertAlmostEqual(get_length_in_planck_units(test_length_si), expected_planck_units,
                               delta=self.tolerance * expected_planck_units)
        
    def test_get_time_in_planck_units(self):
        """Test conversion of SI time to Planck units."""
        test_time_si = 1.0  # 1 second
        expected_planck_units = test_time_si / self.cf.time_planck
        self.assertAlmostEqual(get_time_in_planck_units(test_time_si), expected_planck_units,
                               delta=self.tolerance * expected_planck_units)
        
    def test_get_mass_in_planck_units(self):
        """Test conversion of SI mass to Planck units."""
        test_mass_si = 1.0  # 1 kilogram
        expected_planck_units = test_mass_si / self.cf.mass_planck
        self.assertAlmostEqual(get_mass_in_planck_units(test_mass_si), expected_planck_units,
                               delta=self.tolerance * expected_planck_units)
        
    def test_get_entropy_from_area(self):
        """Test calculation of entropy from area."""
        test_area_si = 1.0  # 1 square meter
        expected_entropy = test_area_si * self.cf.entropy_area_factor
        self.assertAlmostEqual(get_entropy_from_area(test_area_si), expected_entropy,
                               delta=self.tolerance * expected_entropy)
        
    def test_get_entropy_from_bits(self):
        """Test conversion of information bits to entropy."""
        test_bits = 10.0  # 10 bits
        expected_entropy = test_bits * self.cf.info_entropy_factor
        self.assertAlmostEqual(get_entropy_from_bits(test_bits), expected_entropy,
                               delta=self.tolerance * expected_entropy)
        
    def test_get_information_from_mass(self):
        """Test calculation of information processing capacity from mass and time."""
        test_mass_si = 1.0  # 1 kilogram
        test_time_si = 1.0  # 1 second
        expected_info = test_mass_si * test_time_si / self.cf.info_mass_factor
        self.assertAlmostEqual(get_information_from_mass(test_mass_si, test_time_si), expected_info,
                               delta=self.tolerance * expected_info)
        
    def test_natural_unit_conversions(self):
        """Test natural unit conversion factors."""
        # Test hbar_to_si = ħ in J·s
        self.assertEqual(self.cf.hbar_to_si, scipy_constants.hbar)
        
        # Test c_to_si = c in m/s
        self.assertEqual(self.cf.c_to_si, scipy_constants.c)
        
        # Test natural_to_si_energy = ħc in J·m
        expected_energy = scipy_constants.hbar * scipy_constants.c
        self.assertAlmostEqual(self.cf.natural_to_si_energy, expected_energy, 
                              delta=self.tolerance * expected_energy)
        
        # Test natural_to_si_length = ħ/(c·m_P) in m
        expected_length = scipy_constants.hbar / (scipy_constants.c * self.cf.mass_planck)
        self.assertAlmostEqual(self.cf.natural_to_si_length, expected_length, 
                              delta=self.tolerance * expected_length) 