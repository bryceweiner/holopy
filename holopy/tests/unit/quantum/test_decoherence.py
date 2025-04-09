"""
Unit tests for the decoherence module.

These tests verify that the implementation of the decoherence functions
correctly models quantum decoherence in the holographic framework.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from holopy.constants.physical_constants import PhysicalConstants, PHYSICAL_CONSTANTS
from holopy.quantum.modified_schrodinger import WaveFunction
from holopy.quantum.decoherence import (
    coherence_decay,
    spatial_complexity,
    decoherence_rate,
    decoherence_timescale,
    decoherence_length,
    decoherence_evolution
)

class TestCoherenceDecay(unittest.TestCase):
    """Tests for the coherence_decay function."""
    
    def test_basic_coherence_decay(self):
        """Test that coherence decays correctly with time and distance."""
        # Initial coherence
        rho_0 = 1.0
        
        # Test at different times
        times = np.array([0.0, 1.0, 10.0, 100.0])
        
        # Fixed positions
        x1 = 0.0
        x2 = 1.0
        
        # Use a high gamma value for testing
        test_gamma = 0.1
        
        # Calculate coherence decay
        coherence = coherence_decay(rho_0, times, x1, x2, gamma=PHYSICAL_CONSTANTS.get_gamma())
        
        # Coherence should decay exponentially
        expected_coherence = rho_0 * np.exp(-PHYSICAL_CONSTANTS.get_gamma() * times * (x2 - x1)**2)
        
        # Check that the coherence decays as expected
        self.assertTrue(np.allclose(coherence, expected_coherence))
        
        # At time 0, coherence should be rho_0
        self.assertEqual(coherence[0], rho_0)
        
        # Coherence should decrease monotonically with time
        self.assertTrue(np.all(np.diff(coherence) <= 0))
    
    def test_position_dependence(self):
        """Test that coherence decay depends on the distance between positions."""
        # Initial coherence
        rho_0 = 1.0
        
        # Fixed time
        t = 1.0
        
        # Different positions
        x1 = 0.0
        x2_values = np.array([0.0, 1.0, 2.0, 3.0])
        
        # Use a high gamma value for testing
        test_gamma = 0.1
        
        # Calculate coherence decay
        coherence = coherence_decay(rho_0, t, x1, x2_values, gamma=PHYSICAL_CONSTANTS.get_gamma())
        
        # Coherence should decay exponentially with distance squared
        expected_coherence = rho_0 * np.exp(-PHYSICAL_CONSTANTS.get_gamma() * t * (x2_values - x1)**2)
        
        # Check that the coherence decays as expected
        self.assertTrue(np.allclose(coherence, expected_coherence))
        
        # When x1 = x2, coherence should decay only due to time
        self.assertAlmostEqual(coherence[0], rho_0 * np.exp(-PHYSICAL_CONSTANTS.get_gamma() * t * 0))
        
        # Coherence should decrease monotonically with distance
        self.assertTrue(np.all(np.diff(coherence) <= 0))
    
    def test_vector_positions(self):
        """Test that coherence decay works with vector positions."""
        # Initial coherence
        rho_0 = 1.0
        
        # Fixed time
        t = 1.0
        
        # Vector positions in 3D
        x1 = np.array([0.0, 0.0, 0.0])
        x2 = np.array([1.0, 1.0, 1.0])
        
        # Use a high gamma value for testing
        test_gamma = 0.1
        
        # Calculate coherence decay
        coherence = coherence_decay(rho_0, t, x1, x2, gamma=PHYSICAL_CONSTANTS.get_gamma())
        
        # Expected coherence
        # Distance squared is |x2 - x1|^2 = 3.0
        expected_coherence = rho_0 * np.exp(-PHYSICAL_CONSTANTS.get_gamma() * t * 3.0)
        
        # Check that the coherence decays as expected
        # Use numpy's testing functions for arrays
        np.testing.assert_almost_equal(coherence, expected_coherence)

class TestSpatialComplexity(unittest.TestCase):
    """Tests for the spatial_complexity function."""
    
    def test_gaussian_spatial_complexity(self):
        """Test that spatial complexity is calculated correctly for a Gaussian."""
        # Create a Gaussian wavefunction
        def gaussian(x):
            return np.exp(-x**2)
        
        # Calculate spatial complexity
        domain = [(-5.0, 5.0)]
        complexity = spatial_complexity(gaussian, domain, grid_size=1000)
        
        # For a Gaussian exp(-x^2), the complexity is 1.0
        # This is because ∫ 4x^2 * exp(-2x^2) dx = 1.0
        self.assertAlmostEqual(complexity, 1.0, places=2)
    
    def test_wavefunction_object(self):
        """Test that spatial complexity works with a WaveFunction object."""
        # Create a Gaussian wavefunction
        def gaussian(x):
            return np.exp(-x**2)
        
        # Create a WaveFunction object
        domain = [(-5.0, 5.0)]
        grid_size = 1000
        wf = WaveFunction(initial_function=gaussian, domain=domain, grid_size=grid_size)
        
        # Calculate spatial complexity from the WaveFunction object
        complexity = spatial_complexity(wf)
        
        # For a Gaussian exp(-x^2), the complexity is 1.0
        self.assertAlmostEqual(complexity, 1.0, places=2)
    
    def test_plane_wave_complexity(self):
        """Test that spatial complexity is calculated correctly for a plane wave."""
        # Create a plane wave with wavevector k
        def plane_wave(x, k=1.0):
            return np.exp(1j * k * x)
        
        # Calculate spatial complexity
        domain = [(-np.pi, np.pi)]
        grid_size = 1000
        k = 1.0
        complexity = spatial_complexity(lambda x: plane_wave(x, k), domain, grid_size=grid_size)
        
        # For a plane wave exp(ikx), the complexity is k^2
        # This is because |∇ψ|² = k²|ψ|² = k²
        self.assertAlmostEqual(complexity, k**2, places=2)

class TestDecoherenceRate(unittest.TestCase):
    """Tests for the decoherence_rate function."""
    
    def test_rate_scaling(self):
        """Test that decoherence rate scales correctly with system size."""
        # Test different system sizes
        system_sizes = np.array([1e-6, 1e-5, 1e-4, 1e-3])
        
        # Calculate decoherence rates
        rates = np.array([decoherence_rate(size) for size in system_sizes])
        
        # Rates should scale as L^-2
        # If we double the size, rate should decrease by factor of 4
        ratios = rates[:-1] / rates[1:]
        expected_ratios = (system_sizes[1:] / system_sizes[:-1])**2
        
        # Check that the rates scale as expected
        self.assertTrue(np.allclose(ratios, expected_ratios))
    
    def test_timescale_consistency(self):
        """Test that decoherence timescale is consistent with rate."""
        # Test a system size
        system_size = 1e-6  # 1 micrometer
        
        # Calculate rate and timescale
        rate = decoherence_rate(system_size)
        timescale = decoherence_timescale(system_size)
        
        # Timescale should be 1/rate
        self.assertAlmostEqual(timescale, 1.0 / rate)
    
    def test_length_calculation(self):
        """Test that decoherence length is calculated correctly."""
        # Test a timescale
        time = 1.0  # 1 second
        
        # Calculate decoherence length
        length = decoherence_length(time)
        
        # The calculated length should give the correct timescale when used
        timescale = decoherence_timescale(length)
        
        # Timescale should match the input time
        self.assertAlmostEqual(timescale, time, places=6)

class TestDecoherenceEvolution(unittest.TestCase):
    """Tests for the decoherence_evolution function."""
    
    def test_norm_decay(self):
        """Test that the norm decays correctly during pure decoherence evolution."""
        # Create a Gaussian wavefunction
        def gaussian(x):
            return np.exp(-x**2)
        
        # Create a WaveFunction object
        domain = [(-5.0, 5.0)]
        grid_size = 100
        wf = WaveFunction(initial_function=gaussian, domain=domain, grid_size=grid_size)
        
        # Set a high gamma value for testing
        test_gamma = 1.0
        
        # Run the decoherence evolution
        t_span = [0.0, 1.0]
        results = decoherence_evolution(wf, t_span, dt=0.1, gamma=test_gamma)
        
        # Check that the norm decreases monotonically
        self.assertTrue(np.all(np.diff(results['norm']) <= 0))
        
        # Initial norm should be 1.0
        self.assertAlmostEqual(results['norm'][0], 1.0)
        
        # Final norm should be less than initial norm
        self.assertLess(results['norm'][-1], 1.0)
    
    def test_decoherence_decay(self):
        """Test that the decoherence measure itself decays during evolution."""
        # Create a Gaussian wavefunction
        def gaussian(x):
            return np.exp(-x**2)
        
        # Create a WaveFunction object
        domain = [(-5.0, 5.0)]
        grid_size = 100
        wf = WaveFunction(initial_function=gaussian, domain=domain, grid_size=grid_size)
        
        # Set a high gamma value for testing
        test_gamma = 1.0
        
        # Run the decoherence evolution
        t_span = [0.0, 1.0]
        results = decoherence_evolution(wf, t_span, dt=0.1, gamma=test_gamma)
        
        # Check that the decoherence measure decreases monotonically
        self.assertTrue(np.all(np.diff(results['decoherence']) <= 0))
        
        # Initial decoherence should match the spatial complexity of the Gaussian
        # For a Gaussian exp(-x^2), this is approximately 1.0
        self.assertAlmostEqual(results['decoherence'][0], 1.0, places=1)
        
        # Final decoherence should be less than initial
        self.assertLess(results['decoherence'][-1], results['decoherence'][0])

if __name__ == '__main__':
    unittest.main() 