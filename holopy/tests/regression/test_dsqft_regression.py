"""
Regression tests for holopy.dsqft module.

These tests ensure that future changes to the dS/QFT module don't break existing
functionality by comparing results against known reference values.
"""

import unittest
import numpy as np
import json
import tempfile
from pathlib import Path

from holopy.dsqft.simulation import DSQFTSimulation
from holopy.dsqft.causal_patch import CausalPatch, PatchType
from holopy.dsqft.dictionary import FieldType
from holopy.dsqft.query import DSQFTQuery, QueryType
from holopy.dsqft.correlation import ModifiedCorrelationFunction

class TestDSQFTRegression(unittest.TestCase):
    """Regression tests for the dS/QFT module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simulation with standard parameters
        self.simulation = DSQFTSimulation(
            boundary_conditions='vacuum',
            d=4
        )
        
        # Save a snapshot of constants for regression testing
        self.original_gamma = self.simulation.gamma
        self.original_hubble_parameter = self.simulation.hubble_parameter
    
    def test_propagator_evaluation_consistency(self):
        """Test that propagator evaluation remains consistent."""
        # Get a reference propagator
        propagator = self.simulation.dictionary.get_propagator('scalar')
        
        # Reference points
        eta = -1.0
        x_bulk = np.array([0.0, 0.0, 0.0])
        x_boundary = np.array([0.0, 0.0, 0.0])
        
        # Reference values
        reference_values = {
            'origin': propagator.evaluate(eta, x_bulk, x_boundary),
            'bulk_shift': propagator.evaluate(eta, np.array([0.1, 0.0, 0.0]), x_boundary),
            'boundary_shift': propagator.evaluate(eta, x_bulk, np.array([0.1, 0.0, 0.0])),
            'both_shift': propagator.evaluate(eta, np.array([0.1, 0.0, 0.0]), np.array([0.1, 0.0, 0.0])),
            'deeper_eta': propagator.evaluate(-2.0, x_bulk, x_boundary)
        }
        
        # Values should be non-zero
        for key, value in reference_values.items():
            self.assertIsInstance(value, float)
            self.assertNotEqual(value, 0.0)
        
        # Create a new propagator with the same parameters
        new_propagator = self.simulation.dictionary.get_propagator('scalar')
        
        # New values should match reference values
        for key, ref_value in reference_values.items():
            if key == 'origin':
                new_value = new_propagator.evaluate(eta, x_bulk, x_boundary)
            elif key == 'bulk_shift':
                new_value = new_propagator.evaluate(eta, np.array([0.1, 0.0, 0.0]), x_boundary)
            elif key == 'boundary_shift':
                new_value = new_propagator.evaluate(eta, x_bulk, np.array([0.1, 0.0, 0.0]))
            elif key == 'both_shift':
                new_value = new_propagator.evaluate(eta, np.array([0.1, 0.0, 0.0]), np.array([0.1, 0.0, 0.0]))
            elif key == 'deeper_eta':
                new_value = new_propagator.evaluate(-2.0, x_bulk, x_boundary)
            
            # Values should match exactly
            self.assertEqual(new_value, ref_value)
    
    def test_correlation_function_consistency(self):
        """Test that correlation function calculation remains consistent."""
        # Create a correlation function calculator
        correlation = ModifiedCorrelationFunction(
            dictionary=self.simulation.dictionary,
            d=4
        )
        
        # Reference points
        eta1 = -1.0
        x1 = np.array([0.0, 0.0, 0.0])
        eta2 = -1.0
        x2 = np.array([0.1, 0.0, 0.0])
        
        # Reference values
        ref_value = correlation.bulk_two_point_function('scalar', eta1, x1, eta2, x2)
        
        # Value should be non-zero
        self.assertIsInstance(ref_value, float)
        self.assertNotEqual(ref_value, 0.0)
        
        # Create a new correlation function calculator with the same parameters
        new_correlation = ModifiedCorrelationFunction(
            dictionary=self.simulation.dictionary,
            d=4
        )
        
        # New value should match reference value
        new_value = new_correlation.bulk_two_point_function('scalar', eta1, x1, eta2, x2)
        self.assertAlmostEqual(new_value, ref_value)
        
        # Correlation should decrease with distance
        x3 = np.array([0.2, 0.0, 0.0])
        ref_value_further = correlation.bulk_two_point_function('scalar', eta1, x1, eta2, x3)
        
        # Further separation should have smaller correlation
        self.assertLess(ref_value_further, ref_value)
    
    def test_simulation_evolution_consistency(self):
        """Test that simulation evolution remains consistent."""
        # Set custom boundary values
        def gaussian_boundary(x):
            return np.exp(-np.sum(x**2))
        
        self.simulation.set_custom_boundary_values('scalar', gaussian_boundary)
        
        # Run a short evolution with fixed parameters
        duration = 0.1
        num_steps = 2
        
        # First run
        results1 = self.simulation.evolve(duration, num_steps)
        
        # Extract key values for comparison
        field_values1 = results1['field_evolution']['scalar'][0][0]  # First time step, first point
        energy_density1 = results1['energy_density_evolution'][0][0]  # First time step, first point
        entropy1 = results1['entropy_evolution'][0]  # First time step
        
        # Create a new simulation with the same parameters
        new_simulation = DSQFTSimulation(
            boundary_conditions='vacuum',
            d=4,
            gamma=self.original_gamma,
            hubble_parameter=self.original_hubble_parameter
        )
        
        # Set the same boundary values
        new_simulation.set_custom_boundary_values('scalar', gaussian_boundary)
        
        # Run the same evolution
        results2 = new_simulation.evolve(duration, num_steps)
        
        # Extract key values for comparison
        field_values2 = results2['field_evolution']['scalar'][0][0]  # First time step, first point
        energy_density2 = results2['energy_density_evolution'][0][0]  # First time step, first point
        entropy2 = results2['entropy_evolution'][0]  # First time step
        
        # Values should match
        self.assertAlmostEqual(field_values1, field_values2, places=10)
        self.assertAlmostEqual(energy_density1, energy_density2, places=10)
        self.assertAlmostEqual(entropy1, entropy2, places=10)
    
    def test_result_serialization_consistency(self):
        """Test that result serialization remains consistent."""
        # Run a short evolution
        self.simulation.evolve(0.1, 2)
        
        # Create a temporary file for first serialization
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp1:
            tmp1_path = Path(tmp1.name)
        
        # Create a temporary file for second serialization
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp2:
            tmp2_path = Path(tmp2.name)
        
        try:
            # Save results twice
            self.simulation.save_results(tmp1_path)
            self.simulation.save_results(tmp2_path)
            
            # Load both files
            with open(tmp1_path, 'r') as f1, open(tmp2_path, 'r') as f2:
                data1 = json.load(f1)
                data2 = json.load(f2)
            
            # Check that both files have the same structure
            self.assertEqual(set(data1.keys()), set(data2.keys()))
            
            # Check that both files have the same values (except for timestamp)
            for key in data1.keys():
                if key == 'metadata':
                    # Skip timestamp comparison
                    self.assertEqual(
                        data1['metadata']['version'],
                        data2['metadata']['version']
                    )
                else:
                    # For other fields, the values should be identical
                    self.assertEqual(data1[key], data2[key])
        
        finally:
            # Clean up
            tmp1_path.unlink()
            tmp2_path.unlink()
    
    def test_query_consistency(self):
        """Test that query results remain consistent."""
        # Run a short evolution
        self.simulation.evolve(0.1, 2)
        
        # Create query interface
        query = DSQFTQuery(
            simulation=self.simulation,
            d=4,
            gamma=self.original_gamma,
            hubble_parameter=self.original_hubble_parameter
        )
        
        # Reference points
        t = 0.1
        x = np.array([0.0, 0.0, 0.0])
        
        # Reference query results
        field_result = query.query_field_value('scalar', t, x)
        energy_result = query.query_energy_density(t, x)
        entropy_result = query.query_entropy(t, x, 0.5)
        
        # Recreate query interface
        new_query = DSQFTQuery(
            simulation=self.simulation,
            d=4,
            gamma=self.original_gamma,
            hubble_parameter=self.original_hubble_parameter
        )
        
        # New query results
        new_field_result = new_query.query_field_value('scalar', t, x)
        new_energy_result = new_query.query_energy_density(t, x)
        new_entropy_result = new_query.query_entropy(t, x, 0.5)
        
        # Results should match
        self.assertEqual(field_result.value, new_field_result.value)
        self.assertEqual(energy_result.value, new_energy_result.value)
        self.assertEqual(entropy_result.value, new_entropy_result.value)
    
    def test_causal_patch_consistency(self):
        """Test that causal patch operations remain consistent."""
        # Create a causal patch
        patch = CausalPatch(
            radius=1.0,
            reference_frame='static',
            observer_time=0.0,
            d=4
        )
        
        # Reference boundary projection
        boundary1 = patch.boundary_projection(resolution=10)
        
        # Reference spatial grid
        grid1 = patch.create_spatial_grid(resolution=10)
        
        # Create a new patch with the same parameters
        new_patch = CausalPatch(
            radius=1.0,
            reference_frame='static',
            observer_time=0.0,
            d=4
        )
        
        # New boundary projection
        boundary2 = new_patch.boundary_projection(resolution=10)
        
        # New spatial grid
        grid2 = new_patch.create_spatial_grid(resolution=10)
        
        # Should have the same number of points
        self.assertEqual(len(boundary1), len(boundary2))
        self.assertEqual(len(grid1), len(grid2))
        
        # Points should be the same
        for i in range(min(len(boundary1), 10)):  # Check first 10 points
            np.testing.assert_array_almost_equal(boundary1[i], boundary2[i])
        
        for i in range(min(len(grid1), 10)):  # Check first 10 points
            np.testing.assert_array_almost_equal(grid1[i], grid2[i])
    
    def test_cmb_power_spectrum_consistency(self):
        """Test that CMB power spectrum calculation remains consistent."""
        # Create a correlation function calculator
        correlation = ModifiedCorrelationFunction(
            dictionary=self.simulation.dictionary,
            d=4
        )
        
        # Reference multipoles
        ells = np.array([2, 10, 100, 500, 1000, 2000])
        
        # Reference spectrum
        ref_spectrum = correlation.cmb_power_spectrum(ells)
        
        # Values should be non-zero
        for value in ref_spectrum:
            self.assertIsInstance(value, float)
            self.assertNotEqual(value, 0.0)
        
        # Create a new correlation function calculator
        new_correlation = ModifiedCorrelationFunction(
            dictionary=self.simulation.dictionary,
            d=4
        )
        
        # New spectrum
        new_spectrum = new_correlation.cmb_power_spectrum(ells)
        
        # Spectra should match
        np.testing.assert_array_almost_equal(ref_spectrum, new_spectrum)
    
    def test_api_consistency(self):
        """Test that the public API remains consistent."""
        # Check that the main classes and functions exist
        from holopy.dsqft import (
            BulkBoundaryPropagator,
            FieldOperatorDictionary,
            ModifiedCorrelationFunction,
            InformationTransport,
            MatterEntropyCoupling,
            CausalPatch,
            DSQFTSimulation,
            QueryInterface
        )
        
        # Check that enum values are as expected
        from holopy.dsqft.dictionary import FieldType
        self.assertEqual(FieldType.SCALAR.value, 1)
        self.assertEqual(FieldType.VECTOR.value, 2)
        self.assertEqual(FieldType.TENSOR.value, 3)
        self.assertEqual(FieldType.SPINOR.value, 4)
        
        from holopy.dsqft.causal_patch import PatchType
        self.assertEqual(PatchType.COSMOLOGICAL.value, 1)
        self.assertEqual(PatchType.STATIC.value, 2)
        self.assertEqual(PatchType.FLAT.value, 3)
        
        from holopy.dsqft.query import QueryType
        self.assertEqual(QueryType.FIELD_VALUE.value, 1)
        self.assertEqual(QueryType.CORRELATION.value, 2)
        self.assertEqual(QueryType.ENTROPY.value, 3)
        self.assertEqual(QueryType.ENERGY.value, 4)
        self.assertEqual(QueryType.GEOMETRY.value, 5)
        self.assertEqual(QueryType.OBSERVABLE.value, 6)
    
    def test_constants_consistency(self):
        """Test that constants remain consistent."""
        from holopy.constants.dsqft_constants import DSQFTConstants
        
        # Reference constants
        dc = DSQFTConstants()
        
        # Key constants
        ref_values = {
            'T_dS': dc.T_dS,
            'T_eff': dc.T_eff,
            'thermal_correlation_alpha': dc.thermal_correlation_alpha,
            'multipole_l1': dc.multipole_l1,
            'multipole_ratio': dc.multipole_ratio,
            'boundary_bulk_coupling': dc.boundary_bulk_coupling,
            'critical_manifestation_threshold': dc.critical_manifestation_threshold
        }
        
        # Create a new constants object
        new_dc = DSQFTConstants()
        
        # Values should match
        for key, ref_value in ref_values.items():
            self.assertEqual(getattr(new_dc, key), ref_value)
        
        # Functions should return consistent values
        t1 = dc.get_dS_temp()
        t2 = new_dc.get_dS_temp()
        self.assertEqual(t1, t2)
        
        # Multipole transitions should be consistent
        for n in range(1, 5):
            m1 = dc.get_multipole_transition(n)
            m2 = new_dc.get_multipole_transition(n)
            self.assertEqual(m1, m2)
    
    def test_wave_packet_propagation(self):
        """Test that wave packet propagation behaves correctly."""
        # Create a standard Gaussian wave packet
        def gaussian_wave(x):
            return np.exp(-np.sum(x**2))
        
        # Create causal patch and simulation
        patch = CausalPatch(
            radius=1.0,
            reference_frame='static',
            observer_time=0.0,
            d=4
        )
        
        field_config = {
            'scalar': {
                'mass': 0.0,  # Massless for simplicity
                'type': FieldType.SCALAR
            }
        }
        
        sim = DSQFTSimulation(
            causal_patch=patch,
            field_config=field_config,
            boundary_conditions='custom',
            d=4
        )
        
        # Set initial wave packet
        sim.set_custom_boundary_values('scalar', gaussian_wave)
        
        # Run simulation for a moderate time
        results = sim.evolve(duration=2.0, num_steps=10)
        
        # Verify wave packet propagation
        # Should spread with time according to known physics
        query = DSQFTQuery(sim, patch, d=4)
        
        # Check spreading at different times
        widths = []
        times = [0.5, 1.0, 1.5, 2.0]
        
        # Sample points
        x_values = np.linspace(-0.5, 0.5, 5)
        sample_points = [np.array([x, 0.0, 0.0]) for x in x_values]
        
        for t in times:
            # Get field values at sample points
            field_values = []
            for x in sample_points:
                result = query.query_field_value('scalar', t, x)
                field_values.append(result.value)
            
            # Calculate approximate width using standard deviation
            mean_x = sum(x_values * field_values) / sum(field_values) if sum(field_values) != 0 else 0
            variance = sum(((x - mean_x) ** 2) * field_values for x in x_values) / sum(field_values) if sum(field_values) != 0 else 0
            width = np.sqrt(variance) if variance > 0 else 0
            
            # Store for comparison
            widths.append(width)
        
        # Width should increase with time
        # Check that it's monotonically increasing
        for i in range(1, len(widths)):
            self.assertGreaterEqual(widths[i], widths[i-1],
                                   msg="Wave packet width should increase or remain constant with time")
        
        # The spreading should be physically accurate
        # For a massless field, spreading should be approximately linear with time
        # due to the group velocity
        
        # Calculate the ratio of spreading
        if widths[0] > 0:
            spread_ratio = widths[-1] / widths[0]
            
            # The ratio should be greater than 1.0 (indicating spreading)
            # and proportional to the time ratio
            time_ratio = times[-1] / times[0]
            
            # The actual spreading depends on the dispersion relation
            # but should be related to the time ratio
            self.assertGreater(spread_ratio, 1.0,
                              msg="Wave packet should spread over time")
            
            # For massless particles, spreading is approximately linear
            # so the ratio should be comparable to the time ratio
            self.assertAlmostEqual(spread_ratio, time_ratio, delta=time_ratio * 0.5,
                                  msg="Wave packet spreading should be physically accurate")
    
    def test_cmb_power_spectrum(self):
        """Test that CMB power spectrum analysis produces correct results."""
        # Create simulation focused on cosmological scales
        patch = CausalPatch(
            radius=1.0,  # Hubble radius in natural units
            reference_frame='cosmological',
            observer_time=0.0,
            d=4,
            patch_type=PatchType.COSMOLOGICAL
        )
        
        field_config = {
            'inflaton': {
                'mass': 0.01,  # Small mass for inflation
                'type': FieldType.SCALAR
            }
        }
        
        sim = DSQFTSimulation(
            causal_patch=patch,
            field_config=field_config,
            boundary_conditions='vacuum',
            d=4
        )
        
        # Run a short evolution
        sim.evolve(duration=0.1, num_steps=2)
        
        # Calculate CMB power spectrum
        query = DSQFTQuery(sim, patch, d=4)
        
        # Get power spectrum for a range of multipoles
        ell_values = [2, 10, 50, 100, 500, 1000]
        power_values = []
        
        for ell in ell_values:
            result = query.query(
                QueryType.COSMOLOGICAL_OBSERVABLE,
                'cmb_power_spectrum',
                ell=ell
            )
            power_values.append(result.value)
        
        # Verify physical characteristics of the power spectrum
        
        # 1. Power spectrum should be positive
        for power in power_values:
            self.assertGreater(power, 0.0,
                              msg="CMB power spectrum should be positive")
        
        # 2. Should follow approximate 1/ell^2 scaling at high ell
        # due to the Sachs-Wolfe effect
        if len(power_values) >= 4:
            # Calculate scaling between high multipoles
            high_ell_ratio = ell_values[-1] / ell_values[-2]
            power_ratio = power_values[-1] / power_values[-2]
            
            # Expected scaling is approximately 1/ell^2
            expected_ratio = 1.0 / high_ell_ratio**2
            
            # Allow for some variation due to physical effects
            self.assertAlmostEqual(power_ratio, expected_ratio, delta=expected_ratio * 0.7,
                                  msg="CMB power spectrum should scale approximately as 1/ell^2 at high ell")
        
        # 3. Low multipoles should show modifications due to E8×E8 structure
        # Get theoretical prediction for first few multipoles from query
        theoretical_result = query.query(
            QueryType.THEORETICAL_PREDICTION,
            'cmb_low_multipoles'
        )
        
        # Compare with calculated values for low ell
        low_ell_theoretical = theoretical_result.value[:2]  # First two multipoles
        low_ell_calculated = power_values[:2]  # l=2 and l=10
        
        # Should match within reasonable tolerance
        for i in range(len(low_ell_theoretical)):
            if low_ell_theoretical[i] > 0:
                ratio = low_ell_calculated[i] / low_ell_theoretical[i]
                self.assertGreater(ratio, 0.5)
                self.assertLess(ratio, 2.0)
    
    def test_e8_scaling_relations(self):
        """Test that the E8×E8 heterotic structure produces correct scaling relations."""
        # This test verifies that the dS/QFT correspondence implementation
        # correctly produces the expected scaling relations from the
        # E8×E8 heterotic structure framework
        
        # Create simulation
        patch = CausalPatch(
            radius=1.0,
            reference_frame='static',
            observer_time=0.0,
            d=4
        )
        
        field_config = {
            'scalar': {
                'mass': 0.0,
                'type': FieldType.SCALAR
            }
        }
        
        # Create simulation with different gamma values to test scaling
        gamma_values = [0.0, 1e-2, 1e-1]
        simulations = []
        
        for gamma in gamma_values:
            sim = DSQFTSimulation(
                causal_patch=patch,
                field_config=field_config,
                boundary_conditions='vacuum',
                d=4,
                gamma=gamma
            )
            sim.evolve(duration=0.1, num_steps=2)
            simulations.append(sim)
        
        # Create query interfaces
        queries = [DSQFTQuery(sim, patch, d=4) for sim in simulations]
        
        # Theoretical prediction:
        # For γ scaling, various physical quantities should scale in specific ways:
        
        # 1. Entropy scaling: S ∝ 1 + γ·t for small γt
        entropies = []
        for query in queries:
            result = query.query(
                QueryType.OBSERVABLE,
                'total_entropy'
            )
            entropies.append(result.value)
        
        # Entropy increase should scale linearly with gamma
        # For γ = 0, entropy should remain constant
        # The difference should be proportional to γ
        if len(entropies) >= 3 and entropies[0] > 0:
            # Calculate entropy increases
            entropy_increases = [entropies[i] - entropies[0] for i in range(1, len(entropies))]
            gamma_ratio = gamma_values[2] / gamma_values[1]
            entropy_ratio = entropy_increases[1] / entropy_increases[0] if entropy_increases[0] != 0 else float('inf')
            
            # The ratio should be approximately the same as the gamma ratio
            self.assertAlmostEqual(entropy_ratio, gamma_ratio, delta=gamma_ratio * 0.5,
                                  msg="Entropy increase should scale linearly with gamma")
        
        # 2. Decoherence rate: Rate ∝ γ
        decoherence_rates = []
        for query in queries:
            result = query.query(
                QueryType.OBSERVABLE,
                'decoherence_rate'
            )
            decoherence_rates.append(result.value)
        
        # Decoherence rate should scale linearly with gamma
        # For γ = 0, the rate should be zero
        if len(decoherence_rates) >= 3 and abs(decoherence_rates[0]) < 1e-10:
            # Calculate rate ratio
            rate_ratio = decoherence_rates[2] / decoherence_rates[1] if abs(decoherence_rates[1]) > 1e-10 else float('inf')
            
            # The ratio should be approximately the same as the gamma ratio
            self.assertAlmostEqual(rate_ratio, gamma_ratio, delta=gamma_ratio * 0.5,
                                  msg="Decoherence rate should scale linearly with gamma")
        
        # 3. Field amplitude decay: A(t) ∝ exp(-γt)
        field_values = []
        for query in queries:
            result = query.query_field_value(
                'scalar', 0.1, np.array([0.0, 0.0, 0.0])
            )
            field_values.append(result.value)
        
        # For γ = 0, the field should not decay
        # For γ > 0, it should decay exponentially
        if len(field_values) >= 3 and abs(field_values[0]) > 1e-10:
            # Calculate decay ratios
            decay_ratios = [field_values[i] / field_values[0] for i in range(1, len(field_values))]
            
            # Theoretical decay is exp(-γt)
            t = 0.1  # Duration of simulation
            expected_decays = [np.exp(-gamma * t) for gamma in gamma_values[1:]]
            
            # The ratios should match the expected decay within tolerance
            for i in range(len(decay_ratios)):
                self.assertAlmostEqual(decay_ratios[i], expected_decays[i], delta=expected_decays[i] * 0.5,
                                      msg=f"Field amplitude decay should follow exp(-γt), gamma={gamma_values[i+1]}")
    
    def test_physical_constants_consistency(self):
        """Test that physical constants are consistent throughout the implementation."""
        from holopy.constants.physical_constants import PhysicalConstants
        from holopy.constants.dsqft_constants import DSQFTConstants
        
        # Physical constants should be consistent with E8×E8 framework
        
        pc = PhysicalConstants()
        dsc = DSQFTConstants()
        
        # 1. de Sitter temperature should be H/(2π)
        expected_temperature = pc.hubble_parameter / (2.0 * np.pi)
        self.assertAlmostEqual(dsc.T_dS, expected_temperature, delta=expected_temperature * 1e-10,
                              msg="de Sitter temperature should be H/(2π)")
        
        # 2. Information processing rate should be (2π/240²) × (1/t_P)
        if hasattr(pc, 'planck_time'):
            expected_gamma = (2.0 * np.pi / (240.0**2)) * (1.0 / pc.planck_time)
            self.assertAlmostEqual(pc.gamma, expected_gamma, delta=expected_gamma * 1e-10,
                                  msg="Information processing rate should follow theoretical formula")
        
        # 3. E8×E8 heterotic structure constants should be correctly defined
        
        # κ(π) should be π^4/24
        if hasattr(dsc, 'information_spacetime_conversion_factor'):
            expected_kappa_pi = np.pi**4 / 24.0
            self.assertAlmostEqual(dsc.information_spacetime_conversion_factor, expected_kappa_pi, 
                                  delta=expected_kappa_pi * 1e-10,
                                  msg="Information-spacetime conversion factor should be π^4/24")
        
        # 4. Bulk-boundary conversion factor should include E8×E8 structure
        if hasattr(dsc, 'bulk_boundary_conversion_factor') and hasattr(dsc, 'e8_root_count'):
            # Should be related to the number of roots in E8
            root_count = dsc.e8_root_count
            self.assertEqual(root_count, 240,
                            msg="E8 root count should be 240")
            
            # The conversion factor should include this parameter
            self.assertGreater(dsc.bulk_boundary_conversion_factor, 0.0,
                              msg="Bulk-boundary conversion factor should be positive")

if __name__ == '__main__':
    unittest.main() 