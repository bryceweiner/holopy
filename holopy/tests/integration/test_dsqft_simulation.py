"""
Integration tests for holopy.dsqft.simulation module.

These tests verify that the different components of the dS/QFT correspondence framework
work together correctly in simulations.
"""

import unittest
import numpy as np
import tempfile
from pathlib import Path

from holopy.dsqft.simulation import DSQFTSimulation
from holopy.dsqft.causal_patch import CausalPatch, PatchType
from holopy.dsqft.dictionary import FieldType
from holopy.dsqft.query import DSQFTQuery, QueryType

class TestDSQFTSimulation(unittest.TestCase):
    """Integration tests for the DSQFTSimulation class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a causal patch
        self.causal_patch = CausalPatch(
            radius=1.0,
            reference_frame='static',
            observer_time=0.0,
            d=4
        )
        
        # Create field configuration
        self.field_config = {
            'scalar': {
                'mass': 0.0,
                'spin': 0,
                'type': FieldType.SCALAR
            },
            'vector': {
                'mass': 0.0,
                'spin': 1,
                'type': FieldType.VECTOR
            }
        }
        
        # Create simulation
        self.simulation = DSQFTSimulation(
            causal_patch=self.causal_patch,
            field_config=self.field_config,
            boundary_conditions='vacuum',
            d=4
        )
        
        # Create query interface
        self.query = DSQFTQuery(
            simulation=self.simulation,
            causal_patch=self.causal_patch,
            d=4
        )
    
    def test_simulation_initialization(self):
        """Test that simulation initializes correctly."""
        # Check that fields were registered
        self.assertIn('scalar', self.simulation.field_config)
        self.assertIn('vector', self.simulation.field_config)
        
        # Check that components were initialized
        self.assertIsNotNone(self.simulation.dictionary)
        self.assertIsNotNone(self.simulation.correlation)
        self.assertIsNotNone(self.simulation.transport)
        self.assertIsNotNone(self.simulation.coupling)
        
        # Check that spatial grid was created
        self.assertIsNotNone(self.simulation.spatial_grid)
        self.assertGreater(len(self.simulation.spatial_grid), 0)
    
    def test_propagator_dictionary_integration(self):
        """Test integration between propagator and dictionary."""
        # Get the propagator for a field
        propagator = self.simulation.dictionary.get_propagator('scalar')
        
        # Create a simple boundary function
        def boundary_func(x):
            return 1.0
        
        # Get boundary grid
        boundary_grid = self.simulation.causal_patch.boundary_projection()
        
        # Compute a field value using the propagator
        eta = -1.0
        x_bulk = np.array([0.0, 0.0, 0.0])
        
        field_value = propagator.compute_field_from_boundary(
            boundary_func, eta, x_bulk, boundary_grid
        )
        
        # Should get a non-zero value
        self.assertIsInstance(field_value, float)
        self.assertNotEqual(field_value, 0.0)
        
        # The same computation should work through the dictionary
        field_value2 = self.simulation.dictionary.compute_bulk_field_value(
            'scalar', boundary_func, boundary_grid, eta, x_bulk
        )
        
        # Should get the same value
        self.assertAlmostEqual(field_value, field_value2)
    
    def test_compute_bulk_field(self):
        """Test computation of bulk field from boundary values."""
        # Set custom boundary values
        def gaussian_boundary(x):
            return np.exp(-np.sum(x**2))
        
        self.simulation.set_custom_boundary_values('scalar', gaussian_boundary)
        
        # Compute field at a point
        t = 0.0
        x = np.array([0.0, 0.0, 0.0])
        
        field_value = self.simulation.compute_bulk_field('scalar', t, x)
        
        # Should get a non-zero value
        self.assertIsInstance(field_value, float)
        self.assertNotEqual(field_value, 0.0)
        
        # Compute at multiple points
        points = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.1, 0.0, 0.0]),
            np.array([0.0, 0.1, 0.0]),
            np.array([0.0, 0.0, 0.1])
        ]
        
        field_values = self.simulation.compute_bulk_field('scalar', t, points)
        
        # Should get an array of values
        self.assertEqual(len(field_values), len(points))
        
        # Values should decrease with distance from origin for Gaussian boundary
        self.assertGreaterEqual(field_values[0], field_values[1])
        self.assertGreaterEqual(field_values[0], field_values[2])
        self.assertGreaterEqual(field_values[0], field_values[3])
    
    def test_compute_correlation_function(self):
        """Test computation of correlation function."""
        # Compute correlation function
        t1 = 0.0
        x1 = np.array([0.0, 0.0, 0.0])
        t2 = 0.0
        x2 = np.array([0.1, 0.0, 0.0])
        
        correlation = self.simulation.compute_correlation_function(
            'scalar', t1, x1, t2, x2
        )
        
        # Should get a non-zero value
        self.assertIsInstance(correlation, float)
        self.assertNotEqual(correlation, 0.0)
        
        # Correlation should decrease with distance
        x3 = np.array([0.2, 0.0, 0.0])
        
        correlation2 = self.simulation.compute_correlation_function(
            'scalar', t1, x1, t2, x3
        )
        
        self.assertLess(correlation2, correlation)
    
    def test_short_evolution(self):
        """Test a short simulation evolution."""
        # Run a short evolution
        duration = 0.1
        num_steps = 2
        
        results = self.simulation.evolve(duration, num_steps)
        
        # Check that results were produced
        self.assertIn('time_points', results)
        self.assertIn('field_evolution', results)
        self.assertIn('energy_density_evolution', results)
        self.assertIn('entropy_evolution', results)
        
        # Time points should match expectations
        self.assertEqual(len(results['time_points']), num_steps + 1)
        self.assertAlmostEqual(results['time_points'][-1], duration)
        
        # Field evolution should have entries for each field
        self.assertIn('scalar', results['field_evolution'])
        self.assertIn('vector', results['field_evolution'])
        
        # Each field should have 'num_steps' entries
        self.assertEqual(len(results['field_evolution']['scalar']), num_steps)
        
        # Energy density and entropy should have 'num_steps' entries
        self.assertEqual(len(results['energy_density_evolution']), num_steps)
        self.assertEqual(len(results['entropy_evolution']), num_steps)
        
        # Entropy should be non-negative and non-decreasing
        self.assertGreaterEqual(results['entropy_evolution'][0], 0.0)
        if len(results['entropy_evolution']) > 1:
            self.assertGreaterEqual(
                results['entropy_evolution'][1],
                results['entropy_evolution'][0]
            )
    
    def test_simulation_query_integration(self):
        """Test integration between simulation and query interface."""
        # Run a short evolution to generate data
        self.simulation.evolve(0.1, 2)
        
        # Query field values
        field_result = self.query.query_field_value(
            'scalar', 0.1, np.array([0.0, 0.0, 0.0])
        )
        
        # Should get a result
        self.assertIsNotNone(field_result)
        self.assertIsNotNone(field_result.value)
        
        # Query energy density
        energy_result = self.query.query_energy_density(
            0.1, np.array([0.0, 0.0, 0.0])
        )
        
        # Should get a result
        self.assertIsNotNone(energy_result)
        self.assertIsNotNone(energy_result.value)
        
        # Query entropy
        entropy_result = self.query.query_entropy(
            0.1, np.array([0.0, 0.0, 0.0]), 0.5
        )
        
        # Should get a result
        self.assertIsNotNone(entropy_result)
        self.assertIsNotNone(entropy_result.value)
        
        # Query using unified interface
        unified_result = self.query.query(
            QueryType.OBSERVABLE,
            'temperature',
            t=0.1,
            x=np.array([0.0, 0.0, 0.0])
        )
        
        # Should get a result
        self.assertIsNotNone(unified_result)
        self.assertIsNotNone(unified_result.value)
    
    def test_save_load_results(self):
        """Test saving and loading simulation results."""
        # Run a short evolution
        self.simulation.evolve(0.1, 2)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            # Save results
            self.simulation.save_results(tmp_path)
            
            # Create a new simulation
            new_simulation = DSQFTSimulation(
                causal_patch=self.causal_patch,
                field_config=self.field_config
            )
            
            # Load results
            loaded_results = new_simulation.load_results(tmp_path)
            
            # Check that results were loaded
            self.assertIn('time_points', loaded_results)
            self.assertIn('field_evolution', loaded_results)
            self.assertIn('energy_density_evolution', loaded_results)
            self.assertIn('entropy_evolution', loaded_results)
            
            # Time points should match
            np.testing.assert_array_equal(
                loaded_results['time_points'],
                self.simulation.results['time_points']
            )
            
            # Current time should be updated
            self.assertEqual(new_simulation.current_time, self.simulation.current_time)
        
        finally:
            # Clean up
            tmp_path.unlink()
    
    def test_boundary_to_bulk_mapping(self):
        """Test mapping from boundary to bulk."""
        # Set custom boundary values with a simple pattern
        def boundary_func(x):
            return 1.0 - 0.1 * np.sum(x**2)
        
        self.simulation.set_custom_boundary_values('scalar', boundary_func)
        
        # Compute field at the origin
        t = 0.0
        origin = np.array([0.0, 0.0, 0.0])
        
        origin_value = self.simulation.compute_bulk_field('scalar', t, origin)
        
        # Compute field at several distances from origin
        distances = np.linspace(0.1, 0.5, 5)
        points = [np.array([d, 0.0, 0.0]) for d in distances]
        
        distance_values = self.simulation.compute_bulk_field('scalar', t, points)
        
        # Field should vary with position in a way that reflects the boundary pattern
        # For this simple case, values should decrease with distance from origin
        for i in range(1, len(distance_values)):
            self.assertLessEqual(distance_values[i], distance_values[i-1])
        
        # Bulk-to-boundary conversion should work
        for point in points:
            boundary_point = self.query.bulk_to_boundary(t, point)
            self.assertEqual(len(boundary_point), 2)  # (t, x) tuple
            
            # Convert back to bulk
            bulk_point = self.query.boundary_to_bulk(boundary_point[0], boundary_point[1])
            self.assertEqual(len(bulk_point), 2)  # (t, x) tuple
            
            # Round-trip conversion should approximately recover the original point
            np.testing.assert_allclose(bulk_point[1], point, rtol=1e-5)
    
    def test_field_operator_relationship(self):
        """Test relationship between bulk fields and boundary operators."""
        # Set custom boundary values
        def boundary_func(x):
            return np.exp(-np.sum(x**2))
        
        self.simulation.set_custom_boundary_values('scalar', boundary_func)
        
        # Evolve the simulation
        self.simulation.evolve(0.1, 2)
        
        # Compute observable
        observable = self.simulation.calculate_observable('field_expectation', field_name='scalar')
        
        # Should get a non-zero value
        self.assertIsInstance(observable, float)
        
        # Query the same observable
        query_result = self.query.query_observable('field_expectation', field_name='scalar')
        
        # Should get the same value
        self.assertAlmostEqual(observable, query_result.value)
    
    def test_holographic_consistency(self):
        """Test that bulk and boundary calculations are consistent with holography."""
        # Set up a boundary state with a known energy scale
        energy_scale = 1.0
        
        # Create a boundary function with a characteristic energy scale
        def boundary_func(x):
            return np.exp(-energy_scale * np.sum(x**2))
        
        # Set the boundary values
        self.simulation.set_custom_boundary_values('scalar', boundary_func)
        
        # Run a short evolution
        self.simulation.evolve(0.1, 2)
        
        # Query field values in the bulk
        bulk_field_value = self.query.query_field_value(
            'scalar', 0.05, np.array([0.0, 0.0, 0.0])
        )
        
        # Query the boundary operator value
        boundary_value = self.query.query_boundary_operator(
            'scalar', np.array([0.0, 0.0, 0.0])
        )
        
        # The bulk field and boundary operator should be related
        # by the bulk-boundary propagator
        self.assertIsNotNone(bulk_field_value)
        self.assertIsNotNone(boundary_value)
        
        # Get the propagator directly to compare
        propagator = self.simulation.dictionary.get_propagator('scalar')
        
        # Calculate field from boundary directly using propagator
        t = 0.05
        eta = -1.0 / np.exp(t)  # Convert to conformal time
        boundary_grid = self.simulation.causal_patch.boundary_projection()
        
        # Compute field directly
        direct_field = propagator.compute_field_from_boundary(
            boundary_func, eta, np.array([0.0, 0.0, 0.0]), boundary_grid
        )
        
        # The field values should be consistent 
        # They won't be identical due to integration schemes, but should be related
        # through a holographic correspondence scale factor
        field_ratio = bulk_field_value.value / direct_field
        
        # Holographic consistency requires this ratio to be close to 1 
        # but with corrections from information processing constraints
        self.assertGreater(field_ratio, 0.1)
        self.assertLess(field_ratio, 10.0)
        
        # Test that physical energy scales match between bulk and boundary
        bulk_energy = self.query.query(
            QueryType.OBSERVABLE, 
            'energy',
            t=0.05,
            x=np.array([0.0, 0.0, 0.0])
        )
        
        boundary_energy = self.query.query(
            QueryType.BOUNDARY_OBSERVABLE,
            'energy',
            x=np.array([0.0, 0.0, 0.0])
        )
        
        # Energy scale on boundary should be related to bulk energy
        # through the holographic principle
        # Given our boundary function with energy scale = 1.0, these should be related
        self.assertIsNotNone(bulk_energy)
        self.assertIsNotNone(boundary_energy)
        
        # Energy values should be non-zero and positive
        self.assertGreater(bulk_energy.value, 0.0)
        self.assertGreater(boundary_energy.value, 0.0)
        
        # The ratio gives us the effective holographic energy mapping
        energy_ratio = bulk_energy.value / boundary_energy.value
        
        # This should be related to the Hubble parameter in dS space
        # which sets the natural energy scale of the correspondence
        expected_ratio = self.simulation.hubble_parameter
        
        # Allow for factors from information processing constraints
        # The ratio should be within an order of magnitude
        self.assertGreater(energy_ratio / expected_ratio, 0.1)
        self.assertLess(energy_ratio / expected_ratio, 10.0)
    
    def test_e8_heterotic_correction_propagation(self):
        """Test that E8×E8 heterotic structure corrections propagate through modules."""
        # Create two simulations: one with standard gamma and one with zero gamma
        # to isolate the information processing effects
        
        # Keep track of the original simulation
        original_simulation = self.simulation
        
        try:
            # Create a causal patch
            causal_patch_zero = CausalPatch(
                radius=1.0,
                reference_frame='static',
                observer_time=0.0,
                d=4
            )
            
            # Create simulation with zero gamma
            simulation_zero = DSQFTSimulation(
                causal_patch=causal_patch_zero,
                field_config=self.field_config,
                boundary_conditions='vacuum',
                d=4,
                gamma=0.0  # No information processing constraints
            )
            
            # Create query interface
            query_zero = DSQFTQuery(
                simulation=simulation_zero,
                causal_patch=causal_patch_zero,
                d=4
            )
            
            # Set a test boundary function for both simulations
            def test_func(x):
                return np.exp(-np.sum(x**2))
            
            original_simulation.set_custom_boundary_values('scalar', test_func)
            simulation_zero.set_custom_boundary_values('scalar', test_func)
            
            # Run short evolutions
            original_simulation.evolve(0.1, 2)
            simulation_zero.evolve(0.1, 2)
            
            # Compare results at multiple points and times
            test_times = [0.05, 0.1]
            test_points = [
                np.array([0.0, 0.0, 0.0]),  # Origin
                np.array([0.5, 0.0, 0.0]),  # Small distance
            ]
            
            # Store ratios between simulations to detect E8×E8 corrections
            field_ratios = []
            entropy_ratios = []
            
            for t in test_times:
                for x in test_points:
                    # Query field values from both simulations
                    field_standard = self.query.query_field_value('scalar', t, x)
                    field_zero = query_zero.query_field_value('scalar', t, x)
                    
                    # Both should return results
                    self.assertIsNotNone(field_standard)
                    self.assertIsNotNone(field_zero)
                    
                    # The ratio should reflect information processing constraints
                    # and E8×E8 heterotic structure corrections
                    if abs(field_zero.value) > 1e-10:
                        ratio = field_standard.value / field_zero.value
                        field_ratios.append(ratio)
                    
                    # Compare entropy calculations
                    radius = 0.5
                    entropy_standard = self.query.query_entropy(t, x, radius)
                    entropy_zero = query_zero.query_entropy(t, x, radius)
                    
                    # Both should return results
                    self.assertIsNotNone(entropy_standard)
                    self.assertIsNotNone(entropy_zero)
                    
                    # The ratio should reflect differences due to information constraints
                    if entropy_zero.value > 1e-10:
                        entropy_ratio = entropy_standard.value / entropy_zero.value
                        entropy_ratios.append(entropy_ratio)
            
            # There should be multiple valid ratios to compare
            self.assertGreater(len(field_ratios), 1)
            
            # Different spacetime points should have different corrections
            # So the ratios should vary
            field_ratio_variance = np.var(field_ratios)
            self.assertGreater(field_ratio_variance, 0.0, 
                             msg="E8×E8 corrections should vary across spacetime")
            
            # If we have entropy ratios, they should also vary
            if len(entropy_ratios) > 1:
                entropy_ratio_variance = np.var(entropy_ratios)
                self.assertGreater(entropy_ratio_variance, 0.0,
                                 msg="Entropy corrections should vary across spacetime")
            
            # The field values with gamma>0 should be suppressed compared to gamma=0
            # This is a key prediction of the information processing theory
            # The mean ratio should be less than 1.0 due to exponential suppression
            mean_field_ratio = np.mean(field_ratios)
            self.assertLess(mean_field_ratio, 1.0,
                          msg="Information processing should suppress field values")
            
        finally:
            # Restore the original simulation reference
            self.simulation = original_simulation
    
    def test_physically_accurate_observables(self):
        """Test that calculated observables match theoretical predictions."""
        # Run a simulation with physically-motivated parameters
        # to verify that observables match theoretical expectations
        
        # Create a hydrogen-atom sized causal patch
        patch_radius = 5.3e-11  # Bohr radius in meters
        
        hydrogen_patch = CausalPatch(
            radius=patch_radius,
            reference_frame='static',
            observer_time=0.0,
            d=4,
            patch_type=PatchType.STATIC
        )
        
        # Create simulation with physical parameters for hydrogen
        field_config = {
            'electron': {
                'mass': 9.1e-31,  # kg
                'charge': -1.6e-19,  # C
                'type': FieldType.SCALAR
            },
            'proton': {
                'mass': 1.67e-27,  # kg
                'charge': 1.6e-19,  # C
                'type': FieldType.SCALAR
            }
        }
        
        # Create a new simulation
        hydrogen_sim = DSQFTSimulation(
            causal_patch=hydrogen_patch,
            field_config=field_config,
            boundary_conditions='ground_state',
            d=4
        )
        
        # Create query interface
        hydrogen_query = DSQFTQuery(
            simulation=hydrogen_sim,
            causal_patch=hydrogen_patch,
            d=4
        )
        
        # Run simulation
        hydrogen_sim.evolve(1e-16, 2)  # Short time for testing
        
        # Query energy levels - should approximate hydrogen energy levels
        energy_result = hydrogen_query.query(
            QueryType.OBSERVABLE,
            'binding_energy'
        )
        
        # Should get a result
        self.assertIsNotNone(energy_result)
        self.assertIsNotNone(energy_result.value)
        
        # Ground state binding energy of hydrogen is approximately -13.6 eV
        expected_binding_energy = -13.6 * 1.602e-19  # Convert to joules
        
        # Given the simplified model, result should be within an order of magnitude
        energy_ratio = energy_result.value / expected_binding_energy
        self.assertGreater(energy_ratio, 0.1)
        self.assertLess(energy_ratio, 10.0)
        
        # Check that decoherence effects are correctly calculated
        # Information processing constraints predict decoherence rates proportional to L^-2
        # Query decoherence rate
        decoherence_result = hydrogen_query.query(
            QueryType.OBSERVABLE,
            'decoherence_rate'
        )
        
        # Should get a result
        self.assertIsNotNone(decoherence_result)
        self.assertIsNotNone(decoherence_result.value)
        
        # Create another patch with different size for comparison
        larger_patch = CausalPatch(
            radius=patch_radius * 2.0,  # Double the size
            reference_frame='static',
            observer_time=0.0,
            d=4,
            patch_type=PatchType.STATIC
        )
        
        # Create new simulation
        larger_sim = DSQFTSimulation(
            causal_patch=larger_patch,
            field_config=field_config,
            boundary_conditions='ground_state',
            d=4
        )
        
        # Create query interface
        larger_query = DSQFTQuery(
            simulation=larger_sim,
            causal_patch=larger_patch,
            d=4
        )
        
        # Run simulation
        larger_sim.evolve(1e-16, 2)
        
        # Query decoherence rate
        larger_decoherence = larger_query.query(
            QueryType.OBSERVABLE,
            'decoherence_rate'
        )
        
        # Should get a result
        self.assertIsNotNone(larger_decoherence)
        self.assertIsNotNone(larger_decoherence.value)
        
        # Decoherence should scale as L^-2
        # So doubling size should reduce rate by factor of 4
        expected_ratio = 0.25  # 1/4
        actual_ratio = larger_decoherence.value / decoherence_result.value
        
        # Allow for some numerical variation but should be close
        self.assertAlmostEqual(actual_ratio, expected_ratio, delta=0.1,
                              msg="Decoherence rate should scale as L^-2")
    
    def test_conservation_laws(self):
        """Test that physical conservation laws are preserved in the simulation."""
        # Energy, momentum, and information should be conserved appropriately
        # Run a longer simulation
        results = self.simulation.evolve(0.5, 5)
        
        # Extract energy evolution
        energy_values = []
        for t_idx, t in enumerate(results['time_points']):
            # Calculate total energy at each time
            energy_query = self.query.query(
                QueryType.OBSERVABLE,
                'total_energy',
                t=t
            )
            energy_values.append(energy_query.value)
        
        # Due to information processing and holographic constraints,
        # energy is not strictly conserved, but changes should be controlled
        # by the information processing rate
        
        # Get gamma value
        gamma = self.simulation.gamma
        
        # Calculate maximum expected fractional change in energy
        t_span = results['time_points'][-1] - results['time_points'][0]
        max_expected_change = gamma * t_span
        
        # Actual fractional energy change
        if abs(energy_values[0]) > 1e-10:
            actual_change = abs(energy_values[-1] - energy_values[0]) / abs(energy_values[0])
            
            # Should be less than or equal to max expected change
            # with some numerical tolerance
            self.assertLessEqual(actual_change, max_expected_change * 1.1,
                                msg="Energy changes should be bounded by gamma")
        
        # Test information conservation
        entropy_values = results['entropy_evolution']
        
        # Calculate entropy production rate
        if len(entropy_values) > 1:
            t_diff = t_span / (len(entropy_values) - 1)
            entropy_rates = np.diff(entropy_values) / t_diff
            
            # Maximum entropy production rate should be bounded by gamma
            # multiplied by the system size
            system_size = 4*np.pi*self.simulation.causal_patch.radius**2  # Area of horizon
            max_entropy_rate = gamma * system_size
            
            # Check that all rates are below the maximum
            for rate in entropy_rates:
                self.assertLessEqual(rate, max_entropy_rate * 1.1,
                                   msg="Entropy production should be bounded by gamma·Area")
    
    def test_hydrogen_binding_energy(self):
        """Test that hydrogen binding energy calculation works correctly."""
        # Create a hydrogen-atom sized causal patch
        patch_radius = 5.3e-11  # Bohr radius in meters
        
        hydrogen_patch = CausalPatch(
            radius=patch_radius,
            reference_frame='static',
            observer_time=0.0,
            d=4
        )
        
        # Create simulation with physical parameters for hydrogen
        field_config = {
            'electron': {
                'mass': 9.1093837015e-31,  # kg
                'charge': -1.602176634e-19,  # C
                'type': FieldType.SCALAR
            },
            'proton': {
                'mass': 1.67262192369e-27,  # kg
                'charge': 1.602176634e-19,  # C
                'type': FieldType.SCALAR
            }
        }
        
        # Create simulation with hydrogen boundary conditions
        simulation = DSQFTSimulation(
            causal_patch=hydrogen_patch,
            field_config=field_config,
            boundary_conditions='hydrogen',
            d=4
        )
        
        # Create query interface
        query = DSQFTQuery(simulation=simulation)
        
        # Query binding energy
        result = query.query_observable('binding_energy', fields=['electron', 'proton'])
        
        # Expected binding energy is -13.6 eV
        expected_binding_energy = -13.6
        
        # Test that binding energy is close to expected value
        self.assertIsNotNone(result.value)
        self.assertAlmostEqual(result.value, expected_binding_energy, delta=0.1)

if __name__ == '__main__':
    unittest.main() 