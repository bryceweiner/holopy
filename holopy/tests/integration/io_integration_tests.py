"""
Integration tests for the IO module with other parts of HoloPy.
"""

import unittest
import os
import tempfile
import numpy as np

from holopy.io.data_formats import (
    HoloData, E8Data, InformationTensorData, QuantumData, CosmologyData
)
from holopy.io.exporters import export_data
from holopy.io.importers import import_data
from holopy.e8.root_system import RootSystem
from holopy.info.current import InfoCurrentTensor
from holopy.constants.physical_constants import PHYSICAL_CONSTANTS

class TestIOIntegration(unittest.TestCase):
    """Integration tests for the IO module with other HoloPy components."""
    
    def setUp(self):
        """Set up test data."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_e8_data_roundtrip(self):
        """Test roundtrip serialization of E8 root system data."""
        # Create a root system
        root_system = RootSystem()
        
        # Get the roots
        roots = root_system.get_roots()
        
        # Create E8Data object
        e8_data = E8Data(
            data=roots,
            dimension=8,
            root_count=240
        )
        
        # Export to JSON
        filepath = os.path.join(self.temp_dir.name, "e8_roots.json")
        export_data(e8_data, filepath)
        
        # Import from JSON
        imported_data = import_data(filepath)
        
        # Check that the imported data is an E8Data object
        self.assertIsInstance(imported_data, E8Data)
        
        # Check that the roots match
        self.assertTrue(np.array_equal(imported_data.data, roots))
        
        # Create a new root system from the imported data
        new_root_system = RootSystem()
        
        # Verify that both root systems have the same properties
        self.assertEqual(len(new_root_system.get_roots()), len(root_system.get_roots()))
        self.assertEqual(new_root_system.rank, root_system.rank)
        self.assertEqual(new_root_system.dimension, root_system.dimension)
    
    def test_info_tensor_roundtrip(self):
        """Test roundtrip serialization of information tensor data."""
        # Create a simple density function
        def density_function(x):
            return np.exp(-np.sum(x**2))
        
        # Create an information tensor from the density function
        info_tensor = InfoCurrentTensor.from_density(
            density_function=density_function,
            grid_size=5,
            dimension=4
        )
        
        # Get the tensor components
        tensor_data = info_tensor.get_tensor()
        density_data = info_tensor.get_density()
        
        # Create a data structure for export
        tensor_export = {
            "tensor": tensor_data,
            "density": density_data
        }
        
        # Create InformationTensorData object
        info_data = InformationTensorData(
            data=tensor_export,
            dimension=4,
            coordinates="cartesian",
            has_density=True
        )
        
        # Export to JSON
        filepath = os.path.join(self.temp_dir.name, "info_tensor.json")
        export_data(info_data, filepath)
        
        # Import from JSON
        imported_data = import_data(filepath)
        
        # Check that the imported data is an InformationTensorData object
        self.assertIsInstance(imported_data, InformationTensorData)
        
        # Check that the tensor components match
        self.assertTrue(np.array_equal(imported_data.data["tensor"], tensor_data))
        self.assertTrue(np.array_equal(imported_data.data["density"], density_data))
        
        # Create a new information tensor using the imported data
        new_info_tensor = InfoCurrentTensor(
            imported_data.data["tensor"], 
            imported_data.data["density"]
        )
        
        # Verify that both tensors have the same properties
        self.assertTrue(np.array_equal(new_info_tensor.get_tensor(), info_tensor.get_tensor()))
        self.assertTrue(np.array_equal(new_info_tensor.get_density(), info_tensor.get_density()))
        
        # Verify that the conservation law holds for both tensors
        original_div = info_tensor.compute_divergence()
        new_div = new_info_tensor.compute_divergence()
        
        # The divergence should approximately equal gamma * density
        gamma = PHYSICAL_CONSTANTS.get_gamma()
        self.assertTrue(np.allclose(original_div, gamma * density_data, rtol=1e-5))
        self.assertTrue(np.allclose(new_div, gamma * density_data, rtol=1e-5))
    
    def test_quantum_cosmology_data_integration(self):
        """Test integration of quantum and cosmology data."""
        # Create sample quantum data
        quantum_state = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
        # Ensure it's Hermitian (like a density matrix)
        quantum_state = quantum_state + quantum_state.conjugate().T
        
        # Create sample cosmological parameters
        cosmology_params = {
            "H0": 70.0,
            "Omega_m": 0.3,
            "Omega_lambda": 0.7
        }
        
        # Create a data structure that combines quantum and cosmological data
        combined_data = {
            "quantum_state": quantum_state,
            "cosmology_params": cosmology_params,
            "simulation_results": {
                "redshift": np.linspace(0, 10, 100),
                "decoherence_rate": np.exp(-np.linspace(0, 10, 100))
            }
        }
        
        # Create a HoloData object
        holo_data = HoloData(
            data=combined_data
        )
        
        # Export to JSON
        filepath = os.path.join(self.temp_dir.name, "combined_data.json")
        export_data(holo_data, filepath)
        
        # Import from JSON
        imported_data = import_data(filepath)
        
        # Check that the imported data is a HoloData object
        self.assertIsInstance(imported_data, HoloData)
        
        # Check that the data structures match
        imported_combined = imported_data.data
        
        # Check cosmology parameters
        for key, value in cosmology_params.items():
            self.assertEqual(imported_combined["cosmology_params"][key], value)
        
        # Check simulation results
        self.assertTrue(np.array_equal(
            imported_combined["simulation_results"]["redshift"],
            combined_data["simulation_results"]["redshift"]
        ))
        self.assertTrue(np.array_equal(
            imported_combined["simulation_results"]["decoherence_rate"],
            combined_data["simulation_results"]["decoherence_rate"]
        ))
        
        # The quantum state might have imaginary components, so check real and imag parts
        # Note: JSON serialization will split complex numbers into real and imag parts
        org_real = quantum_state.real
        org_imag = quantum_state.imag
        imp_real = np.array(imported_combined["quantum_state"]["real"])
        imp_imag = np.array(imported_combined["quantum_state"]["imag"])
        
        self.assertTrue(np.allclose(org_real, imp_real))
        self.assertTrue(np.allclose(org_imag, imp_imag))

if __name__ == '__main__':
    unittest.main() 