"""
Regression tests for the IO module.

These tests ensure that serialization and deserialization remains consistent
across versions of the library.
"""

import unittest
import os
import json
import numpy as np
import tempfile
from pathlib import Path

from holopy.io.data_formats import (
    HoloData, E8Data, InformationTensorData, QuantumData, CosmologyData,
    data_to_json, json_to_data
)
from holopy.io.exporters import export_json
from holopy.io.importers import import_json

class TestIORegression(unittest.TestCase):
    """Regression tests for the IO module serialization formats."""
    
    def setUp(self):
        """Set up test data."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test data
        self.test_array = np.eye(3)
        self.test_data = HoloData(self.test_array)
        
        # Reference values directory (would be populated in a real project)
        self.reference_dir = os.path.join(os.path.dirname(__file__), 'reference_data')
        
        # Ensure the reference directory exists
        Path(self.reference_dir).mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def _save_reference_data(self, data_dict, filename):
        """
        Save reference data for future regression tests.
        
        Args:
            data_dict: Dictionary to save
            filename: Filename to save to
        """
        filepath = os.path.join(self.reference_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data_dict, f)
    
    def _load_reference_data(self, filename):
        """
        Load reference data for regression tests.
        
        Args:
            filename: Filename to load from
            
        Returns:
            dict: Loaded data
        """
        filepath = os.path.join(self.reference_dir, filename)
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def test_holodata_json_format(self):
        """Test HoloData JSON format remains consistent."""
        # Create a JSON serialization
        json_str = data_to_json(self.test_data)
        json_dict = json.loads(json_str)
        
        # Check against reference data if it exists
        ref_data = self._load_reference_data('holodata_reference.json')
        
        if ref_data is None:
            # If reference doesn't exist, save this as reference
            self._save_reference_data(json_dict, 'holodata_reference.json')
            self.skipTest("Reference data not found, creating reference")
        else:
            # Compare structure of current output with reference
            self.assertEqual(set(json_dict.keys()), set(ref_data.keys()))
            self.assertEqual(set(json_dict['metadata'].keys()), set(ref_data['metadata'].keys()))
            
            # Don't compare actual data values, just structure, as data might change
            self.assertEqual(type(json_dict['data']), type(ref_data['data']))
    
    def test_specialized_data_json_format(self):
        """Test specialized data formats remain consistent."""
        # Create all specialized data types
        e8_data = E8Data(
            np.random.rand(240, 8), 
            dimension=8, 
            root_count=240
        )
        
        info_data = InformationTensorData(
            np.random.rand(4, 4),
            dimension=4,
            coordinates="cartesian",
            has_density=True
        )
        
        quantum_data = QuantumData(
            np.random.rand(10, 10),
            is_density_matrix=True,
            time_dependent=True,
            decoherence_included=False
        )
        
        cosmology_data = CosmologyData(
            np.random.rand(100),
            redshift_range=(0.0, 2.0),
            parameters={"H0": 70.0},
            observables=["CMB"]
        )
        
        # Test each specialized data type
        for name, data in [
            ("e8", e8_data),
            ("info", info_data),
            ("quantum", quantum_data),
            ("cosmology", cosmology_data),
        ]:
            # Serialize to JSON
            json_str = data_to_json(data)
            json_dict = json.loads(json_str)
            
            # Reference filename
            ref_filename = f'{name}_data_reference.json'
            
            # Check against reference data if it exists
            ref_data = self._load_reference_data(ref_filename)
            
            if ref_data is None:
                # If reference doesn't exist, save this as reference
                self._save_reference_data(json_dict, ref_filename)
                self.skipTest(f"Reference data for {name} not found, creating reference")
            else:
                # Compare structure of current output with reference
                self.assertEqual(set(json_dict.keys()), set(ref_data.keys()))
                
                # Check specific fields that should always be present
                self.assertIn('type', json_dict)
                self.assertEqual(json_dict['type'], ref_data['type'])
                
                # Check type-specific fields
                if name == 'e8':
                    self.assertEqual(json_dict['dimension'], ref_data['dimension'])
                    self.assertEqual(json_dict['root_count'], ref_data['root_count'])
                elif name == 'info':
                    self.assertEqual(json_dict['dimension'], ref_data['dimension'])
                    self.assertEqual(json_dict['coordinates'], ref_data['coordinates'])
                elif name == 'quantum':
                    self.assertEqual(json_dict['is_density_matrix'], ref_data['is_density_matrix'])
                    self.assertEqual(json_dict['time_dependent'], ref_data['time_dependent'])
                elif name == 'cosmology':
                    self.assertEqual(json_dict['parameters'].keys(), ref_data['parameters'].keys())
                    self.assertEqual(sorted(json_dict['observables']), sorted(ref_data['observables']))
    
    def test_serialization_version_compatibility(self):
        """
        Test backward compatibility with previous serialization formats.
        
        This test ensures that we can read data saved with previous versions.
        """
        # This test would use specific reference files that represent
        # serialization formats from different versions of the library
        
        # For example, we might have a reference file called 'v0.1.0_holodata.json'
        version = 'v0.1.0'
        ref_filename = f'{version}_holodata.json'
        
        # Check if the reference file exists
        ref_data = self._load_reference_data(ref_filename)
        
        if ref_data is None:
            # If reference doesn't exist, create a placeholder for future tests
            # In a real project, we'd manually save reference files for each version
            current_json = data_to_json(self.test_data)
            self._save_reference_data(json.loads(current_json), ref_filename)
            self.skipTest(f"Reference data for {version} not found, creating placeholder")
        else:
            # Create JSON string from the reference data
            ref_json_str = json.dumps(ref_data)
            
            # Try to import the data from the old format
            try:
                imported_data = json_to_data(ref_json_str)
                self.assertIsInstance(imported_data, HoloData)
            except Exception as e:
                self.fail(f"Failed to import data from {version} format: {str(e)}")

if __name__ == '__main__':
    unittest.main() 