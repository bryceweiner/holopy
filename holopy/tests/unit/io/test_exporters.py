"""
Unit tests for the exporters module.
"""

import unittest
import os
import json
import tempfile
import numpy as np
from holopy.io.data_formats import (
    DataFormat, HoloData, E8Data, InformationTensorData, QuantumData, CosmologyData
)
from holopy.io.exporters import (
    export_json, export_csv, export_data, export_hdf5, export_fits
)

# Check for optional dependencies
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    from astropy.io import fits
    FITS_AVAILABLE = True
except ImportError:
    FITS_AVAILABLE = False

class TestExporters(unittest.TestCase):
    """Tests for the exporters module functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create test data
        self.test_array = np.eye(3)
        self.test_data = HoloData(self.test_array)
        
        # Create specialized test data
        self.e8_data = E8Data(
            np.random.rand(240, 8), 
            dimension=8, 
            root_count=240
        )
        
        self.info_data = InformationTensorData(
            np.random.rand(4, 4),
            dimension=4,
            coordinates="cartesian",
            has_density=True
        )
        
        self.quantum_data = QuantumData(
            np.random.rand(10, 10),
            is_density_matrix=True,
            time_dependent=True,
            decoherence_included=False
        )
        
        self.cosmology_data = CosmologyData(
            np.random.rand(100),
            redshift_range=(0.0, 2.0),
            parameters={"H0": 70.0},
            observables=["CMB"]
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_export_json(self):
        """Test export_json function."""
        # Define output file path
        filepath = os.path.join(self.temp_dir.name, "test.json")
        
        # Export data
        export_json(self.test_data, filepath)
        
        # Check that file exists
        self.assertTrue(os.path.exists(filepath))
        
        # Check file content
        with open(filepath, 'r') as f:
            content = json.load(f)
        
        self.assertEqual(content["format"], "numpy")
        self.assertTrue(np.array_equal(np.array(content["data"]), self.test_array))
        
        # Test with pretty formatting
        filepath = os.path.join(self.temp_dir.name, "test_pretty.json")
        export_json(self.test_data, filepath, pretty=True)
        self.assertTrue(os.path.exists(filepath))
    
    def test_export_csv(self):
        """Test export_csv function."""
        # Define output file path
        filepath = os.path.join(self.temp_dir.name, "test.csv")
        
        # Export data
        export_csv(self.test_data, filepath)
        
        # Check that file exists
        self.assertTrue(os.path.exists(filepath))
        
        # Check file content - should contain the identity matrix
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Skip metadata lines (first 5 lines)
        data_lines = lines[6:]  # First line after metadata is header
        self.assertEqual(len(data_lines), 3)  # 3x3 matrix = 3 rows
        
        # Test with dictionary data
        dict_data = HoloData({
            "x": np.array([1, 2, 3]),
            "y": np.array([4, 5, 6])
        })
        
        filepath = os.path.join(self.temp_dir.name, "test_dict.csv")
        export_csv(dict_data, filepath)
        self.assertTrue(os.path.exists(filepath))
        
        # Test with unsupported data
        with self.assertRaises(ValueError):
            export_csv(HoloData(np.ones((3, 3, 3))), filepath)
    
    @unittest.skipIf(not HDF5_AVAILABLE, "h5py not available")
    def test_export_hdf5(self):
        """Test export_hdf5 function."""
        # Define output file path
        filepath = os.path.join(self.temp_dir.name, "test.h5")
        
        # Export data
        export_hdf5(self.test_data, filepath)
        
        # Check that file exists
        self.assertTrue(os.path.exists(filepath))
        
        # Check file content
        with h5py.File(filepath, 'r') as f:
            self.assertIn('data', f)
            self.assertIn('metadata', f)
            self.assertTrue(np.array_equal(f['data'][()], self.test_array))
        
        # Test with E8Data
        filepath = os.path.join(self.temp_dir.name, "test_e8.h5")
        export_hdf5(self.e8_data, filepath)
        self.assertTrue(os.path.exists(filepath))
        
        with h5py.File(filepath, 'r') as f:
            self.assertEqual(f.attrs['dimension'], 8)
            self.assertEqual(f.attrs['root_count'], 240)
    
    @unittest.skipIf(not FITS_AVAILABLE, "astropy not available")
    def test_export_fits(self):
        """Test export_fits function."""
        # Define output file path
        filepath = os.path.join(self.temp_dir.name, "test.fits")
        
        # Export data
        export_fits(self.test_data, filepath)
        
        # Check that file exists
        self.assertTrue(os.path.exists(filepath))
        
        # Check file content
        with fits.open(filepath) as hdul:
            self.assertEqual(len(hdul), 1)  # One HDU
            self.assertTrue(np.array_equal(hdul[0].data, self.test_array))
            self.assertIn('CREATOR', hdul[0].header)
        
        # Test with dictionary data
        dict_data = HoloData({
            "primary": np.eye(3),
            "secondary": np.ones((2, 2))
        })
        
        filepath = os.path.join(self.temp_dir.name, "test_dict.fits")
        export_fits(dict_data, filepath)
        self.assertTrue(os.path.exists(filepath))
        
        with fits.open(filepath) as hdul:
            self.assertEqual(len(hdul), 3)  # Primary + 2 extensions
    
    def test_export_data(self):
        """Test export_data function."""
        # Test with explicit format
        filepath = os.path.join(self.temp_dir.name, "test_explicit.json")
        export_data(self.test_data, filepath, format=DataFormat.JSON)
        self.assertTrue(os.path.exists(filepath))
        
        # Test with format inferred from extension
        filepath = os.path.join(self.temp_dir.name, "test_inferred.json")
        export_data(self.test_data, filepath)
        self.assertTrue(os.path.exists(filepath))
        
        # Test with no extension but format specified
        filepath = os.path.join(self.temp_dir.name, "test_no_ext")
        export_data(self.test_data, filepath, format=DataFormat.JSON)
        self.assertTrue(os.path.exists(filepath + ".json"))
        
        # Test with invalid format
        with self.assertRaises(ValueError):
            export_data(self.test_data, "invalid_format.xyz")
    
    def test_specialized_exports(self):
        """Test exporting specialized data types."""
        # Export each specialized data type to JSON
        for i, data in enumerate([self.e8_data, self.info_data, self.quantum_data, self.cosmology_data]):
            filepath = os.path.join(self.temp_dir.name, f"specialized_{i}.json")
            export_json(data, filepath)
            self.assertTrue(os.path.exists(filepath))
            
            # Verify content
            with open(filepath, 'r') as f:
                content = json.load(f)
            
            # Check type-specific fields
            if isinstance(data, E8Data):
                self.assertEqual(content["type"], "E8Data")
                self.assertEqual(content["dimension"], 8)
                self.assertEqual(content["root_count"], 240)
            elif isinstance(data, InformationTensorData):
                self.assertEqual(content["type"], "InformationTensorData")
                self.assertEqual(content["dimension"], 4)
                self.assertEqual(content["coordinates"], "cartesian")
            elif isinstance(data, QuantumData):
                self.assertEqual(content["type"], "QuantumData")
                self.assertEqual(content["is_density_matrix"], True)
                self.assertEqual(content["time_dependent"], True)
                self.assertEqual(content["decoherence_included"], False)
            elif isinstance(data, CosmologyData):
                self.assertEqual(content["type"], "CosmologyData")
                self.assertEqual(content["redshift_range"], [0.0, 2.0])
                self.assertEqual(content["parameters"], {"H0": 70.0})
                self.assertEqual(content["observables"], ["CMB"])

if __name__ == '__main__':
    unittest.main() 