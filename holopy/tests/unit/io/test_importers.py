"""
Unit tests for the importers module.
"""

import unittest
import os
import json
import tempfile
import numpy as np
from holopy.io.data_formats import (
    DataFormat, HoloData, E8Data, InformationTensorData, QuantumData, CosmologyData,
    data_to_json
)
from holopy.io.importers import (
    import_json, import_csv, import_data, import_hdf5, import_fits
)
from holopy.io.exporters import (
    export_json, export_csv, export_hdf5, export_fits
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

class TestImporters(unittest.TestCase):
    """Tests for the importers module functions."""
    
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
    
    def test_import_json(self):
        """Test import_json function."""
        # First export data to JSON
        filepath = os.path.join(self.temp_dir.name, "test.json")
        export_json(self.test_data, filepath)
        
        # Import the data
        imported_data = import_json(filepath)
        
        # Check that the imported data matches the original
        self.assertIsInstance(imported_data, HoloData)
        self.assertTrue(np.array_equal(imported_data.data, self.test_array))
        
        # Export specialized data
        e8_filepath = os.path.join(self.temp_dir.name, "e8_test.json")
        export_json(self.e8_data, e8_filepath)
        
        # Import specialized data
        imported_e8 = import_json(e8_filepath)
        
        # Check specialized data
        self.assertIsInstance(imported_e8, E8Data)
        self.assertEqual(imported_e8.dimension, 8)
        self.assertEqual(imported_e8.root_count, 240)
    
    def test_import_csv(self):
        """Test import_csv function."""
        # First export data to CSV
        filepath = os.path.join(self.temp_dir.name, "test.csv")
        export_csv(self.test_data, filepath)
        
        # Import the data
        imported_data = import_csv(filepath)
        
        # Check that the imported data has the correct type
        self.assertIsInstance(imported_data, HoloData)
        
        # Create a CSV file with metadata indicating it's E8Data
        with open(os.path.join(self.temp_dir.name, "e8_test.csv"), 'w', newline='') as f:
            f.write("# HoloPy Data Export\n")
            f.write("# Creator: Test\n")
            f.write("# Date: 2023-01-01\n")
            f.write("# Description: Test E8 data\n")
            f.write("# Data Type: E8Data\n")
            f.write("Column_0,Column_1\n")
            f.write("1.0,2.0\n")
            f.write("3.0,4.0\n")
        
        # Import the data
        imported_e8 = import_csv(os.path.join(self.temp_dir.name, "e8_test.csv"))
        
        # Check that it's recognized as E8Data
        self.assertIsInstance(imported_e8, E8Data)
        self.assertEqual(imported_e8.metadata.creator, "Test")
        self.assertEqual(imported_e8.metadata.creation_date, "2023-01-01")
    
    @unittest.skipIf(not HDF5_AVAILABLE, "h5py not available")
    def test_import_hdf5(self):
        """Test import_hdf5 function."""
        # First export data to HDF5
        filepath = os.path.join(self.temp_dir.name, "test.h5")
        export_hdf5(self.test_data, filepath)
        
        # Import the data
        imported_data = import_hdf5(filepath)
        
        # Check that the imported data matches the original
        self.assertIsInstance(imported_data, HoloData)
        self.assertTrue(np.array_equal(imported_data.data, self.test_array))
        
        # Export specialized data
        for name, data in [
            ("e8", self.e8_data),
            ("info", self.info_data),
            ("quantum", self.quantum_data),
            ("cosmology", self.cosmology_data),
        ]:
            spec_filepath = os.path.join(self.temp_dir.name, f"{name}_test.h5")
            export_hdf5(data, spec_filepath)
            
            # Import specialized data
            imported_spec = import_hdf5(spec_filepath)
            
            # Check type
            self.assertIsInstance(imported_spec, type(data))
            
            # Check specialized attributes
            if isinstance(data, E8Data):
                self.assertEqual(imported_spec.dimension, data.dimension)
                self.assertEqual(imported_spec.root_count, data.root_count)
            elif isinstance(data, InformationTensorData):
                self.assertEqual(imported_spec.dimension, data.dimension)
                self.assertEqual(imported_spec.coordinates, data.coordinates)
            elif isinstance(data, QuantumData):
                self.assertEqual(imported_spec.is_density_matrix, data.is_density_matrix)
                self.assertEqual(imported_spec.time_dependent, data.time_dependent)
            elif isinstance(data, CosmologyData):
                self.assertEqual(imported_spec.redshift_range, data.redshift_range)
    
    @unittest.skipIf(not FITS_AVAILABLE, "astropy not available")
    def test_import_fits(self):
        """Test import_fits function."""
        # First export data to FITS
        filepath = os.path.join(self.temp_dir.name, "test.fits")
        export_fits(self.test_data, filepath)
        
        # Import the data
        imported_data = import_fits(filepath)
        
        # Check that the imported data matches the original
        self.assertIsInstance(imported_data, HoloData)
        self.assertTrue(np.array_equal(imported_data.data, self.test_array))
        
        # Export specialized data
        for name, data in [
            ("e8", self.e8_data),
            ("info", self.info_data),
            ("quantum", self.quantum_data),
            ("cosmology", self.cosmology_data),
        ]:
            spec_filepath = os.path.join(self.temp_dir.name, f"{name}_test.fits")
            export_fits(data, spec_filepath)
            
            # Import specialized data
            imported_spec = import_fits(spec_filepath)
            
            # Check type
            self.assertIsInstance(imported_spec, type(data))
    
    def test_import_data(self):
        """Test import_data function."""
        # Test with explicit format
        json_filepath = os.path.join(self.temp_dir.name, "test.json")
        export_json(self.test_data, json_filepath)
        
        imported_data = import_data(json_filepath, format=DataFormat.JSON)
        self.assertIsInstance(imported_data, HoloData)
        
        # Test with format inferred from extension
        imported_data = import_data(json_filepath)
        self.assertIsInstance(imported_data, HoloData)
        
        # Test with invalid format
        with self.assertRaises(ValueError):
            import_data(json_filepath, format="invalid")
        
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            import_data("nonexistent_file.json")
    
    def test_roundtrip_specialized_data(self):
        """Test roundtrip export/import for specialized data types."""
        # Test roundtrip for each specialized data type with JSON
        for name, data in [
            ("e8", self.e8_data),
            ("info", self.info_data),
            ("quantum", self.quantum_data),
            ("cosmology", self.cosmology_data),
        ]:
            # Export to JSON
            json_filepath = os.path.join(self.temp_dir.name, f"{name}_roundtrip.json")
            export_json(data, json_filepath)
            
            # Import from JSON
            imported_data = import_json(json_filepath)
            
            # Check type
            self.assertIsInstance(imported_data, type(data))
            
            # Check attributes
            if isinstance(data, E8Data):
                self.assertEqual(imported_data.dimension, data.dimension)
                self.assertEqual(imported_data.root_count, data.root_count)
            elif isinstance(data, InformationTensorData):
                self.assertEqual(imported_data.dimension, data.dimension)
                self.assertEqual(imported_data.coordinates, data.coordinates)
                self.assertEqual(imported_data.has_density, data.has_density)
            elif isinstance(data, QuantumData):
                self.assertEqual(imported_data.is_density_matrix, data.is_density_matrix)
                self.assertEqual(imported_data.time_dependent, data.time_dependent)
                self.assertEqual(imported_data.decoherence_included, data.decoherence_included)
            elif isinstance(data, CosmologyData):
                self.assertEqual(imported_data.redshift_range, data.redshift_range)
                self.assertEqual(sorted(imported_data.observables), sorted(data.observables))
                for key in data.parameters:
                    self.assertEqual(imported_data.parameters.get(key), data.parameters[key])

if __name__ == '__main__':
    unittest.main() 