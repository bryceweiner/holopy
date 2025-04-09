"""
Unit tests for the data_formats module.
"""

import unittest
import json
import numpy as np
from holopy.io.data_formats import (
    DataFormat, MetaData, HoloData, E8Data, InformationTensorData,
    QuantumData, CosmologyData, data_to_json, json_to_data
)

class TestDataFormats(unittest.TestCase):
    """Tests for the data_formats module classes and functions."""
    
    def test_metadata(self):
        """Test MetaData class."""
        # Test default initialization
        metadata = MetaData()
        self.assertEqual(metadata.creator, "HoloPy")
        self.assertEqual(metadata.parameters, {})
        
        # Test custom initialization
        metadata = MetaData(
            creator="Test User",
            creation_date="2023-01-01",
            description="Test description",
            version="1.0.0",
            parameters={"param1": 1, "param2": "test"}
        )
        self.assertEqual(metadata.creator, "Test User")
        self.assertEqual(metadata.creation_date, "2023-01-01")
        self.assertEqual(metadata.description, "Test description")
        self.assertEqual(metadata.version, "1.0.0")
        self.assertEqual(metadata.parameters, {"param1": 1, "param2": "test"})
    
    def test_holodata(self):
        """Test HoloData class."""
        # Test with numpy array
        data = np.eye(3)
        holo_data = HoloData(data)
        self.assertTrue(np.array_equal(holo_data.data, data))
        self.assertEqual(holo_data.format, DataFormat.NUMPY)
        
        # Test with other data type
        data = {"key1": [1, 2, 3], "key2": "value"}
        holo_data = HoloData(data, format=DataFormat.JSON)
        self.assertEqual(holo_data.data, data)
        self.assertEqual(holo_data.format, DataFormat.JSON)
    
    def test_e8data(self):
        """Test E8Data class."""
        data = np.random.rand(240, 8)
        e8_data = E8Data(data, dimension=8, root_count=240)
        self.assertTrue(np.array_equal(e8_data.data, data))
        self.assertEqual(e8_data.dimension, 8)
        self.assertEqual(e8_data.root_count, 240)
    
    def test_information_tensor_data(self):
        """Test InformationTensorData class."""
        data = np.random.rand(4, 4)
        info_data = InformationTensorData(
            data, dimension=4, coordinates="spherical", has_density=True
        )
        self.assertTrue(np.array_equal(info_data.data, data))
        self.assertEqual(info_data.dimension, 4)
        self.assertEqual(info_data.coordinates, "spherical")
        self.assertTrue(info_data.has_density)
    
    def test_quantum_data(self):
        """Test QuantumData class."""
        data = np.random.rand(10, 10)
        quantum_data = QuantumData(
            data, is_density_matrix=True, time_dependent=True, decoherence_included=True
        )
        self.assertTrue(np.array_equal(quantum_data.data, data))
        self.assertTrue(quantum_data.is_density_matrix)
        self.assertTrue(quantum_data.time_dependent)
        self.assertTrue(quantum_data.decoherence_included)
    
    def test_cosmology_data(self):
        """Test CosmologyData class."""
        data = np.random.rand(100)
        cosmology_data = CosmologyData(
            data,
            redshift_range=(0.0, 10.0),
            parameters={"H0": 70.0, "Omega_m": 0.3},
            observables=["CMB", "BAO", "SNe"]
        )
        self.assertTrue(np.array_equal(cosmology_data.data, data))
        self.assertEqual(cosmology_data.redshift_range, (0.0, 10.0))
        self.assertEqual(cosmology_data.parameters, {"H0": 70.0, "Omega_m": 0.3})
        self.assertEqual(cosmology_data.observables, ["CMB", "BAO", "SNe"])
    
    def test_data_to_json(self):
        """Test data_to_json function."""
        # Test with HoloData
        data = np.eye(3)
        metadata = MetaData(creator="Test User", description="Test description")
        holo_data = HoloData(data, metadata=metadata)
        
        # Convert to JSON
        json_str = data_to_json(holo_data)
        
        # Parse JSON and check content
        json_dict = json.loads(json_str)
        self.assertEqual(json_dict["format"], "numpy")
        self.assertEqual(json_dict["metadata"]["creator"], "Test User")
        self.assertEqual(json_dict["metadata"]["description"], "Test description")
        self.assertTrue(np.array_equal(np.array(json_dict["data"]), data))
        
        # Test with E8Data
        e8_data = E8Data(data, dimension=8, root_count=240)
        json_str = data_to_json(e8_data)
        json_dict = json.loads(json_str)
        self.assertEqual(json_dict["type"], "E8Data")
        self.assertEqual(json_dict["dimension"], 8)
        self.assertEqual(json_dict["root_count"], 240)
    
    def test_json_to_data(self):
        """Test json_to_data function."""
        # Create test data
        data = np.eye(3).tolist()
        json_dict = {
            "format": "numpy",
            "type": "E8Data",
            "metadata": {
                "creator": "Test User",
                "creation_date": "2023-01-01",
                "description": "Test description",
                "version": "1.0.0",
                "parameters": {}
            },
            "dimension": 8,
            "root_count": 240,
            "data": data
        }
        
        # Convert to JSON
        json_str = json.dumps(json_dict)
        
        # Convert back to HoloData
        holo_data = json_to_data(json_str)
        
        # Check result
        self.assertIsInstance(holo_data, E8Data)
        self.assertEqual(holo_data.metadata.creator, "Test User")
        self.assertEqual(holo_data.metadata.creation_date, "2023-01-01")
        self.assertEqual(holo_data.metadata.description, "Test description")
        self.assertEqual(holo_data.dimension, 8)
        self.assertEqual(holo_data.root_count, 240)
        self.assertTrue(np.array_equal(holo_data.data, np.array(data)))
    
    def test_serialization_roundtrip(self):
        """Test serialization and deserialization roundtrip."""
        # Create test data for each data type
        data_types = [
            HoloData(np.random.rand(3, 3)),
            E8Data(np.random.rand(240, 8), dimension=8, root_count=240),
            InformationTensorData(np.random.rand(4, 4), dimension=4, coordinates="cartesian"),
            QuantumData(np.random.rand(10), is_density_matrix=False, time_dependent=True),
            CosmologyData(np.random.rand(100), redshift_range=(0.0, 10.0))
        ]
        
        for original_data in data_types:
            # Serialize
            json_str = data_to_json(original_data)
            
            # Deserialize
            restored_data = json_to_data(json_str)
            
            # Check that types match
            self.assertEqual(type(restored_data), type(original_data))
            
            # Check that attributes match
            for attr in original_data.__dict__:
                # Skip data attribute for separate numpy comparison
                if attr == 'data':
                    continue
                
                # Skip metadata for separate comparison
                if attr == 'metadata':
                    continue
                
                self.assertEqual(getattr(restored_data, attr), getattr(original_data, attr))
            
            # Special comparison for numpy data
            if isinstance(original_data.data, np.ndarray):
                self.assertTrue(np.array_equal(restored_data.data, original_data.data))
                
            # Check metadata
            self.assertEqual(restored_data.metadata.creator, original_data.metadata.creator)
            self.assertEqual(restored_data.metadata.description, original_data.metadata.description)

if __name__ == '__main__':
    unittest.main() 