"""
Data Formats Module for HoloPy.

This module defines standard data formats and structures for storing and
exchanging data in the HoloPy library.
"""

import numpy as np
import json
import logging
from typing import Dict, Any, List, Tuple, Union, Optional
from dataclasses import dataclass, field
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)

class DataFormat(Enum):
    """Enumeration of supported data formats."""
    NUMPY = "numpy"
    JSON = "json"
    HDF5 = "hdf5"
    CSV = "csv"
    FITS = "fits"

@dataclass
class MetaData:
    """
    Class for storing metadata about HoloPy datasets.
    
    Attributes:
        creator: Name or identifier of the creator
        creation_date: Creation date of the dataset
        description: Description of the dataset
        version: Version of HoloPy used to create the dataset
        parameters: Additional parameters specific to the dataset
    """
    creator: str = "HoloPy"
    creation_date: str = ""
    description: str = ""
    version: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HoloData:
    """
    Base class for HoloPy data structures.
    
    Attributes:
        data: The actual data (numpy array, dictionary, etc.)
        metadata: Metadata about the dataset
        format: Format of the data
    """
    data: Any
    metadata: MetaData = field(default_factory=MetaData)
    format: DataFormat = DataFormat.NUMPY

    def __post_init__(self):
        """Validate data after initialization."""
        if self.format == DataFormat.NUMPY and not isinstance(self.data, np.ndarray):
            logger.warning(f"Data is marked as NUMPY but is of type {type(self.data)}")

@dataclass
class E8Data(HoloData):
    """
    Class for storing E8 lattice and root system data.
    
    Attributes:
        dimension: Dimension of the E8 data (usually 8 or 16 for E8×E8)
        root_count: Number of roots in the dataset
    """
    dimension: int = 8
    root_count: int = 240

@dataclass
class InformationTensorData(HoloData):
    """
    Class for storing information current tensor data.
    
    Attributes:
        dimension: Spacetime dimension (usually 4)
        coordinates: Coordinate system used ('cartesian', 'spherical', etc.)
        has_density: Whether the dataset includes density information
    """
    dimension: int = 4
    coordinates: str = "cartesian"
    has_density: bool = True

@dataclass
class QuantumData(HoloData):
    """
    Class for storing quantum state and evolution data.
    
    Attributes:
        is_density_matrix: Whether the data is a density matrix or a wavefunction
        time_dependent: Whether the data includes time evolution
        decoherence_included: Whether decoherence effects are included
    """
    is_density_matrix: bool = False
    time_dependent: bool = False
    decoherence_included: bool = False

@dataclass
class CosmologyData(HoloData):
    """
    Class for storing cosmological simulation data.
    
    Attributes:
        redshift_range: Range of redshifts covered
        parameters: Cosmological parameters used
        observables: List of observables included in the data
    """
    redshift_range: Tuple[float, float] = (0.0, 0.0)
    parameters: Dict[str, float] = field(default_factory=dict)
    observables: List[str] = field(default_factory=list)

def data_to_json(data: HoloData) -> str:
    """
    Convert HoloData to a JSON string.
    
    Args:
        data: HoloData object to convert
        
    Returns:
        JSON string representation
    """
    # Create a dictionary version of the data
    result = {
        "format": data.format.value,
        "metadata": {
            "creator": data.metadata.creator,
            "creation_date": data.metadata.creation_date,
            "description": data.metadata.description,
            "version": data.metadata.version,
            "parameters": data.metadata.parameters
        }
    }
    
    # Add specific fields based on data type
    if isinstance(data, E8Data):
        result["type"] = "E8Data"
        result["dimension"] = data.dimension
        result["root_count"] = data.root_count
    elif isinstance(data, InformationTensorData):
        result["type"] = "InformationTensorData"
        result["dimension"] = data.dimension
        result["coordinates"] = data.coordinates
        result["has_density"] = data.has_density
    elif isinstance(data, QuantumData):
        result["type"] = "QuantumData"
        result["is_density_matrix"] = data.is_density_matrix
        result["time_dependent"] = data.time_dependent
        result["decoherence_included"] = data.decoherence_included
    elif isinstance(data, CosmologyData):
        result["type"] = "CosmologyData"
        result["redshift_range"] = data.redshift_range
        result["parameters"] = data.parameters
        result["observables"] = data.observables
    
    # Handle the actual data
    if data.format == DataFormat.NUMPY:
        if isinstance(data.data, np.ndarray):
            result["data"] = data.data.tolist()
        else:
            result["data"] = data.data
    else:
        result["data"] = data.data
    
    return json.dumps(result)

def json_to_data(json_str: str) -> HoloData:
    """
    Convert a JSON string to a HoloData object.
    
    Args:
        json_str: JSON string to convert
        
    Returns:
        HoloData object
    """
    data_dict = json.loads(json_str)
    
    # Create metadata
    metadata = MetaData(
        creator=data_dict["metadata"]["creator"],
        creation_date=data_dict["metadata"]["creation_date"],
        description=data_dict["metadata"]["description"],
        version=data_dict["metadata"]["version"],
        parameters=data_dict["metadata"]["parameters"]
    )
    
    # Get format
    format_str = data_dict["format"]
    format_enum = DataFormat(format_str)
    
    # Convert data back to numpy if needed
    data_content = data_dict["data"]
    if format_enum == DataFormat.NUMPY:
        data_content = np.array(data_content)
    
    # Create the appropriate data object based on type
    data_type = data_dict.get("type", "HoloData")
    
    if data_type == "E8Data":
        return E8Data(
            data=data_content,
            metadata=metadata,
            format=format_enum,
            dimension=data_dict["dimension"],
            root_count=data_dict["root_count"]
        )
    elif data_type == "InformationTensorData":
        return InformationTensorData(
            data=data_content,
            metadata=metadata,
            format=format_enum,
            dimension=data_dict["dimension"],
            coordinates=data_dict["coordinates"],
            has_density=data_dict["has_density"]
        )
    elif data_type == "QuantumData":
        return QuantumData(
            data=data_content,
            metadata=metadata,
            format=format_enum,
            is_density_matrix=data_dict["is_density_matrix"],
            time_dependent=data_dict["time_dependent"],
            decoherence_included=data_dict["decoherence_included"]
        )
    elif data_type == "CosmologyData":
        return CosmologyData(
            data=data_content,
            metadata=metadata,
            format=format_enum,
            redshift_range=tuple(data_dict["redshift_range"]),
            parameters=data_dict["parameters"],
            observables=data_dict["observables"]
        )
    else:
        return HoloData(
            data=data_content,
            metadata=metadata,
            format=format_enum
        ) 