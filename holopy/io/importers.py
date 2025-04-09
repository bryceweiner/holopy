"""
Importers Module for HoloPy.

This module provides functions for importing data from various formats,
such as JSON, HDF5, CSV, and FITS files, and converting them to HoloPy data structures.
"""

import os
import json
import numpy as np
import csv
import logging
from typing import Dict, Any, Union, Optional, List, Tuple, Type
import datetime

from holopy.io.data_formats import (
    HoloData, E8Data, InformationTensorData, QuantumData, CosmologyData,
    json_to_data, DataFormat, MetaData
)

# Setup logging
logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    logger.warning("h5py not available - HDF5 export/import functionality disabled")

try:
    from astropy.io import fits
    FITS_AVAILABLE = True
except ImportError:
    FITS_AVAILABLE = False
    logger.warning("astropy not available - FITS export/import functionality disabled")

def import_json(filepath: str) -> HoloData:
    """
    Import data from a JSON file.
    
    Args:
        filepath: Path to the input file
        
    Returns:
        HoloData object
    """
    with open(filepath, 'r') as f:
        json_str = f.read()
    
    data = json_to_data(json_str)
    logger.info(f"Data imported from JSON file: {filepath}")
    
    return data

def import_hdf5(filepath: str) -> HoloData:
    """
    Import data from an HDF5 file.
    
    Args:
        filepath: Path to the input file
        
    Returns:
        HoloData object
        
    Raises:
        ImportError: If h5py is not available
    """
    if not HDF5_AVAILABLE:
        raise ImportError("h5py is required for HDF5 import functionality")
    
    with h5py.File(filepath, 'r') as f:
        # Create metadata
        metadata = MetaData()
        
        if 'metadata' in f:
            metadata_group = f['metadata']
            metadata.creator = metadata_group.attrs.get('creator', metadata.creator)
            metadata.creation_date = metadata_group.attrs.get('creation_date', metadata.creation_date)
            metadata.description = metadata_group.attrs.get('description', metadata.description)
            metadata.version = metadata_group.attrs.get('version', metadata.version)
            
            if 'parameters' in metadata_group:
                param_group = metadata_group['parameters']
                for key in param_group.attrs:
                    metadata.parameters[key] = param_group.attrs[key]
        
        # Get data type
        data_type = f.attrs.get('data_type', 'HoloData')
        
        # Get the actual data
        if 'data' in f:
            if isinstance(f['data'], h5py.Dataset):
                data_content = f['data'][()]
            else:
                # Data is a group with multiple datasets
                data_content = {}
                for key in f['data']:
                    if isinstance(f['data'][key], h5py.Dataset):
                        data_content[key] = f['data'][key][()]
                    else:
                        logger.warning(f"Skipping non-dataset item in data group: {key}")
        else:
            logger.warning("No data found in HDF5 file, using empty array")
            data_content = np.array([])
        
        # Create the appropriate data object based on type
        if data_type == 'E8Data':
            dimension = f.attrs.get('dimension', 8)
            root_count = f.attrs.get('root_count', 240)
            
            return E8Data(
                data=data_content,
                metadata=metadata,
                format=DataFormat.NUMPY,
                dimension=dimension,
                root_count=root_count
            )
        elif data_type == 'InformationTensorData':
            dimension = f.attrs.get('dimension', 4)
            coordinates = f.attrs.get('coordinates', 'cartesian')
            has_density = f.attrs.get('has_density', True)
            
            return InformationTensorData(
                data=data_content,
                metadata=metadata,
                format=DataFormat.NUMPY,
                dimension=dimension,
                coordinates=coordinates,
                has_density=has_density
            )
        elif data_type == 'QuantumData':
            is_density_matrix = f.attrs.get('is_density_matrix', False)
            time_dependent = f.attrs.get('time_dependent', False)
            decoherence_included = f.attrs.get('decoherence_included', False)
            
            return QuantumData(
                data=data_content,
                metadata=metadata,
                format=DataFormat.NUMPY,
                is_density_matrix=is_density_matrix,
                time_dependent=time_dependent,
                decoherence_included=decoherence_included
            )
        elif data_type == 'CosmologyData':
            redshift_min = f.attrs.get('redshift_min', 0.0)
            redshift_max = f.attrs.get('redshift_max', 0.0)
            redshift_range = (redshift_min, redshift_max)
            
            # Get cosmology parameters
            parameters = {}
            if 'cosmology_parameters' in f:
                param_group = f['cosmology_parameters']
                for key in param_group.attrs:
                    parameters[key] = param_group.attrs[key]
            
            # Get observables
            observables = []
            if 'observables' in f:
                observables_dataset = f['observables']
                observables = [o.decode('utf-8') if isinstance(o, bytes) else o 
                               for o in observables_dataset[()]]
            
            return CosmologyData(
                data=data_content,
                metadata=metadata,
                format=DataFormat.NUMPY,
                redshift_range=redshift_range,
                parameters=parameters,
                observables=observables
            )
        else:
            return HoloData(
                data=data_content,
                metadata=metadata,
                format=DataFormat.NUMPY
            )

def import_csv(filepath: str, data_class: Type[HoloData] = HoloData) -> HoloData:
    """
    Import data from a CSV file.
    
    Args:
        filepath: Path to the input file
        data_class: Class to use for the data object
        
    Returns:
        HoloData object
    """
    # Read the CSV file
    with open(filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        
        # Process header and metadata
        metadata = MetaData()
        data_type = None
        header = None
        
        # Read initial rows to find metadata and header
        rows = []
        for i, row in enumerate(reader):
            if i < 5 and row and row[0].startswith('#'):
                # This is a metadata row
                if "Creator:" in row[0]:
                    metadata.creator = row[0].split("Creator:")[1].strip()
                elif "Date:" in row[0]:
                    metadata.creation_date = row[0].split("Date:")[1].strip()
                elif "Description:" in row[0]:
                    metadata.description = row[0].split("Description:")[1].strip()
                elif "Data Type:" in row[0]:
                    data_type = row[0].split("Data Type:")[1].strip()
            elif header is None:
                # This is the header row
                header = row
            else:
                # This is a data row
                rows.append(row)
        
        # Convert data rows to numpy array
        data = np.array(rows, dtype=float)
        
        # Create the appropriate data object based on type
        if data_type == 'E8Data':
            return E8Data(
                data=data,
                metadata=metadata,
                format=DataFormat.NUMPY
            )
        elif data_type == 'InformationTensorData':
            return InformationTensorData(
                data=data,
                metadata=metadata,
                format=DataFormat.NUMPY
            )
        elif data_type == 'QuantumData':
            return QuantumData(
                data=data,
                metadata=metadata,
                format=DataFormat.NUMPY
            )
        elif data_type == 'CosmologyData':
            return CosmologyData(
                data=data,
                metadata=metadata,
                format=DataFormat.NUMPY
            )
        else:
            return data_class(
                data=data,
                metadata=metadata,
                format=DataFormat.NUMPY
            )

def import_fits(filepath: str) -> HoloData:
    """
    Import data from a FITS file.
    
    Args:
        filepath: Path to the input file
        
    Returns:
        HoloData object
        
    Raises:
        ImportError: If astropy is not available
    """
    if not FITS_AVAILABLE:
        raise ImportError("astropy is required for FITS import functionality")
    
    with fits.open(filepath) as hdul:
        # Get primary HDU
        primary_hdu = hdul[0]
        
        # Extract metadata from header
        header = primary_hdu.header
        metadata = MetaData(
            creator=header.get('CREATOR', 'Unknown'),
            creation_date=header.get('CREATDAT', ''),
            description=header.get('DESCRIPT', ''),
            version=header.get('VERSION', '')
        )
        
        # Get data type
        data_type = header.get('DATATYPE', 'HoloData')
        
        # Get the actual data
        data_content = primary_hdu.data
        
        # Check for additional data in other HDUs
        if len(hdul) > 1:
            # If we have multiple HDUs, use a dictionary to store the data
            data_dict = {'primary': data_content}
            for i in range(1, len(hdul)):
                if hdul[i].name:
                    data_dict[hdul[i].name] = hdul[i].data
                else:
                    data_dict[f'hdu_{i}'] = hdul[i].data
            data_content = data_dict
        
        # Create the appropriate data object based on type
        if data_type == 'E8Data':
            dimension = header.get('E8DIM', 8)
            root_count = header.get('ROOTCNT', 240)
            
            return E8Data(
                data=data_content,
                metadata=metadata,
                format=DataFormat.NUMPY,
                dimension=dimension,
                root_count=root_count
            )
        elif data_type == 'InformationTensorData':
            dimension = header.get('SPDIM', 4)
            coordinates = header.get('COORDS', 'cartesian')
            has_density = header.get('HASDENS', True)
            
            return InformationTensorData(
                data=data_content,
                metadata=metadata,
                format=DataFormat.NUMPY,
                dimension=dimension,
                coordinates=coordinates,
                has_density=has_density
            )
        elif data_type == 'QuantumData':
            is_density_matrix = header.get('ISDENMAT', False)
            time_dependent = header.get('TIMEDEP', False)
            decoherence_included = header.get('DECOHERE', False)
            
            return QuantumData(
                data=data_content,
                metadata=metadata,
                format=DataFormat.NUMPY,
                is_density_matrix=is_density_matrix,
                time_dependent=time_dependent,
                decoherence_included=decoherence_included
            )
        elif data_type == 'CosmologyData':
            redshift_min = header.get('ZMIN', 0.0)
            redshift_max = header.get('ZMAX', 0.0)
            redshift_range = (redshift_min, redshift_max)
            
            return CosmologyData(
                data=data_content,
                metadata=metadata,
                format=DataFormat.NUMPY,
                redshift_range=redshift_range
            )
        else:
            return HoloData(
                data=data_content,
                metadata=metadata,
                format=DataFormat.NUMPY
            )

def import_data(filepath: str, format: Optional[DataFormat] = None) -> HoloData:
    """
    Import data from a file.
    
    Args:
        filepath: Path to the input file
        format: Format of the file (if None, inferred from extension)
        
    Returns:
        HoloData object
        
    Raises:
        ValueError: If an unsupported format is specified
    """
    # Check if file exists
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Get the file extension
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    
    # Infer format from extension if not specified
    if format is None:
        if ext == '.json':
            format = DataFormat.JSON
        elif ext == '.h5' or ext == '.hdf5':
            format = DataFormat.HDF5
        elif ext == '.csv':
            format = DataFormat.CSV
        elif ext == '.fits' or ext == '.fit':
            format = DataFormat.FITS
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    # Import based on format
    if format == DataFormat.JSON:
        return import_json(filepath)
    elif format == DataFormat.HDF5:
        return import_hdf5(filepath)
    elif format == DataFormat.CSV:
        return import_csv(filepath)
    elif format == DataFormat.FITS:
        return import_fits(filepath)
    else:
        raise ValueError(f"Unsupported import format: {format}") 