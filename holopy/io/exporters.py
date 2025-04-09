"""
Exporters Module for HoloPy.

This module provides functions for exporting HoloPy data to various formats,
such as JSON, HDF5, CSV, and FITS files.
"""

import os
import json
import numpy as np
import csv
import logging
from typing import Dict, Any, Union, Optional, List, Tuple
import datetime

from holopy.io.data_formats import (
    HoloData, E8Data, InformationTensorData, QuantumData, CosmologyData,
    data_to_json, DataFormat
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

def export_json(data: HoloData, filepath: str, pretty: bool = False) -> None:
    """
    Export data to a JSON file.
    
    Args:
        data: HoloData object to export
        filepath: Path to the output file
        pretty: Whether to format the JSON for readability
    """
    # Add creation date if not set
    if not data.metadata.creation_date:
        data.metadata.creation_date = datetime.datetime.now().isoformat()
    
    # Convert to JSON
    json_str = data_to_json(data)
    
    # Pretty print if requested
    if pretty:
        json_data = json.loads(json_str)
        json_str = json.dumps(json_data, indent=2)
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write(json_str)
    
    logger.info(f"Data exported to JSON file: {filepath}")

def export_hdf5(data: HoloData, filepath: str) -> None:
    """
    Export data to an HDF5 file.
    
    Args:
        data: HoloData object to export
        filepath: Path to the output file
    
    Raises:
        ImportError: If h5py is not available
    """
    if not HDF5_AVAILABLE:
        raise ImportError("h5py is required for HDF5 export functionality")
    
    # Create HDF5 file
    with h5py.File(filepath, 'w') as f:
        # Store metadata
        metadata_group = f.create_group('metadata')
        metadata_group.attrs['creator'] = data.metadata.creator
        metadata_group.attrs['creation_date'] = (data.metadata.creation_date or 
                                                datetime.datetime.now().isoformat())
        metadata_group.attrs['description'] = data.metadata.description
        metadata_group.attrs['version'] = data.metadata.version
        
        # Store parameters as separate dataset
        if data.metadata.parameters:
            param_group = metadata_group.create_group('parameters')
            for key, value in data.metadata.parameters.items():
                param_group.attrs[key] = value
        
        # Store the type of data
        data_type = type(data).__name__
        f.attrs['data_type'] = data_type
        
        # Store specific attributes based on data type
        if isinstance(data, E8Data):
            f.attrs['dimension'] = data.dimension
            f.attrs['root_count'] = data.root_count
        elif isinstance(data, InformationTensorData):
            f.attrs['dimension'] = data.dimension
            f.attrs['coordinates'] = data.coordinates
            f.attrs['has_density'] = data.has_density
        elif isinstance(data, QuantumData):
            f.attrs['is_density_matrix'] = data.is_density_matrix
            f.attrs['time_dependent'] = data.time_dependent
            f.attrs['decoherence_included'] = data.decoherence_included
        elif isinstance(data, CosmologyData):
            f.attrs['redshift_min'] = data.redshift_range[0]
            f.attrs['redshift_max'] = data.redshift_range[1]
            
            # Store parameters as separate dataset
            if data.parameters:
                param_group = f.create_group('cosmology_parameters')
                for key, value in data.parameters.items():
                    param_group.attrs[key] = value
            
            # Store observables
            if data.observables:
                f.create_dataset('observables', data=np.array(data.observables, dtype='S'))
        
        # Store the actual data
        if isinstance(data.data, np.ndarray):
            f.create_dataset('data', data=data.data)
        elif isinstance(data.data, dict):
            data_group = f.create_group('data')
            for key, value in data.data.items():
                if isinstance(value, np.ndarray):
                    data_group.create_dataset(key, data=value)
                else:
                    # Try to store as attribute if simple type
                    try:
                        data_group.attrs[key] = value
                    except:
                        logger.warning(f"Could not store key {key} in HDF5 file")
        else:
            # Try to convert to numpy array
            try:
                f.create_dataset('data', data=np.array(data.data))
            except:
                logger.error(f"Could not store data of type {type(data.data)} in HDF5 file")
                raise ValueError(f"Cannot export data of type {type(data.data)} to HDF5")
    
    logger.info(f"Data exported to HDF5 file: {filepath}")

def export_csv(data: HoloData, filepath: str) -> None:
    """
    Export data to a CSV file.
    
    Only works for 1D or 2D numerical data (either arrays or dictionaries of 1D arrays).
    Includes metadata as comments in the header.
    
    Args:
        data: HoloData object to export
        filepath: Path to the output file
        
    Raises:
        ValueError: If data cannot be exported to CSV
    """
    # Check if data is exportable to CSV
    if isinstance(data.data, np.ndarray):
        if data.data.ndim > 2:
            raise ValueError(f"Cannot export {data.data.ndim}D array to CSV, only 1D or 2D supported")
        
        # Write the CSV file
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write metadata as comments
            writer.writerow(['# HoloPy Data Export'])
            writer.writerow([f'# Creator: {data.metadata.creator}'])
            writer.writerow([f'# Date: {data.metadata.creation_date or datetime.datetime.now().isoformat()}'])
            writer.writerow([f'# Description: {data.metadata.description}'])
            writer.writerow([f'# Data Type: {type(data).__name__}'])
            
            # Write the data
            if data.data.ndim == 1:
                writer.writerow(['Value'])
                for value in data.data:
                    writer.writerow([value])
            else:  # 2D array
                # Create header
                header = [f'Column_{i}' for i in range(data.data.shape[1])]
                writer.writerow(header)
                
                # Write the data rows
                for row in data.data:
                    writer.writerow(row)
    
    elif isinstance(data.data, dict):
        # Check if all values are 1D arrays of the same length
        if not all(isinstance(v, np.ndarray) and v.ndim == 1 for v in data.data.values()):
            raise ValueError("For dictionary data, all values must be 1D numpy arrays")
        
        # Check if all arrays have the same length
        lengths = [len(v) for v in data.data.values()]
        if not all(l == lengths[0] for l in lengths):
            raise ValueError("All arrays in dictionary data must have the same length")
        
        # Write the CSV file
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write metadata as comments
            writer.writerow(['# HoloPy Data Export'])
            writer.writerow([f'# Creator: {data.metadata.creator}'])
            writer.writerow([f'# Date: {data.metadata.creation_date or datetime.datetime.now().isoformat()}'])
            writer.writerow([f'# Description: {data.metadata.description}'])
            writer.writerow([f'# Data Type: {type(data).__name__}'])
            
            # Write the header
            header = list(data.data.keys())
            writer.writerow(header)
            
            # Write the data rows
            rows = list(zip(*[data.data[key] for key in header]))
            for row in rows:
                writer.writerow(row)
    
    else:
        raise ValueError(f"Cannot export data of type {type(data.data)} to CSV")
    
    logger.info(f"Data exported to CSV file: {filepath}")

def export_fits(data: HoloData, filepath: str) -> None:
    """
    Export data to a FITS file.
    
    Args:
        data: HoloData object to export
        filepath: Path to the output file
        
    Raises:
        ImportError: If astropy is not available
        ValueError: If data cannot be exported to FITS
    """
    if not FITS_AVAILABLE:
        raise ImportError("astropy is required for FITS export functionality")
    
    # Check if data is exportable to FITS
    if not isinstance(data.data, np.ndarray):
        # Handle dictionary data
        if isinstance(data.data, dict):
            # Use the first array in the dictionary as primary data if possible
            for key, value in data.data.items():
                if isinstance(value, np.ndarray):
                    array_data = value
                    break
            else:
                # If no arrays found, create an empty array
                logger.warning("No numpy arrays found in dictionary data, using empty array for primary HDU")
                array_data = np.array([])
        else:
            # Try to convert to numpy array
            try:
                array_data = np.array(data.data)
                if array_data.dtype.kind not in 'biufcO':  # Check if numeric or object type
                    logger.warning(f"Converted data has non-numeric dtype: {array_data.dtype}")
                    # Create a placeholder array
                    array_data = np.array([])
            except Exception as e:
                logger.error(f"Failed to convert data to numpy array: {str(e)}")
                # Create a placeholder array
                array_data = np.array([])
                logger.warning("Using empty array for primary HDU")
    else:
        array_data = data.data
    
    # Create primary HDU
    try:
        primary_hdu = fits.PrimaryHDU(array_data)
    except Exception as e:
        logger.error(f"Failed to create PrimaryHDU: {str(e)}")
        # Create an empty HDU as a fallback
        primary_hdu = fits.PrimaryHDU(np.array([]))
        logger.warning("Using empty array for primary HDU due to conversion error")
    
    # Add metadata to header
    primary_hdu.header['CREATOR'] = (data.metadata.creator, 'Creator of the data')
    primary_hdu.header['CREATDAT'] = (data.metadata.creation_date or 
                                     datetime.datetime.now().isoformat(), 'Creation date')
    primary_hdu.header['DESCRIPT'] = (data.metadata.description, 'Description')
    primary_hdu.header['VERSION'] = (data.metadata.version, 'HoloPy version')
    primary_hdu.header['DATATYPE'] = (type(data).__name__, 'Type of HoloPy data')
    
    # Add specific metadata based on data type
    if isinstance(data, E8Data):
        primary_hdu.header['E8DIM'] = (data.dimension, 'E8 dimension')
        primary_hdu.header['ROOTCNT'] = (data.root_count, 'Root count')
    elif isinstance(data, InformationTensorData):
        primary_hdu.header['SPDIM'] = (data.dimension, 'Spacetime dimension')
        primary_hdu.header['COORDS'] = (data.coordinates, 'Coordinate system')
        primary_hdu.header['HASDENS'] = (data.has_density, 'Has density information')
    elif isinstance(data, QuantumData):
        primary_hdu.header['ISDENMAT'] = (data.is_density_matrix, 'Is density matrix')
        primary_hdu.header['TIMEDEP'] = (data.time_dependent, 'Time dependent')
        primary_hdu.header['DECOHERE'] = (data.decoherence_included, 'Includes decoherence')
    elif isinstance(data, CosmologyData):
        primary_hdu.header['ZMIN'] = (data.redshift_range[0], 'Minimum redshift')
        primary_hdu.header['ZMAX'] = (data.redshift_range[1], 'Maximum redshift')
    
    # Create HDU list
    hdu_list = fits.HDUList([primary_hdu])
    
    # Add additional data if needed
    if isinstance(data.data, dict):
        for key, value in data.data.items():
            if isinstance(value, np.ndarray):
                try:
                    hdu = fits.ImageHDU(value, name=key)
                    hdu_list.append(hdu)
                except Exception as e:
                    logger.warning(f"Could not add {key} to FITS file: {str(e)}")
    
    # Write to file
    try:
        hdu_list.writeto(filepath, overwrite=True)
        logger.info(f"Data exported to FITS file: {filepath}")
    except Exception as e:
        logger.error(f"Failed to write FITS file: {str(e)}")
        raise

def export_data(data: HoloData, filepath: str, format: Optional[DataFormat] = None) -> None:
    """
    Export data to a file in the specified format.
    
    Args:
        data: HoloData object to export
        filepath: Path to the output file
        format: Format to use for export (defaults to format specified in data)
        
    Raises:
        ValueError: If an unsupported format is specified
    """
    # Use the format from the data object if none specified
    if format is None:
        format = data.format
    
    # Get the file extension
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    
    # If no extension and no format specified, default to JSON
    if not ext and format == None:
        format = DataFormat.JSON
        filepath += '.json'
    # If extension provided but no format, infer format from extension
    elif ext and format == None:
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
    
    # If format is NUMPY, convert to JSON for export
    if format == DataFormat.NUMPY:
        logger.info(f"Converting DataFormat.NUMPY to DataFormat.JSON for export")
        format = DataFormat.JSON
    
    # If no extension, add appropriate extension based on format
    if not ext:
        if format == DataFormat.JSON:
            filepath += '.json'
        elif format == DataFormat.HDF5:
            filepath += '.h5'
        elif format == DataFormat.CSV:
            filepath += '.csv'
        elif format == DataFormat.FITS:
            filepath += '.fits'
    # If extension doesn't match format, raise an error
    elif format == DataFormat.JSON and ext != '.json':
        raise ValueError(f"File extension {ext} doesn't match format {format}")
    elif format == DataFormat.HDF5 and ext not in ['.h5', '.hdf5']:
        raise ValueError(f"File extension {ext} doesn't match format {format}")
    elif format == DataFormat.CSV and ext != '.csv':
        raise ValueError(f"File extension {ext} doesn't match format {format}")
    elif format == DataFormat.FITS and ext not in ['.fits', '.fit']:
        raise ValueError(f"File extension {ext} doesn't match format {format}")
    
    # Export based on format
    if format == DataFormat.JSON:
        export_json(data, filepath)
    elif format == DataFormat.HDF5:
        export_hdf5(data, filepath)
    elif format == DataFormat.CSV:
        export_csv(data, filepath)
    elif format == DataFormat.FITS:
        export_fits(data, filepath)
    else:
        raise ValueError(f"Unsupported export format: {format}") 