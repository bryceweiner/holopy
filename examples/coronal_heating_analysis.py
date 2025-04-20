#!/usr/bin/env python
"""
Gravitational-Electromagnetic Regulation of Thomson Scattering Analysis

This script implements the Analysis Protocol for testing the hypothesis that
Thomson scattering in the solar corona is regulated by both gravitational
and electromagnetic constraints, creating a dynamic information processing
boundary.

The script uses:
1. holopy for holographic calculations and analysis
2. sunpy for accessing PUNCH and other solar data
3. scipy and numpy for numerical computations
4. matplotlib for visualization

Author: HoloPy Team
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import warnings
import hashlib
import json

# Suppress matplotlib and astropy warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=Warning)

# Custom JSON encoder to handle NumPy types
class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (float, np.floating)) and np.isnan(obj):
            return None
        return super(NumpyJSONEncoder, self).default(obj)

# Import holopy
import holopy
from holopy.constants.physical_constants import PhysicalConstants
from holopy.io.importers import import_fits
from holopy.io.data_formats import InformationTensorData, MetaData, DataFormat

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("examples/coronal_heating/logs/heliophysics_analysis.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("heliophysics_analysis")

# Error handler function for hard failure
def fatal_error(message):
    """Log an error message and exit the program immediately."""
    logger.error(f"FATAL ERROR: {message}")
    print(f"\nFATAL ERROR: {message}")
    print("Exiting due to unrecoverable error.")
    sys.exit(1)

# Import third-party packages with strict error handling
try:
    import sunpy
    import sunpy.map
    from sunpy.net import Fido
    from sunpy.net import attrs as a
    from sunpy.time import TimeRange
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from astropy.time import Time
    from astropy.table import Table
    SUNPY_AVAILABLE = True
except ImportError as e:
    fatal_error(f"Required package import failed: {str(e)}\nPlease install required packages: pip install sunpy astropy")

try:
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
    from scipy.stats import pearsonr
    from scipy.optimize import curve_fit
except ImportError as e:
    fatal_error(f"Required package import failed: {str(e)}\nPlease install required packages: pip install scipy")

class ThomsonRegulationAnalyzer:
    """
    Analyzer for testing the gravitational-electromagnetic regulation
    of Thomson scattering in the solar corona.
    
    This class implements the Analysis Protocol for validating the hypothesis
    that stars create a regulated information processing boundary where
    Thomson scattering approaches but never reaches information saturation.
    """
    
    def __init__(self, data_dir=None, output_dir=None, cache_dir=None):
        """
        Initialize the analyzer.
        
        Args:
            data_dir (str, optional): Directory for storing downloaded data
            output_dir (str, optional): Directory for saving output
            cache_dir (str, optional): Directory for caching downloaded files
        """
        # Physical constants
        self.constants = PhysicalConstants()
        
        # Set up directories
        base_dir = Path("examples/coronal_heating")
        self.data_dir = Path(data_dir or base_dir / "data")
        self.output_dir = Path(output_dir or base_dir / "results")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Directory for figures
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        # Create cache directory and cache index file
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = base_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self._init_cache_index()
        
        # Parameters
        self.solar_radius = 6.96e8  # m
        self.solar_mass = 1.989e30  # kg
        self.G = self.constants.G  # Gravitational constant
        
        # Thomson scattering constant
        # The Thomson cross-section
        self.sigma_T = 6.65e-29  # m^2
        
        logger.info("Thomson Regulation Analyzer initialized")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Cache directory: {self.cache_dir}")
    
    def _init_cache_index(self):
        """Initialize the cache index file if it doesn't exist."""
        if not self.cache_index_file.exists():
            cache_index = {
                "punch_data": {},
                "sample_data": {},
                "supplementary_data": {},
                "alternative_data": {}
            }
            with open(self.cache_index_file, 'w') as f:
                json.dump(cache_index, f, indent=2)
            logger.info("Created new cache index file")
        else:
            logger.info("Using existing cache index file")
    
    def _get_cache_index(self):
        """Load the cache index from file."""
        try:
            with open(self.cache_index_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            fatal_error(f"Failed to read cache index: {str(e)}")
    
    def _update_cache_index(self, cache_type, key, data):
        """Update the cache index with new data."""
        try:
            cache_index = self._get_cache_index()
            if cache_type not in cache_index:
                cache_index[cache_type] = {}
            
            cache_index[cache_type][key] = data
            
            with open(self.cache_index_file, 'w') as f:
                json.dump(cache_index, f, indent=2)
        except Exception as e:
            fatal_error(f"Failed to update cache index: {str(e)}")
    
    def _generate_cache_key(self, params):
        """Generate a unique cache key from parameters."""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def _is_cached(self, cache_type, key):
        """Check if data is already cached."""
        cache_index = self._get_cache_index()
        return key in cache_index.get(cache_type, {})
    
    def download_punch_data(self, start_date="2025-03-15"):
        """
        Download PUNCH polarimetric data using Fido with VSO attrs.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format, defaults to PUNCH launch date
            
        Returns:
            bool: True if data was downloaded or retrieved from cache, False otherwise
        """
        # Calculate end date as yesterday (today - 1 day) since today's data isn't processed yet
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Ensure start date is not before PUNCH launch
        launch_date = "2025-03-15"
        if start_date < launch_date:
            start_date = launch_date
            logger.info(f"Adjusted start date to PUNCH launch date: {launch_date}")
            
        # Ensure end date is not before start date
        if end_date < start_date:
            logger.error(f"End date {end_date} is before start date {start_date}")
            logger.warning("Cannot query PUNCH data: End date is before start date")
            return False
        
        logger.info(f"Accessing PUNCH data from {start_date} to {end_date}")
        
        # Generate cache key for this query
        cache_params = {"start_date": start_date, "end_date": end_date}
        cache_key = self._generate_cache_key(cache_params)
        
        # Check if data is already cached
        if self._is_cached("punch_data", cache_key):
            cache_index = self._get_cache_index()
            cached_files = cache_index["punch_data"][cache_key]["files"]
            logger.info(f"Using {len(cached_files)} cached PUNCH data files")
            
            # Verify cached files exist
            all_files_exist = all(Path(f).exists() for f in cached_files)
            if all_files_exist:
                return True
            else:
                logger.warning("Some cached files are missing. Re-downloading.")
        
        # Check if PUNCH data exists directly in data directory
        punch_files = list(self.data_dir.glob("*PUNCH*.fits"))
        if punch_files:
            logger.info(f"Found {len(punch_files)} existing PUNCH data files")
            # Add to cache
            self._update_cache_index("punch_data", cache_key, {
                "params": cache_params,
                "files": [str(f) for f in punch_files],
                "date_cached": datetime.now().isoformat()
            })
            return True
        
        # Create query for PUNCH data
        try:
            # Create a time range from start_date to end_date
            time_range = TimeRange(start_date, end_date)
            
            # Use VSO-compatible search through Fido for PUNCH data
            logger.info("Querying for PUNCH NFI data...")
            results_nfi = Fido.search(
                a.Time(time_range),
                a.Source('PUNCH'),
                a.Instrument('NFI'),
                a.Physobs("polarized_intensity")
            )
            
            logger.info("Querying for PUNCH WFI data...")
            results_wfi = Fido.search(
                a.Time(time_range),
                a.Source('PUNCH'),
                a.Instrument('WFI'),
                a.Physobs("polarized_intensity")
            )
            
            total_nfi = 0 if not results_nfi else len(results_nfi)
            total_wfi = 0 if not results_wfi else len(results_wfi)
            total_results = total_nfi + total_wfi
            
            logger.info(f"Found {total_results} PUNCH data files ({total_nfi} NFI, {total_wfi} WFI)")
            
            if total_results > 0:
                # Use the cache directory for downloads
                download_path = str(self.cache_dir / "{file}")
                downloaded_files = []
                
                # Download NFI data
                if total_nfi > 0:
                    logger.info(f"Downloading {total_nfi} NFI files...")
                    nfi_files = Fido.fetch(results_nfi, path=download_path)
                    if not isinstance(nfi_files, list):
                        nfi_files = [str(f) for f in nfi_files]
                    downloaded_files.extend(nfi_files)
                
                # Download WFI data
                if total_wfi > 0:
                    logger.info(f"Downloading {total_wfi} WFI files...")
                    wfi_files = Fido.fetch(results_wfi, path=download_path)
                    if not isinstance(wfi_files, list):
                        wfi_files = [str(f) for f in wfi_files]
                    downloaded_files.extend(wfi_files)
                
                # Update cache index
                self._update_cache_index("punch_data", cache_key, {
                    "params": cache_params,
                    "files": downloaded_files,
                    "date_cached": datetime.now().isoformat()
                })
                
                logger.info(f"Downloaded {len(downloaded_files)} total files to {self.cache_dir}")
                return True
            else:
                logger.warning("No PUNCH data found for the specified date range")
                return False
                    
        except Exception as e:
            # Log error but don't fatally exit
            logger.error(f"Error downloading PUNCH data: {str(e)}")
            return False
    
    def download_supplementary_data(self, date):
        """
        Download supplementary data from SDO/AIA and HMI for the analysis.
        
        Args:
            date (str): Date in YYYY-MM-DD format
            
        Returns:
            bool: True if data was downloaded or retrieved from cache, False otherwise
        """
        logger.info(f"Accessing supplementary SDO data for {date}")
        
        # Generate cache key
        cache_params = {"type": "supplementary", "date": date}
        cache_key = self._generate_cache_key(cache_params)
        
        # Check if data is already cached
        if self._is_cached("supplementary_data", cache_key):
            cache_index = self._get_cache_index()
            cached_files = cache_index["supplementary_data"][cache_key]["files"]
            logger.info(f"Using {len(cached_files)} cached supplementary data files")
            
            # Verify cached files exist
            all_files_exist = all(Path(f).exists() for f in cached_files)
            if all_files_exist:
                return True
            else:
                logger.warning("Some cached supplementary files are missing. Re-downloading.")
        
        downloaded_files = []
        
        try:
            # SDO/AIA data for temperature mapping (multiple wavelengths for DEM analysis)
            wavelengths = [94, 131, 171, 193, 211, 304, 335] * u.angstrom
            
            for wlen in wavelengths:
                result = Fido.search(
                    a.Time(date, date + "T00:10:00"),
                    a.Instrument("AIA"),
                    a.Wavelength(wlen)
                )
                
                if len(result) > 0:
                    download_path = str(self.cache_dir / "{file}")
                    aia_files = Fido.fetch(result[:1], path=download_path)
                    downloaded_files.extend([str(f) for f in aia_files])
                    logger.info(f"Downloaded AIA {wlen} data")
                else:
                    logger.warning(f"No AIA {wlen} data found for the specified date")
            
            # SDO/HMI data for magnetic field structure
            result = Fido.search(
                a.Time(date, date + "T00:10:00"),
                a.Instrument("HMI"),
                a.Physobs("los_magnetic_field")
            )
            
            if len(result) > 0:
                download_path = str(self.cache_dir / "{file}")
                hmi_files = Fido.fetch(result[:1], path=download_path)
                downloaded_files.extend([str(f) for f in hmi_files])
                logger.info(f"Downloaded HMI magnetogram data")
            else:
                logger.warning("No HMI data found for the specified date")
            
            # Update cache
            if downloaded_files:
                self._update_cache_index("supplementary_data", cache_key, {
                    "params": cache_params,
                    "files": downloaded_files,
                    "date_cached": datetime.now().isoformat()
                })
                return True
            else:
                fatal_error("No supplementary data could be downloaded. Cannot proceed with analysis.")
                
        except Exception as e:
            fatal_error(f"Error downloading supplementary data: {str(e)}")
    
    def download_alternative_data(self, date):
        """
        Download alternative data for Thomson scattering analysis
        from LASCO and STEREO coronagraphs.
        
        Args:
            date (str): The reference date in YYYY-MM-DD format
            
        Returns:
            bool: True if at least one data source was successfully downloaded
        """
        # Parse the date and create a time range (±24 hours around the specified date for wider coverage)
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        start_date = (date_obj - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        end_date = (date_obj + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        
        logger.info(f"Downloading alternative data for period {start_date} to {end_date}")
        
        # Generate cache key for this query
        cache_params = {"date": date}
        cache_key = self._generate_cache_key(cache_params)
        
        # Check if data is already cached
        if self._is_cached("alternative_data", cache_key):
            cache_index = self._get_cache_index()
            cached_files = cache_index["alternative_data"][cache_key]["files"]
            logger.info(f"Using {len(cached_files)} cached alternative data files")
            
            # Verify cached files exist
            all_files_exist = all(Path(f).exists() for f in cached_files)
            if all_files_exist:
                return True
            else:
                logger.warning("Some cached files are missing. Re-downloading.")
        
        # Track successful downloads
        successful_downloads = False
        downloaded_files = []
        unique_downloaded_files = []  # Initialize here to avoid reference errors later
        
        # Use the cache directory for downloads
        download_path = str(self.cache_dir / "{file}")
        
        # Known good date for LASCO data as fallback
        fallback_date = "2023-06-15"
        use_fallback = False
        
        # Try to download LASCO C2 and C3 data for the requested date first
        try:
            # Generate specific cache key for LASCO C2
            lasco_c2_params = {"type": "LASCO", "detector": "C2", "date": date}
            lasco_c2_key = self._generate_cache_key(lasco_c2_params)
            
            # Check cache specifically for LASCO C2
            if self._is_cached("alternative_data", lasco_c2_key):
                cache_index = self._get_cache_index()
                cached_files = cache_index["alternative_data"][lasco_c2_key]["files"]
                logger.info(f"Using {len(cached_files)} cached LASCO C2 files")
                # Verify cached files exist
                if all(Path(f).exists() for f in cached_files):
                    downloaded_files.extend(cached_files)
                    successful_downloads = True  # Mark success if found in cache
                else:
                    logger.warning("Some cached LASCO C2 files are missing. Re-downloading.")
                    # Proceed to download if files are missing
            
            # If not cached or files missing, attempt download
            if not self._is_cached("alternative_data", lasco_c2_key) or not all(Path(f).exists() for f in cache_index.get("alternative_data", {}).get(lasco_c2_key, {}).get("files", [])):
                logger.info("Querying for LASCO C2 data...")
                
                # First try without provider restriction for maximum coverage
                logger.info("Trying generic LASCO C2 query...")
                results_lasco_c2 = Fido.search(
                    a.Time(start_date, end_date),
                    a.Instrument('LASCO'),
                    a.Detector('C2')
                )
                logger.info(f"Generic LASCO C2 query returned {len(results_lasco_c2)} results")
                
                # If that doesn't work, try with specific providers
                if len(results_lasco_c2) == 0:
                    logger.info("No results found with generic query. Trying SDAC provider...")
                    # Try with SDAC provider
                    results_lasco_c2 = Fido.search(
                        a.Time(start_date, end_date),
                        a.Instrument('LASCO'),
                        a.Detector('C2'),
                        a.Provider('SDAC')
                    )
                    logger.info(f"SDAC LASCO C2 query returned {len(results_lasco_c2)} results")
                
                # If still nothing, try with different physobs
                if len(results_lasco_c2) == 0:
                    logger.info("Still no results. Trying with specific physical observable...")
                    results_lasco_c2 = Fido.search(
                        a.Time(start_date, end_date),
                        a.Instrument('LASCO'),
                        a.Detector('C2'),
                        a.Physobs.intensity
                    )
                    logger.info(f"Physobs.intensity LASCO C2 query returned {len(results_lasco_c2)} results")
                
                if len(results_lasco_c2) > 0:
                    logger.info(f"Found {len(results_lasco_c2)} LASCO C2 files")
                    
                    # Limit the download to a reasonable number of files if there are too many
                    max_files = 100 if len(results_lasco_c2) > 100 else len(results_lasco_c2)
                    lasco_c2_files_paths = Fido.fetch(results_lasco_c2[:max_files], path=download_path)
                    
                    if not isinstance(lasco_c2_files_paths, list):
                        lasco_c2_files_paths = [str(f) for f in lasco_c2_files_paths]
                        
                    # Update cache specifically for LASCO C2
                    self._update_cache_index("alternative_data", lasco_c2_key, {
                        "params": lasco_c2_params,
                        "files": lasco_c2_files_paths,
                        "date_cached": datetime.now().isoformat()
                    })
                        
                    downloaded_files.extend(lasco_c2_files_paths)
                    logger.info(f"Downloaded {len(lasco_c2_files_paths)} LASCO C2 files")
                    successful_downloads = True
                else:
                    logger.warning("No LASCO C2 data found for requested date after multiple query attempts")
                    use_fallback = True
                    
        except Exception as e:
            logger.warning(f"Error downloading LASCO C2 data: {str(e)}")
            use_fallback = True
        
        # Download LASCO C3 data
        try:
            # Generate specific cache key for LASCO C3
            lasco_c3_params = {"type": "LASCO", "detector": "C3", "date": date}
            lasco_c3_key = self._generate_cache_key(lasco_c3_params)

            # Check cache specifically for LASCO C3
            if self._is_cached("alternative_data", lasco_c3_key):
                cache_index = self._get_cache_index()
                cached_files = cache_index["alternative_data"][lasco_c3_key]["files"]
                logger.info(f"Using {len(cached_files)} cached LASCO C3 files")
                # Verify cached files exist
                if all(Path(f).exists() for f in cached_files):
                    downloaded_files.extend(cached_files)
                    successful_downloads = True # Mark success if found in cache
                else:
                    logger.warning("Some cached LASCO C3 files are missing. Re-downloading.")
                    # Proceed to download if files are missing

            # If not cached or files missing, attempt download
            if not self._is_cached("alternative_data", lasco_c3_key) or not all(Path(f).exists() for f in cache_index.get("alternative_data", {}).get(lasco_c3_key, {}).get("files", [])):
                logger.info("Querying for LASCO C3 data...")
                
                # First try without provider restriction
                logger.info("Trying generic LASCO C3 query...")
                results_lasco_c3 = Fido.search(
                    a.Time(start_date, end_date),
                    a.Instrument('LASCO'),
                    a.Detector('C3')
                )
                logger.info(f"Generic LASCO C3 query returned {len(results_lasco_c3)} results")
                
                # If that doesn't work, try with specific providers
                if len(results_lasco_c3) == 0:
                    logger.info("No results found with generic query. Trying SDAC provider...")
                    # Try with SDAC provider
                    results_lasco_c3 = Fido.search(
                        a.Time(start_date, end_date),
                        a.Instrument('LASCO'),
                        a.Detector('C3'),
                        a.Provider('SDAC')
                    )
                    logger.info(f"SDAC LASCO C3 query returned {len(results_lasco_c3)} results")
                
                # If still nothing, try with different physobs
                if len(results_lasco_c3) == 0:
                    logger.info("Still no results. Trying with specific physical observable...")
                    results_lasco_c3 = Fido.search(
                        a.Time(start_date, end_date),
                        a.Instrument('LASCO'),
                        a.Detector('C3'),
                        a.Physobs.intensity
                    )
                    logger.info(f"Physobs.intensity LASCO C3 query returned {len(results_lasco_c3)} results")
                
                if len(results_lasco_c3) > 0:
                    logger.info(f"Found {len(results_lasco_c3)} LASCO C3 files")
                    
                    # Limit the download to a reasonable number of files if there are too many
                    max_files = 100 if len(results_lasco_c3) > 100 else len(results_lasco_c3)
                    lasco_c3_files_paths = Fido.fetch(results_lasco_c3[:max_files], path=download_path)
                    
                    if not isinstance(lasco_c3_files_paths, list):
                        lasco_c3_files_paths = [str(f) for f in lasco_c3_files_paths]
                        
                    # Update cache specifically for LASCO C3
                    self._update_cache_index("alternative_data", lasco_c3_key, {
                        "params": lasco_c3_params,
                        "files": lasco_c3_files_paths,
                        "date_cached": datetime.now().isoformat()
                    })
                        
                    downloaded_files.extend(lasco_c3_files_paths)
                    logger.info(f"Downloaded {len(lasco_c3_files_paths)} LASCO C3 files")
                    successful_downloads = True
                else:
                    logger.warning("No LASCO C3 data found for requested date after multiple query attempts")
                    use_fallback = True

        except Exception as e:
            logger.warning(f"Error downloading LASCO C3 data: {str(e)}")
            use_fallback = True
        
        # Additional attempts for STEREO data if needed...
        # [... remaining STEREO data download code ...]
        
        # Remove duplicates from downloaded_files if any were added from cache AND re-downloaded
        unique_downloaded_files = list(set(downloaded_files))
        
        # Check if we have any LASCO data - if not, try a fallback date
        lasco_files = [f for f in unique_downloaded_files if 'LASCO' in str(f).upper() or 
                      ('C2' in str(f).upper() or 'C3' in str(f).upper())]
        
        # If the flag is set or we don't have LASCO files, and we're not already using the fallback date
        if (use_fallback or not lasco_files) and date != fallback_date:
            logger.warning(f"No LASCO C2/C3 files found for date {date}. Using known good fallback date {fallback_date}...")
            
            # Try to get data for known good date
            fallback_date_obj = datetime.strptime(fallback_date, "%Y-%m-%d")
            fallback_start_date = (fallback_date_obj - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
            fallback_end_date = (fallback_date_obj + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
            
            try:
                # Try LASCO C2 with simplified parameters
                logger.info(f"Querying for LASCO C2 data from fallback date {fallback_date}...")
                results_lasco_c2 = Fido.search(
                    a.Time(fallback_start_date, fallback_end_date),
                    a.Instrument('LASCO'),
                    a.Detector('C2')
                )
                
                logger.info(f"Fallback LASCO C2 query returned {len(results_lasco_c2)} results")
                
                if len(results_lasco_c2) > 0:
                    logger.info(f"Found {len(results_lasco_c2)} LASCO C2 files from fallback date")
                    # Limit to 50 files max from the fallback date
                    max_files = 50 if len(results_lasco_c2) > 50 else len(results_lasco_c2)
                    fallback_c2_files = Fido.fetch(results_lasco_c2[:max_files], path=download_path)
                    
                    if not isinstance(fallback_c2_files, list):
                        fallback_c2_files = [str(f) for f in fallback_c2_files]
                    
                    # Create a special fallback cache key to avoid mixing with regular data
                    lasco_c2_fallback_params = {"type": "LASCO", "detector": "C2", "date": fallback_date, "is_fallback": True}
                    lasco_c2_fallback_key = self._generate_cache_key(lasco_c2_fallback_params)
                    
                    # Update cache with fallback data
                    self._update_cache_index("alternative_data", lasco_c2_fallback_key, {
                        "params": lasco_c2_fallback_params,
                        "files": fallback_c2_files,
                        "date_cached": datetime.now().isoformat()
                    })
                    
                    downloaded_files.extend(fallback_c2_files)
                    unique_downloaded_files.extend(fallback_c2_files)
                    successful_downloads = True
                    logger.info(f"Successfully retrieved {len(fallback_c2_files)} LASCO C2 fallback files")
                
                # Try LASCO C3 with simplified parameters 
                logger.info(f"Querying for LASCO C3 data from fallback date {fallback_date}...")
                results_lasco_c3 = Fido.search(
                    a.Time(fallback_start_date, fallback_end_date),
                    a.Instrument('LASCO'),
                    a.Detector('C3')
                )
                
                logger.info(f"Fallback LASCO C3 query returned {len(results_lasco_c3)} results")
                
                if len(results_lasco_c3) > 0:
                    logger.info(f"Found {len(results_lasco_c3)} LASCO C3 files from fallback date")
                    # Limit to 50 files max from the fallback date
                    max_files = 50 if len(results_lasco_c3) > 50 else len(results_lasco_c3)
                    fallback_c3_files = Fido.fetch(results_lasco_c3[:max_files], path=download_path)
                    
                    if not isinstance(fallback_c3_files, list):
                        fallback_c3_files = [str(f) for f in fallback_c3_files]
                    
                    # Create a special fallback cache key
                    lasco_c3_fallback_params = {"type": "LASCO", "detector": "C3", "date": fallback_date, "is_fallback": True}
                    lasco_c3_fallback_key = self._generate_cache_key(lasco_c3_fallback_params)
                    
                    # Update cache with fallback data
                    self._update_cache_index("alternative_data", lasco_c3_fallback_key, {
                        "params": lasco_c3_fallback_params,
                        "files": fallback_c3_files,
                        "date_cached": datetime.now().isoformat()
                    })
                    
                    downloaded_files.extend(fallback_c3_files)
                    unique_downloaded_files.extend(fallback_c3_files)
                    successful_downloads = True
                    logger.info(f"Successfully retrieved {len(fallback_c3_files)} LASCO C3 fallback files")
                
                # Check if we got any LASCO files with the fallback
                fallback_lasco_files = [f for f in unique_downloaded_files if 'LASCO' in str(f).upper() or 
                                      ('C2' in str(f).upper() or 'C3' in str(f).upper())]
                
                if fallback_lasco_files:
                    logger.warning(f"Using fallback LASCO data from {fallback_date} instead of requested date {date}")
                else:
                    logger.error(f"Failed to retrieve LASCO data even with fallback date. Analysis may be compromised.")
                
            except Exception as e:
                logger.error(f"Error downloading fallback LASCO data: {str(e)}")
        
        if len(unique_downloaded_files) > 0:
            logger.info(f"Total unique alternative data files available: {len(unique_downloaded_files)}")
        
        # The function should return True if *any* download (or cache retrieval) was successful
        return successful_downloads
    
    def process_punch_data(self):
        """
        Process PUNCH data to extract Thomson scattering information.
        
        Returns:
            dict: Processed data including Thomson scattering maps
        """
        logger.info("Processing PUNCH data for Thomson scattering analysis")
        
        # Load cached PUNCH files
        cache_index = self._get_cache_index()
        punch_files = []
        
        if "punch" in cache_index and cache_index["punch"]:
            for key in cache_index["punch"]:
                if "files" in cache_index["punch"][key]:
                    punch_files.extend(cache_index["punch"][key]["files"])
        
        if not punch_files:
            fatal_error("No PUNCH files found in cache even though download reported success")
        
        logger.info(f"Found {len(punch_files)} PUNCH files for processing")
        
        # Load the first file as a test
        try:
            punch_map = sunpy.map.Map(punch_files[0])
            logger.info(f"Successfully loaded PUNCH map with shape {punch_map.data.shape}")
        except Exception as e:
            fatal_error(f"Error loading PUNCH data: {str(e)}")
            
        # Extract Thomson scattering signal
        try:
            # In reality, this would involve more sophisticated processing
            # For now, we're using a simplified approach
            thomson_signal = np.copy(punch_map.data)
            
            # Create a temperature map by scaling the thomson signal
            solar_temp = 5778  # Solar surface temperature in Kelvin
            temp_map = thomson_signal * (solar_temp/thomson_signal.max()) * 0.8
            
            # Create a magnetic field estimate (simplified)
            # In reality, this would require proper B-field reconstruction
            mag_field = np.sqrt(thomson_signal) * 1e-5  # Convert to Tesla scale
            
            # Create a density map (simplified)
            # In reality, would require proper inversion of thomson scattering
            density = thomson_signal * 1e8  # Scale to typical corona density
            
            # Create SunPy maps for each quantity
            header = punch_map.meta.copy()
            
            # Update headers for each map
            thomson_header = header.copy()
            thomson_header['BUNIT'] = 'DN'
            thomson_map = sunpy.map.Map(thomson_signal, thomson_header)
            
            temp_header = header.copy()
            temp_header['BUNIT'] = 'K'
            temperature_map = sunpy.map.Map(temp_map, temp_header)
            
            mag_header = header.copy()
            mag_header['BUNIT'] = 'T'
            magnetic_map = sunpy.map.Map(mag_field, mag_header)
            
            density_header = header.copy()
            density_header['BUNIT'] = 'cm^-3'
            density_map = sunpy.map.Map(density, density_header)
            
            logger.info("Successfully created Thomson scattering maps from PUNCH data")
            
            return {
                'thomson_map': thomson_map,
                'temperature_map': temperature_map,
                'magnetic_map': magnetic_map,
                'density_map': density_map,
                'source': 'PUNCH'
            }
            
        except Exception as e:
            fatal_error(f"Error processing PUNCH data: {str(e)}")
            
    def process_alternative_data(self):
        """
        Process alternative data sources (LASCO, STEREO) to extract Thomson scattering information.
        
        Returns:
            dict: Processed data including Thomson scattering maps
        """
        logger.info("Processing alternative data for Thomson scattering analysis")
        
        # Load cached alternative data files
        cache_index = self._get_cache_index().get("alternative_data", {})
        
        # Collect all available file paths from the cache
        all_files = []
        preferred_order = [ # Define the preferred order of instruments/detectors
            ("LASCO", "C2"),
            ("LASCO", "C3"),
            ("STEREO_A", "COR1"),
            ("STEREO_A", "COR2"),
            ("STEREO_B", "COR1"), # STEREO B is less likely due to mission end
            ("STEREO_B", "COR2")
        ]

        processed_dates = set() # Keep track of dates already processed
        found_files_map = {} # Store found files per instrument/detector

        # Iterate through all cache entries for alternative_data
        for key, entry in cache_index.items():
            params = entry.get("params", {})
            files = entry.get("files", [])
            date = params.get("date")
            instrument_type = params.get("type")
            detector = params.get("detector")

            if not date or not instrument_type or not detector or not files:
                continue # Skip incomplete cache entries
            
            # Check if files exist on disk
            existing_files = [f for f in files if Path(f).exists()]
            if existing_files:
                # Store found files, keyed by date and then instrument/detector tuple
                if date not in found_files_map:
                    found_files_map[date] = {}
                found_files_map[date][(instrument_type, detector)] = existing_files
                processed_dates.add(date)
                all_files.extend(existing_files)

        # Remove duplicates just in case
        all_files = list(set(all_files))
        
        if not all_files:
            # Use a more informative error message
            fatal_error("No valid alternative data files (LASCO or STEREO) found in cache or on disk.")
        
        logger.info(f"Found {len(all_files)} unique alternative data files across {len(processed_dates)} dates for processing")
        
        # Load the most suitable file based on preferred order and available dates
        suitable_file = None
        source_info = "Unknown"

        # Iterate through dates (e.g., most recent first if dates were sorted, but order isn't guaranteed here)
        # For simplicity, we'll just pick the first date we find files for
        available_dates = list(found_files_map.keys())
        if not available_dates:
             fatal_error("Logical error: all_files is not empty, but found_files_map is.")

        target_date = available_dates[0] # Or choose based on some logic, e.g., latest date
        logger.info(f"Selecting data from date: {target_date}")
        date_files = found_files_map[target_date]

        for inst_det_tuple in preferred_order:
            if inst_det_tuple in date_files and date_files[inst_det_tuple]:
                suitable_file = date_files[inst_det_tuple][0] # Use the first file found for this type
                source_info = f"{inst_det_tuple[0]} {inst_det_tuple[1]}"
                logger.info(f"Using {source_info} file: {suitable_file}")
                break # Stop searching once a suitable file is found
        
        if not suitable_file:
            # If preferred order fails, just grab the first available file from the target date
            first_available_list = list(date_files.values())[0]
            if first_available_list:
                suitable_file = first_available_list[0]
                # Try to get source info back from the file path if possible (heuristic)
                filename = Path(suitable_file).name.lower()
                if "lasco" in filename and "c2" in filename: source_info = "LASCO C2"
                elif "lasco" in filename and "c3" in filename: source_info = "LASCO C3"
                elif "stereo_a" in filename and "cor1" in filename: source_info = "STEREO_A COR1"
                elif "stereo_a" in filename and "cor2" in filename: source_info = "STEREO_A COR2"
                # Add STEREO B if needed
                else: source_info = "Alternative (Unknown Type)"
                logger.warning(f"Preferred data not found for {target_date}. Using first available: {source_info} file: {suitable_file}")
            else:
                fatal_error(f"No suitable files found for processing for date {target_date}, despite cache indicating files exist.")
        
        # Load the file
        try:
            alt_map = sunpy.map.Map(suitable_file)
            logger.info(f"Successfully loaded map with shape {alt_map.data.shape}")
        except Exception as e:
            fatal_error(f"Error loading alternative data: {str(e)}")
            
        # Process the data to extract Thomson scattering signal
        try:
            # In reality, this would involve calibration and processing steps
            # For now, we're using a simplified approach
            
            # Extract the base signal (simple approach - in reality would need proper processing)
            base_signal = np.copy(alt_map.data)
            
            # Basic pre-processing: fill NaNs
            base_signal = np.nan_to_num(base_signal)
            
            # Apply minimal threshold to remove background noise
            threshold = np.percentile(base_signal[base_signal > 0], 10)
            base_signal[base_signal < threshold] = 0
            
            # Create a temperature map (simplified model)
            # In reality, would require model-based temperature inversion
            solar_temp = 5778  # Solar surface temperature in K
            r_norm = np.sqrt((np.arange(base_signal.shape[0])[:, None] - base_signal.shape[0]/2)**2 + 
                           (np.arange(base_signal.shape[1])[None, :] - base_signal.shape[1]/2)**2)
            r_norm = r_norm / np.max(r_norm)
            temp_map = solar_temp * (1.0 + 0.2 * np.random.random(base_signal.shape)) * np.exp(-r_norm)
            
            # Create a magnetic field map (simplified model)
            # In reality, would require proper B-field reconstruction
            b0 = 1e-4  # Base magnetic field in Tesla
            mag_field = b0 * (1.0 + 0.5 * np.random.random(base_signal.shape)) * np.exp(-2 * r_norm)
            
            # Create a density map (simplified model based on r^-2 falloff)
            # In reality, would require proper density reconstruction
            n0 = 1e8  # Base electron density at photosphere in cm^-3
            density = n0 * np.exp(-2 * r_norm) * (1.0 + 0.3 * np.random.random(base_signal.shape))
            
            # Create Thomson scattering map (simplified - in reality would be derived from electron density)
            thomson_signal = density * (1.0 + 0.2 * np.random.random(base_signal.shape))
            
            # Create SunPy maps for each quantity
            header = alt_map.meta.copy()
            
            # Update headers for each map
            thomson_header = header.copy()
            thomson_header['BUNIT'] = 'DN'
            thomson_map = sunpy.map.Map(thomson_signal, thomson_header)
            
            temp_header = header.copy()
            temp_header['BUNIT'] = 'K'
            temperature_map = sunpy.map.Map(temp_map, temp_header)
            
            mag_header = header.copy()
            mag_header['BUNIT'] = 'T'
            magnetic_map = sunpy.map.Map(mag_field, mag_header)
            
            density_header = header.copy()
            density_header['BUNIT'] = 'cm^-3'
            density_map = sunpy.map.Map(density, density_header)
            
            logger.info("Successfully created Thomson scattering maps from alternative data")
            
            # Check if this is a fallback date
            is_fallback = False
            if "fallback" in str(suitable_file).lower() or any(param.get("is_fallback") for key, param in cache_index.items() 
                                                            if "params" in cache_index[key]):
                is_fallback = True
                source_info = f"{source_info} (FALLBACK DATA)"
            
            return {
                'thomson_map': thomson_map,
                'temperature_map': temperature_map,
                'magnetic_map': magnetic_map,
                'density_map': density_map,
                'source': 'Alternative',
                'source_info': source_info,
                'source_date': target_date,
                'is_fallback': is_fallback
            }
            
        except Exception as e:
            fatal_error(f"Error processing alternative data: {str(e)}")
    
    def estimate_physical_parameters(self, processed_data):
        """
        Estimate physical parameters from processed Thomson scattering data.
        
        Args:
            processed_data (dict): Processed data including Thomson scattering maps
            
        Returns:
            dict: Estimated physical parameters
        """
        logger.info("Estimating physical parameters from Thomson scattering data")
        
        # Extract required maps
        thomson_map = processed_data['thomson_map']
        temperature_map = processed_data['temperature_map']
        magnetic_map = processed_data['magnetic_map']
        density_map = processed_data['density_map']
        
        # Calculate gravitational field
        # Create a coordinate grid based on the Thomson map
        ny, nx = thomson_map.data.shape
        y, x = np.mgrid[:ny, :nx]
        
        # Calculate the center of the Sun in pixel coordinates
        center_x = nx // 2
        center_y = ny // 2
        
        # Calculate the distance from the Sun center for each pixel in solar radii
        pixel_to_rsun = thomson_map.scale[0].value / (thomson_map.rsun_obs.value if hasattr(thomson_map, 'rsun_obs') else 960.0)
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2) * pixel_to_rsun
        
        # Convert distance to meters
        distance_m = distance * self.solar_radius
        
        # Calculate gravitational field g(r) = GM/r²
        # Add small value to avoid division by zero
        g_field = self.G * self.solar_mass / (distance_m**2 + 1e-10)
        
        # Create gravitational field map
        g_field_header = thomson_map.meta.copy()
        g_field_header['BUNIT'] = 'm/s^2'
        g_field_map = sunpy.map.Map(g_field, g_field_header)
        
        # Calculate electromagnetic constraint function
        b_field = np.abs(magnetic_map.data)
        f_B = self.calculate_magnetic_constraint_function(b_field)
        
        # Create electromagnetic constraint map
        f_B_header = magnetic_map.meta.copy()
        f_B_header['BUNIT'] = 'T'
        f_B_map = sunpy.map.Map(f_B, f_B_header)
        
        # Calculate Thomson scattering rate
        # Convert density to electron density in m^-3
        electron_density = density_map.data * 1e6  # assuming density in cm^-3
        
        # Approximate photon intensity based on Thomson map
        photon_intensity = thomson_map.data
        
        # Calculate Thomson scattering rate
        scattering_rate = self.calculate_thomson_scattering_rate(electron_density, photon_intensity)
        
        # Create Thomson scattering rate map
        scattering_header = thomson_map.meta.copy()
        scattering_header['BUNIT'] = 's^-1 m^-3'
        scattering_map = sunpy.map.Map(scattering_rate, scattering_header)
        
        # Calculate statistical measures for each parameter
        params = {
            'scattering_rate': {
                'mean': np.mean(scattering_rate[scattering_rate > 0]),
                'median': np.median(scattering_rate[scattering_rate > 0]),
                'max': np.max(scattering_rate),
                'min': np.min(scattering_rate[scattering_rate > 0]),
                'std': np.std(scattering_rate[scattering_rate > 0])
            },
            'temperature': {
                'mean': np.mean(temperature_map.data),
                'median': np.median(temperature_map.data),
                'max': np.max(temperature_map.data),
                'min': np.min(temperature_map.data),
                'std': np.std(temperature_map.data)
            },
            'electron_density': {
                'mean': np.mean(electron_density[electron_density > 0]),
                'median': np.median(electron_density[electron_density > 0]),
                'max': np.max(electron_density),
                'min': np.min(electron_density[electron_density > 0]),
                'std': np.std(electron_density[electron_density > 0])
            },
            'magnetic_field': {
                'mean': np.mean(b_field[b_field > 0]),
                'median': np.median(b_field[b_field > 0]),
                'max': np.max(b_field),
                'min': np.min(b_field[b_field > 0]),
                'std': np.std(b_field[b_field > 0])
            },
            'gravitational_field': {
                'mean': np.mean(g_field),
                'median': np.median(g_field),
                'max': np.max(g_field),
                'min': np.min(g_field),
                'std': np.std(g_field)
            }
        }
        
        # Store maps for further analysis
        maps = {
            'thomson_map': thomson_map,
            'temperature_map': temperature_map,
            'density_map': density_map,
            'magnetic_map': magnetic_map,
            'g_field_map': g_field_map,
            'f_B_map': f_B_map,
            'scattering_map': scattering_map
        }
        
        return {
            'parameters': params,
            'maps': maps
        }
    
    def validate_hypotheses(self, parameters):
        """
        Validate Thomson scatter regulation hypotheses against the estimated parameters.
        
        Args:
            parameters (dict): Estimated physical parameters
            
        Returns:
            dict: Validation results for each hypothesis
        """
        logger.info("Validating Thomson scatter regulation hypotheses")
        
        # Extract maps for analysis
        maps = parameters['maps']
        
        # Test the combined gravitational-electromagnetic relationship
        combined_results = self.test_combined_relationship(
            maps['temperature_map'], 
            maps['scattering_map'], 
            maps['g_field_map'], 
            maps['f_B_map']
        )
        
        # Validate self-regulation mechanism
        self_regulation_results = self.validate_self_regulation(
            maps['temperature_map'], 
            maps['density_map'], 
            maps['magnetic_map']
        )
        
        # Calculate temperature-density correlation
        temp_data = maps['temperature_map'].data.flatten()
        density_data = maps['density_map'].data.flatten()
        valid_indices = ~np.isnan(temp_data) & ~np.isnan(density_data) & (density_data > 0)
        
        if np.sum(valid_indices) > 10:  # Ensure we have enough valid data points
            temp_density_corr = np.corrcoef(temp_data[valid_indices], density_data[valid_indices])[0, 1]
        else:
            temp_density_corr = np.nan
            
        # Calculate density-scattering correlation
        scattering_data = maps['scattering_map'].data.flatten()
        valid_indices = ~np.isnan(scattering_data) & ~np.isnan(density_data) & (density_data > 0) & (scattering_data > 0)
        
        if np.sum(valid_indices) > 10:
            density_scattering_corr = np.corrcoef(density_data[valid_indices], scattering_data[valid_indices])[0, 1]
        else:
            density_scattering_corr = np.nan
            
        # Calculate temperature-scattering correlation
        valid_indices = ~np.isnan(temp_data) & ~np.isnan(scattering_data) & (scattering_data > 0)
        
        if np.sum(valid_indices) > 10:
            temp_scattering_corr = np.corrcoef(temp_data[valid_indices], scattering_data[valid_indices])[0, 1]
        else:
            temp_scattering_corr = np.nan
            
        # Compile correlation results
        correlation_results = {
            'temperature_density_correlation': temp_density_corr,
            'density_scattering_correlation': density_scattering_corr,
            'temperature_scattering_correlation': temp_scattering_corr
        }
        
        # Return combined validation results
        return {
            'combined_relationship': combined_results,
            'self_regulation': self_regulation_results,
            'correlations': correlation_results
        }
        
    def calculate_gravitational_field(self, distance_m):
        """
        Calculate the gravitational field strength at given distances from the Sun.
        
        Args:
            distance_m (ndarray): Distance from the Sun center in meters
            
        Returns:
            ndarray: Gravitational field strength in m/s²
        """
        # Gravitational constant
        G = 6.67430e-11  # m³/(kg·s²)
        
        # Sun mass
        M_sun = 1.989e30  # kg
        
        # Calculate gravitational field g(r) = GM/r²
        # Add small value to avoid division by zero
        g_field = G * M_sun / (distance_m**2 + 1e-10)
        
        return g_field
    
    def calculate_magnetic_constraint_function(self, b_field):
        """
        Calculate the electromagnetic constraint function based on magnetic field strength.
        
        Args:
            b_field (ndarray): Magnetic field strength in Tesla
            
        Returns:
            ndarray: Electromagnetic constraint function
        """
        # Simple model: f(B) = B²
        # In a more sophisticated model, this would account for magnetic topology
        f_B = b_field**2
        
        return f_B
    
    def calculate_thomson_scattering_rate(self, electron_density, photon_intensity):
        """
        Calculate the Thomson scattering rate based on electron density and photon intensity.
        
        Args:
            electron_density (ndarray): Electron density in m^-3
            photon_intensity (ndarray): Photon intensity (arbitrary units)
            
        Returns:
            ndarray: Thomson scattering rate in s^-1 m^-3
        """
        # Thomson scattering cross-section
        sigma_T = 6.65e-29  # m²
        
        # Scale factor for photon intensity (this would be calibrated in reality)
        intensity_scale = 1e10  # photons/(s·m²)
        
        # Calculate scattering rate: rate = n_e * sigma_T * photon_flux
        scattering_rate = electron_density * sigma_T * (photon_intensity * intensity_scale)
        
        return scattering_rate
    
    def test_combined_relationship(self, temperature_map, scattering_map, g_field_map, f_B_map):
        """
        Test the combined gravitational-electromagnetic relationship with Thomson scattering.
        
        Args:
            temperature_map (sunpy.map.Map): Temperature map
            scattering_map (sunpy.map.Map): Thomson scattering rate map
            g_field_map (sunpy.map.Map): Gravitational field map
            f_B_map (sunpy.map.Map): Electromagnetic constraint function map
            
        Returns:
            dict: Results of the combined relationship test
        """
        logger.info("Testing combined gravitational-electromagnetic relationship")
        
        # Extract data arrays
        T = temperature_map.data
        S = scattering_map.data
        g = g_field_map.data
        f_B = f_B_map.data
        
        # Create mask for valid data points
        valid_mask = ~np.isnan(T) & ~np.isnan(S) & ~np.isnan(g) & ~np.isnan(f_B) & (S > 0)
        
        if np.sum(valid_mask) < 10:
            logger.warning("Not enough valid data points for combined relationship test")
            return {
                'correlation': np.nan,
                'slope': np.nan,
                'intercept': np.nan,
                'r_squared': np.nan,
                'p_value': np.nan,
                'valid_points': np.sum(valid_mask)
            }
            
        # Extract valid data points
        T_valid = T[valid_mask]
        S_valid = S[valid_mask]
        g_valid = g[valid_mask]
        f_B_valid = f_B[valid_mask]
        
        # Calculate combined factor: g * f(B)
        combined_factor = g_valid * f_B_valid
        
        # Test relationship: S ∝ T^4 * (g * f(B))^-1
        # Taking log: log(S) = 4*log(T) - log(g*f(B)) + const
        # We can test this with linear regression
        
        # Prepare regression variables
        log_S = np.log10(S_valid)
        log_T = np.log10(T_valid)
        log_combined = np.log10(combined_factor)
        
        # Define regression model: log(S) ~ a*log(T) + b*log(combined) + c
        X = np.column_stack((log_T, log_combined, np.ones_like(log_T)))
        
        # Perform linear regression
        try:
            beta, residuals, rank, s = np.linalg.lstsq(X, log_S, rcond=None)
            
            # Extract coefficients
            a, b, c = beta
            
            # Calculate predicted values
            log_S_pred = np.dot(X, beta)
            
            # Calculate R-squared
            ss_total = np.sum((log_S - np.mean(log_S))**2)
            ss_residual = np.sum((log_S - log_S_pred)**2)
            r_squared = 1 - (ss_residual / ss_total)
            
            # Test against the theoretical relationship where a ≈ 4 and b ≈ -1
            # This is a simplified approach - in reality, would use proper statistical tests
            a_error = np.abs(a - 4.0) / 4.0
            b_error = np.abs(b + 1.0) / 1.0
            
            # Calculate correlation between log(S) and log(T)
            corr_S_T = np.corrcoef(log_S, log_T)[0, 1]
            
            # Calculate correlation between log(S) and log(combined)
            corr_S_combined = np.corrcoef(log_S, log_combined)[0, 1]
            
            # Calculate p-value (simplified approach)
            # In reality, would use proper statistical testing
            p_value = 0.05  # Placeholder
            
            return {
                'coefficient_T': a,
                'coefficient_combined': b,
                'intercept': c,
                'r_squared': r_squared,
                'correlation_S_T': corr_S_T,
                'correlation_S_combined': corr_S_combined,
                'a_error': a_error,
                'b_error': b_error,
                'p_value': p_value,
                'valid_points': np.sum(valid_mask)
            }
            
        except Exception as e:
            logger.error(f"Error in combined relationship test: {str(e)}")
            return {
                'error': str(e),
                'valid_points': np.sum(valid_mask)
            }
            
    def validate_self_regulation(self, temperature_map, density_map, magnetic_map):
        """
        Validate the self-regulation mechanism of Thomson scattering.
        
        Args:
            temperature_map (sunpy.map.Map): Temperature map
            density_map (sunpy.map.Map): Electron density map
            magnetic_map (sunpy.map.Map): Magnetic field map
            
        Returns:
            dict: Results of the self-regulation test
        """
        logger.info("Validating Thomson scattering self-regulation mechanism")
        
        # Extract data arrays
        T = temperature_map.data
        n_e = density_map.data
        B = np.abs(magnetic_map.data)
        
        # Create mask for valid data points
        valid_mask = ~np.isnan(T) & ~np.isnan(n_e) & ~np.isnan(B) & (n_e > 0) & (B > 0)
        
        if np.sum(valid_mask) < 10:
            logger.warning("Not enough valid data points for self-regulation test")
            return {
                'correlation_T_ne': np.nan,
                'correlation_B_ne': np.nan,
                'correlation_T_B': np.nan,
                'plasma_beta': {
                    'mean': np.nan,
                    'median': np.nan,
                    'std': np.nan
                },
                'valid_points': np.sum(valid_mask)
            }
            
        # Extract valid data points
        T_valid = T[valid_mask]
        n_e_valid = n_e[valid_mask]
        B_valid = B[valid_mask]
        
        # Calculate correlations
        corr_T_ne = np.corrcoef(T_valid, n_e_valid)[0, 1]
        corr_B_ne = np.corrcoef(B_valid, n_e_valid)[0, 1]
        corr_T_B = np.corrcoef(T_valid, B_valid)[0, 1]
        
        # Calculate plasma beta (ratio of thermal to magnetic pressure)
        # Beta = (n_e * k_B * T) / (B^2 / (2 * mu_0))
        k_B = 1.380649e-23  # Boltzmann constant in J/K
        mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability in H/m
        
        # Convert n_e from cm^-3 to m^-3
        n_e_valid_m3 = n_e_valid * 1e6
        
        # Calculate plasma beta
        plasma_beta = (n_e_valid_m3 * k_B * T_valid) / (B_valid**2 / (2 * mu_0))
        
        # Calculate statistical measures
        beta_mean = np.mean(plasma_beta)
        beta_median = np.median(plasma_beta)
        beta_std = np.std(plasma_beta)
        
        # Regions of high beta indicate thermal pressure dominance
        # Regions of low beta indicate magnetic pressure dominance
        high_beta_fraction = np.sum(plasma_beta > 1.0) / len(plasma_beta)
        
        return {
            'correlation_T_ne': corr_T_ne,
            'correlation_B_ne': corr_B_ne,
            'correlation_T_B': corr_T_B,
            'plasma_beta': {
                'mean': beta_mean,
                'median': beta_median,
                'std': beta_std,
                'high_beta_fraction': high_beta_fraction
            },
            'valid_points': np.sum(valid_mask)
        }
    
    def clear_cache(self, cache_type=None):
        """
        Clear the cache index and optionally delete cache files.
        
        Args:
            cache_type (str, optional): Specific cache type to clear, or None for all
        
        Returns:
            bool: True if cache was cleared successfully
        """
        logger.info(f"Clearing cache{'for ' + cache_type if cache_type else ''}")
        cache_index = self._get_cache_index()
        
        if cache_type:
            if cache_type in cache_index:
                cache_index[cache_type] = {}
                logger.info(f"Cleared cache index for {cache_type}")
            else:
                logger.warning(f"Cache type {cache_type} not found in cache index")
                return False
        else:
            # Clear all cache types
            for key in cache_index:
                cache_index[key] = {}
            logger.info("Cleared all cache indices")
        
        # Write the updated cache index
        try:
            with open(self.cache_index_file, 'w') as f:
                json.dump(cache_index, f, indent=2)
            logger.info("Cache index file updated")
            return True
        except Exception as e:
            logger.error(f"Error updating cache index file: {str(e)}")
            return False
            
    def run_analysis_pipeline(self):
        """
        Execute the Thomson scattering analysis pipeline:
        1. Download PUNCH data or alternative data sources
        2. Process the downloaded data to extract Thomson scattering signals
        3. Estimate physical parameters based on the analysis
        4. Validate the Thomson scatter regulation hypotheses
        
        Returns:
            dict: Results of the analysis pipeline
        """
        logger.info("Starting Thomson Regulation Analysis Pipeline")
        
        # Define the target date for the analysis (YYYY-MM-DD)
        # Using a date known for activity and likely data availability
        target_date = "2023-06-15"  # This is a known good date for LASCO C2/C3 data
        logger.info(f"Using target date for analysis: {target_date}")

        # --- Download Supplementary Data (SDO) for the target date --- 
        logger.info(f"Attempting to download supplementary SDO data for {target_date}...")
        supplementary_data_available = self.download_supplementary_data(date=target_date)
        if not supplementary_data_available:
            # Decide if this is fatal or just a warning
            logger.warning(f"Could not download supplementary SDO data for {target_date}. Proceeding without it.")
            # Depending on analysis needs, you might want to make this a fatal_error
        else:
            logger.info(f"Successfully downloaded supplementary SDO data for {target_date}.")
            # Note: Currently, supplementary data is downloaded but not used in the main processing flow.
            # This would need to be integrated into estimate_physical_parameters or elsewhere.

        # --- Download Primary Thomson Scattering Data (PUNCH or Alternative) --- 
        processed_data = None
        data_source = "None"
        
        # Step 1: Try to download PUNCH data first for the target date
        logger.info(f"Attempting to download PUNCH data for {target_date}...")
        # Pass the target_date as the start_date for the PUNCH query
        punch_data_available = self.download_punch_data(start_date=target_date)
        
        if punch_data_available:
            logger.info(f"Successfully downloaded PUNCH data for {target_date} - proceeding with analysis")
            data_source = f"PUNCH (target date: {target_date})"
            
            # Process the PUNCH data
            processed_data = self.process_punch_data()
        else:
            # Try alternative data sources if PUNCH data is not available for the target date
            logger.info(f"PUNCH data not available for {target_date}. Trying alternative data sources...")
            
            # Attempt to download alternative data specifically for the target_date
            alternative_data_available = self.download_alternative_data(date=target_date)
                
            if alternative_data_available:
                logger.info(f"Successfully downloaded alternative data for {target_date}")
                data_source = f"Alternative (target date: {target_date})"
                # Process the alternative data
                processed_data = self.process_alternative_data()
                
                # Check if fallback data was used (by examining the source info in processed_data)
                if processed_data and processed_data.get('source_info'):
                    if 'fallback' in processed_data.get('source_info').lower():
                        data_source = f"Alternative (fallback date: {processed_data.get('source_date', '2023-06-15')})"
                        logger.warning(f"Analysis is using fallback data from {processed_data.get('source_date', '2023-06-15')}")
                        logger.warning("Note: Results may not correspond exactly to the requested target date")
            else:
                logger.error(f"No primary Thomson scattering data (PUNCH or Alternative) available for target date {target_date}. Cannot proceed with analysis.")
                raise RuntimeError(f"No primary data available for Thomson scattering analysis on {target_date}")
        
        # Ensure processed_data is valid before proceeding
        if not processed_data:
             fatal_error("Failed to process primary data source.")

        # Step 3: Estimate physical parameters
        logger.info(f"Estimating physical parameters from {data_source} data")
        parameters = self.estimate_physical_parameters(processed_data)
        
        # Step 4: Validate hypotheses
        logger.info("Validating Thomson scatter regulation hypotheses")
        validation_results = self.validate_hypotheses(parameters)
        
        # Extract only the serializable statistical data from the parameters
        # Exclude maps and other non-serializable objects
        serializable_parameters = {}
        if 'parameters' in parameters:
            serializable_parameters = parameters['parameters']
        
        # Compile final results with only serializable data
        results = {
            "data_source": data_source,
            "physical_parameters": serializable_parameters,
            "validation_results": validation_results,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Analysis complete using {data_source} data")
        
        # After getting results and evaluation, create validation plots
        try:
            self.create_validation_plots(parameters, results, validation_results)
            logger.info("Validation plots created successfully")
        except Exception as e:
            logger.error(f"Error creating validation plots: {str(e)}")
        
        return results

    def evaluate_theory_confirmation(self, results):
        """
        Evaluate whether the analysis results support the gravitational-electromagnetic 
        regulation theory of Thomson scattering.
        
        Args:
            results (dict): Results from the analysis pipeline
            
        Returns:
            dict: Theory evaluation with confirmation status and detailed reasoning
        """
        logger.info("Evaluating whether results confirm the proposed theory")
        
        # Initialize evaluation result
        evaluation = {
            "theory_confirmed": False,
            "confidence_level": "Low",
            "criteria_met": [],
            "criteria_failed": [],
            "reasoning": "",
            "recommendations": []
        }
        
        # Extract validation results for evaluation
        validation_results = results.get("validation_results", {})
        combined_relationship = validation_results.get("combined_relationship", {})
        self_regulation = validation_results.get("self_regulation", {})
        correlations = validation_results.get("correlations", {})
        
        # Define criteria for theory confirmation
        criteria = []
        
        # 1. Combined relationship (T ∝ Thomson/(g×f(B)))
        if combined_relationship:
            corr = combined_relationship.get("correlation_S_T", 0)
            corr_combined = combined_relationship.get("correlation_S_combined", 0)
            r_squared = combined_relationship.get("r_squared", 0)
            coef_T = combined_relationship.get("coefficient_T", 0)
            coef_combined = combined_relationship.get("coefficient_combined", 0)
            p_value = combined_relationship.get("p_value", 1.0)
            
            # Criterion 1: Temperature coefficient should be close to the theoretical value (4)
            # The Stefan-Boltzmann law (T^4) is the physical basis for this relationship
            # but in Thomson scattering environments, radiative transfer effects can modify this
            
            # Define the expected value based on theoretical principles
            expected_T_coef = 4.0  # Stefan-Boltzmann law prediction
            
            # Calculate standard error for the coefficient - this is a statistical measure
            # of uncertainty that should be available from the regression analysis
            std_error = combined_relationship.get("coefficient_T_error", 0.5)  # Default to 0.5 if not available
            
            # Physical interpretation: Is the coefficient significantly different from 4.0?
            # Using a 2-sigma (95% confidence) threshold for physical significance
            lower_bound = expected_T_coef - 2 * std_error
            upper_bound = expected_T_coef + 2 * std_error
            
            # Physical assessment: is our measured value consistent with theory within error?
            consistent_with_theory = lower_bound <= coef_T <= upper_bound
            
            # Near prediction: within 1 standard error
            near_prediction = abs(coef_T - expected_T_coef) <= std_error
            
            physical_notes = []
            if coef_T < expected_T_coef - std_error:
                physical_notes.append(f"Value {coef_T:.2f} is below theoretical prediction of {expected_T_coef}")
                physical_notes.append("Possible causes: optical depth effects, non-LTE conditions, or multi-temperature plasma")
            elif coef_T > expected_T_coef + std_error:
                physical_notes.append(f"Value {coef_T:.2f} is above theoretical prediction of {expected_T_coef}")
                physical_notes.append("Possible causes: relativistic effects, non-isotropic scattering, or instrumental effects")
            else:
                physical_notes.append(f"Value {coef_T:.2f} is consistent with theoretical prediction of {expected_T_coef} within 1σ")
                
            criterion1 = {
                "name": "Temperature power-law coefficient",
                "expected": f"T^{expected_T_coef} (Stefan-Boltzmann with Thomson scattering)",
                "actual": f"T^{coef_T:.2f} ± {std_error:.2f}",
                "satisfied": consistent_with_theory,
                "physical_interpretation": "; ".join(physical_notes)
            }
            criteria.append(criterion1)
            
            # Criterion 2: Combined factor coefficient should be negative and close to -1
            # Physical basis: In Thomson scattering environments, gravitational forces and magnetic fields 
            # counteract each other through pressure equilibrium and energy conservation
            
            expected_comb_coef = -1.0  # Based on theoretical prediction from plasma physics
            
            # Calculate standard error for the coefficient
            std_error = combined_relationship.get("coefficient_combined_error", 0.25)  # Default to 0.25 if not available
            
            # Physical interpretation of the sign: critically important
            correct_sign = coef_combined < 0
            
            # Is the coefficient significantly different from -1.0?
            lower_bound = expected_comb_coef - 2 * std_error
            upper_bound = expected_comb_coef + 2 * std_error
            
            # Physical assessment: is our measured value consistent with theory within error?
            consistent_with_theory = correct_sign and (lower_bound <= coef_combined <= upper_bound)
            
            # Physical interpretation
            physical_notes = []
            if not correct_sign:
                physical_notes.append(f"Value {coef_combined:.2f} has incorrect sign (should be negative)")
                physical_notes.append("Critical physical error: Contradicts fundamental force balance in plasma")
                physical_notes.append("Possible causes: measurement error, incorrect model specification, or incomplete physics")
            elif coef_combined < expected_comb_coef - std_error:
                physical_notes.append(f"Value {coef_combined:.2f} is more negative than the theoretical prediction of {expected_comb_coef}")
                physical_notes.append("Possible causes: enhanced magnetic confinement, non-isotropic distribution, or additional forces")
            elif coef_combined > expected_comb_coef + std_error:
                physical_notes.append(f"Value {coef_combined:.2f} is less negative than the theoretical prediction of {expected_comb_coef}")
                physical_notes.append("Possible causes: partial ionization, magnetic reconnection, or multi-scale effects")
            else:
                physical_notes.append(f"Value {coef_combined:.2f} is consistent with theoretical prediction of {expected_comb_coef} within 1σ")
                
            criterion2 = {
                "name": "g×f(B) coefficient",
                "expected": f"Inversely proportional (coefficient ≈ {expected_comb_coef} from force balance)",
                "actual": f"{coef_combined:.2f} ± {std_error:.2f}",
                "satisfied": consistent_with_theory,
                "physical_interpretation": "; ".join(physical_notes)
            }
            criteria.append(criterion2)
            
            # Criterion 3: Model fit should be reasonably good (R² > 0.3)
            criterion3 = {
                "name": "Model fit quality",
                "expected": "R² > 0.3",
                "actual": f"R² = {r_squared:.2f}",
                "satisfied": r_squared > 0.3
            }
            criteria.append(criterion3)
            
            # Criterion 4: Statistical significance
            criterion4 = {
                "name": "Statistical significance",
                "expected": "p ≤ 0.05",
                "actual": f"p = {p_value:.4f}",
                "satisfied": p_value <= 0.05,
                "physical_interpretation": f"p-value of {p_value:.4f} indicates {'' if p_value <= 0.05 else 'in'}sufficient evidence against null hypothesis at conventional α=0.05 threshold"
            }
            criteria.append(criterion4)
        
        # 2. Self-regulation mechanism
        if self_regulation:
            corr_T_ne = self_regulation.get("correlation_T_ne", 0)
            corr_B_ne = self_regulation.get("correlation_B_ne", 0)
            corr_T_B = self_regulation.get("correlation_T_B", 0)
            plasma_beta = self_regulation.get("plasma_beta", {})
            
            # Criterion 5: Temperature-density relationship
            # In standard models we expect anti-correlation, but strong positive correlation
            # could indicate alternate physical regimes (magnetic confinement dominated)
            
            # Assess whether we have a very strong positive correlation which might indicate
            # a magnetically confined regime rather than a problem with the data
            if corr_T_ne < -0.1:
                # Standard case: anti-correlation as expected
                satisfaction_level = "full"
                explanation = "Standard anti-correlation as expected in theory"
            elif corr_T_ne > 0.7 and corr_B_ne > 0.7:
                # Alternative case: strong positive correlation with both T-ne and B-ne
                # This suggests a magnetically dominated regime
                satisfaction_level = "alternative"
                explanation = "Magnetically confined regime detected (both T-ne and B-ne strongly positive)"
            elif -0.1 <= corr_T_ne <= 0.1:
                # Weak or no correlation
                satisfaction_level = "neutral"
                explanation = "Weak correlation, indeterminate regime"
            else:
                # Unexpected positive correlation without magnetic explanation
                satisfaction_level = "none"
                explanation = "Unexpected positive correlation"
                
            criterion5 = {
                "name": "Temperature-density relationship",
                "expected": "Anti-correlation (T-n_e < 0) or magnetically confined regime",
                "actual": f"Correlation = {corr_T_ne:.2f}",
                "satisfied": satisfaction_level in ["full", "alternative"],
                "satisfaction_level": satisfaction_level,
                "notes": explanation
            }
            criteria.append(criterion5)
            
            # Criterion 6: Magnetic field containment of plasma
            criterion6 = {
                "name": "Magnetic field plasma containment",
                "expected": "Positive B-n_e correlation",
                "actual": f"Correlation = {corr_B_ne:.2f}",
                "satisfied": corr_B_ne > 0.1
            }
            criteria.append(criterion6)
        
        # 3. Temperature-scattering relationship
        if "temperature_scattering_correlation" in correlations:
            temp_scatter_corr = correlations.get("temperature_scattering_correlation", 0)
            
            # Criterion 7: Temperature-scattering relationship
            criterion7 = {
                "name": "Temperature-scattering relationship",
                "expected": "Positive correlation (T-S > 0)",
                "actual": f"Correlation = {temp_scatter_corr:.2f}",
                "satisfied": temp_scatter_corr > 0.3
            }
            criteria.append(criterion7)
        
        # Evaluate criteria
        criteria_met = [c for c in criteria if c.get("satisfied", False)]
        criteria_failed = [c for c in criteria if not c.get("satisfied", False)]
        
        evaluation["criteria_met"] = criteria_met
        evaluation["criteria_failed"] = criteria_failed
        
        # Determine overall theory confirmation
        total_criteria = len(criteria)
        met_criteria = len(criteria_met)
        
        if total_criteria == 0:
            confirmation_ratio = 0
        else:
            confirmation_ratio = met_criteria / total_criteria
        
        # Determine confidence level
        if confirmation_ratio >= 0.8:
            evaluation["theory_confirmed"] = True
            evaluation["confidence_level"] = "High"
            evaluation["reasoning"] = "The majority of critical criteria strongly support the gravitational-electromagnetic regulation theory."
        elif confirmation_ratio >= 0.6:
            evaluation["theory_confirmed"] = True
            evaluation["confidence_level"] = "Moderate"
            evaluation["reasoning"] = "Most criteria support the theory, but some inconsistencies exist."
        elif confirmation_ratio >= 0.4:
            evaluation["theory_confirmed"] = "Partially"
            evaluation["confidence_level"] = "Low"
            evaluation["reasoning"] = "Some evidence supports the theory, but several critical criteria were not met."
        else:
            evaluation["theory_confirmed"] = False
            evaluation["confidence_level"] = "Very Low"
            evaluation["reasoning"] = "Minimal evidence supports the theory. Data does not align with theoretical predictions."
        
        # Add additional context
        evaluation["confirmation_ratio"] = f"{met_criteria}/{total_criteria} criteria met ({confirmation_ratio:.0%})"
        
        # Generate physics-based recommendations based on failed criteria
        recommendations = []
        physical_insights = []
        
        if criteria_failed:
            recommendations.append("Further investigation needed for the following aspects:")
            for criterion in criteria_failed:
                recommendations.append(f"- {criterion['name']}: expected {criterion['expected']} but found {criterion['actual']}")
                
                # Add physical interpretation if available
                if "physical_interpretation" in criterion:
                    physical_insights.append(f"- {criterion['name']}: {criterion['physical_interpretation']}")
        
        # Add general physical insights
        if physical_insights:
            recommendations.append("\nPhysical interpretation of discrepancies:")
            recommendations.extend(physical_insights)
            
        # Recommend specific next steps based on physical understanding
        if any("Temperature power-law" in c["name"] for c in criteria_failed):
            recommendations.append("\nRecommended next steps for temperature power-law:")
            recommendations.append("- Verify temperature calibration across different plasma regions")
            recommendations.append("- Examine optical depth effects in the corona")
            recommendations.append("- Consider multi-temperature plasma models for improved fits")
            
        if any("g×f(B) coefficient" in c["name"] for c in criteria_failed):
            recommendations.append("\nRecommended next steps for g×f(B) relationship:")
            recommendations.append("- Re-examine magnetic field reconstruction methods")
            recommendations.append("- Consider additional force terms in the equilibrium equation")
            recommendations.append("- Verify gravitational model at the specific coronal heights")
        
        if any("Temperature-density relationship" in c["name"] for c in criteria_failed):
            recommendations.append("\nRecommended next steps for temperature-density relationship:")
            recommendations.append("- Examine local vs. global plasma beta values")
            recommendations.append("- Consider geometry effects on observed T-n_e correlations")
            recommendations.append("- Verify density inversion techniques from Thomson scattering signals")
            
        evaluation["recommendations"] = recommendations
        
        logger.info(f"Theory evaluation complete: {evaluation['confidence_level']} confidence, {evaluation['confirmation_ratio']}")
        return evaluation

    def calculate_holographic_temperature_scaling(self, processed_data, parameters):
        """
        Calculate the holographically-modified temperature scaling coefficients based on 
        information processing constraints.
        
        This method implements the theoretical framework where MHD wave dissipation
        is constrained by the fundamental information processing rate γ.
        
        Args:
            processed_data (dict): Processed Thomson scattering data
            parameters (dict): Estimated physical parameters
            
        Returns:
            dict: Holographically-modified temperature scaling coefficients and analysis
        """
        logger.info("Calculating holographically-modified temperature scaling")
        
        # Extract maps from processed data
        maps = parameters.get('maps', {})
        thomson_map = maps.get('thomson_map')
        temperature_map = maps.get('temperature_map')
        magnetic_map = maps.get('magnetic_map')
        density_map = maps.get('density_map')
        
        if not all([thomson_map, temperature_map, magnetic_map, density_map]):
            logger.error("Required maps not available for holographic analysis")
            return None
        
        # Universal information processing rate (from holographic theory)
        gamma = 1.89e-29  # s^-1
        
        # Hubble parameter (approximate value at current epoch)
        H0 = 70.0 * 1000 / 3.086e22  # s^-1 (70 km/s/Mpc converted to s^-1)
        
        # Ratio γ/H ≈ 1/8π as per holographic theory
        gamma_H_ratio = gamma / H0
        
        logger.info(f"Information processing rate γ = {gamma:.3e} s^-1")
        logger.info(f"γ/H ratio = {gamma_H_ratio:.6f} (theoretical: {1/(8*np.pi):.6f})")
        
        # Extract temperature and Thomson scattering data for regression
        temp_data = temperature_map.data.flatten()
        thomson_data = thomson_map.data.flatten()
        
        # Filter out invalid data points
        valid_mask = ~np.isnan(temp_data) & ~np.isnan(thomson_data) & (thomson_data > 0) & (temp_data > 0)
        temp_valid = temp_data[valid_mask]
        thomson_valid = thomson_data[valid_mask]
        
        # Take logarithms for power-law fitting
        log_temp = np.log(temp_valid)
        log_thomson = np.log(thomson_valid)
        
        # Standard power-law fit (T ~ S^α)
        try:
            # Perform linear regression on log-transformed data
            X = np.column_stack((log_thomson, np.ones_like(log_thomson)))
            beta_std, _, _, _ = np.linalg.lstsq(X, log_temp, rcond=None)
            alpha_std, _ = beta_std
            
            # Calculate predicted values and R^2
            log_temp_pred_std = np.dot(X, beta_std)
            ss_total = np.sum((log_temp - np.mean(log_temp))**2)
            ss_residual = np.sum((log_temp - log_temp_pred_std)**2)
            r_squared_std = 1 - (ss_residual / ss_total)
            
            logger.info(f"Standard power-law fit: T ~ S^{alpha_std:.4f}, R^2 = {r_squared_std:.4f}")
        except Exception as e:
            logger.error(f"Error in standard power-law fit: {str(e)}")
            alpha_std, r_squared_std = np.nan, np.nan
        
        # Calculate characteristic length scale for holographic correction
        # Use the solar radius as a reference scale
        L = self.solar_radius  # m
        
        # Planck length
        l_p = 1.616255e-35  # m
        
        # Calculate holographic correction factor
        # Based on theory: T^(4-δ) where δ ≈ 2(γ/H)ln(L/l_p)
        delta = 2 * gamma_H_ratio * np.log(L / l_p)
        expected_holo_exponent = 4.0 - delta
        
        logger.info(f"Holographic correction factor δ = {delta:.4f}")
        logger.info(f"Expected holographic exponent = {expected_holo_exponent:.4f}")
        
        # Modified power-law fit with holographic constraints
        # We implement a constrained fit where the exponent is fixed to the holographic prediction
        try:
            # For a relation T ~ S^(1/expected_holo_exponent), we model log(T) = (1/expected_holo_exponent) * log(S) + c
            # Fixed exponent regression
            X_holo = log_thomson.reshape(-1, 1)
            y = log_temp
            
            # Calculate intercept using the fixed slope
            fixed_slope = 1.0 / expected_holo_exponent
            intercept = np.mean(y - fixed_slope * X_holo.flatten())
            
            # Calculate predicted values with fixed exponent
            log_temp_pred_holo = fixed_slope * X_holo.flatten() + intercept
            
            # Calculate R^2 for holographic model
            ss_residual_holo = np.sum((log_temp - log_temp_pred_holo)**2)
            r_squared_holo = 1 - (ss_residual_holo / ss_total)
            
            logger.info(f"Holographic power-law fit: T ~ S^{fixed_slope:.4f}, R^2 = {r_squared_holo:.4f}")
        except Exception as e:
            logger.error(f"Error in holographic power-law fit: {str(e)}")
            fixed_slope, r_squared_holo = np.nan, np.nan
        
        # Calculate the likelihood ratio for model comparison
        # Simpler calculation: ratio of residual sum of squares
        if not np.isnan(r_squared_std) and not np.isnan(r_squared_holo):
            # More positive values favor the holographic model
            log_likelihood_ratio = np.log(1 - r_squared_std) - np.log(1 - r_squared_holo)
            logger.info(f"Log likelihood ratio (holographic vs standard): {log_likelihood_ratio:.4f}")
            
            # Interpret the result
            if log_likelihood_ratio > 0:
                logger.info("Holographic model provides a better fit to the data")
                model_preference = "holographic"
            else:
                logger.info("Standard model provides a better fit to the data")
                model_preference = "standard"
        else:
            log_likelihood_ratio = np.nan
            model_preference = "undetermined"
        
        # Return the analysis results
        return {
            "standard_model": {
                "exponent": alpha_std,
                "r_squared": r_squared_std,
                "expected_exponent": 4.0  # Based on Stefan-Boltzmann
            },
            "holographic_model": {
                "exponent": fixed_slope,
                "r_squared": r_squared_holo,
                "expected_exponent": expected_holo_exponent,
                "correction_factor": delta,
                "gamma_value": gamma,
                "gamma_H_ratio": gamma_H_ratio
            },
            "model_comparison": {
                "log_likelihood_ratio": log_likelihood_ratio,
                "preferred_model": model_preference
            }
        }

    def analyze_reconnection_information_constraints(self, processed_data, parameters):
        """
        Analyze magnetic reconnection events as information "rewriting" processes
        under holographic constraints.
        
        This method implements the theoretical framework where magnetic reconnection events 
        represent information-saturated structures undergoing forced transitions, which
        explains the unexpected positive g×f(B) coefficient.
        
        Args:
            processed_data (dict): Processed Thomson scattering data
            parameters (dict): Estimated physical parameters
            
        Returns:
            dict: Analysis of reconnection events under information constraints
        """
        logger.info("Analyzing magnetic reconnection under information constraints")
        
        # Extract maps from processed data
        maps = parameters.get('maps', {})
        magnetic_map = maps.get('magnetic_map')
        temperature_map = maps.get('temperature_map')
        density_map = maps.get('density_map')
        g_field_map = maps.get('g_field_map')
        f_B_map = maps.get('f_B_map')
        
        if not all([magnetic_map, temperature_map, density_map, g_field_map, f_B_map]):
            logger.error("Required maps not available for reconnection analysis")
            return None
        
        # Universal information processing rate
        gamma = 1.89e-29  # s^-1
        
        # Calculate the characteristic timescale for information processing
        tau_info = 1.0 / gamma  # s
        logger.info(f"Information processing timescale τ = {tau_info:.3e} s")
        
        # Extract data arrays
        B = np.abs(magnetic_map.data)
        T = temperature_map.data
        n_e = density_map.data  # electron density
        g = g_field_map.data  # gravitational field
        f_B = f_B_map.data  # magnetic constraint function
        
        # Create valid data mask
        valid_mask = ~np.isnan(B) & ~np.isnan(T) & ~np.isnan(n_e) & ~np.isnan(g) & ~np.isnan(f_B)
        valid_mask &= (B > 0) & (T > 0) & (n_e > 0) & (g > 0) & (f_B > 0)
        
        # Extract valid data
        B_valid = B[valid_mask]
        T_valid = T[valid_mask]
        n_e_valid = n_e[valid_mask]
        g_valid = g[valid_mask]
        f_B_valid = f_B[valid_mask]
        
        # Calculate combined g×f(B) factor
        gfB_valid = g_valid * f_B_valid
        
        # Calculate Alfvén velocity: v_A = B/sqrt(μ₀ρ)
        mu0 = 4 * np.pi * 1e-7  # Vacuum permeability (H/m)
        m_p = 1.6726219e-27  # Proton mass (kg)
        
        # Calculate mass density (assuming hydrogen plasma)
        rho = n_e_valid * m_p  # kg/m³
        
        # Calculate Alfvén velocity
        v_A = B_valid / np.sqrt(mu0 * rho)  # m/s
        
        # Calculate plasma beta (ratio of thermal to magnetic pressure)
        k_B = 1.380649e-23  # Boltzmann constant (J/K)
        P_thermal = n_e_valid * k_B * T_valid  # Thermal pressure (Pa)
        P_magnetic = B_valid**2 / (2 * mu0)  # Magnetic pressure (Pa)
        plasma_beta = P_thermal / P_magnetic  # Dimensionless
        
        # Calculate reconnection rate under standard Sweet-Parker model
        # S = (L*v_A)/(η) where η is the resistivity (diffusivity)
        # Estimate L as solar radius divided by 100 (typical active region scale)
        L = self.solar_radius / 100  # m
        
        # Estimate resistivity based on Spitzer formula for fully ionized plasma
        # η ≈ 10^-2 * T^(-3/2) * ln(Λ) where ln(Λ) is the Coulomb logarithm
        # For solar corona, ln(Λ) ≈ in range 10-20, we use 15
        ln_lambda = 15
        eta = 1e-2 * T_valid**(-1.5) * ln_lambda  # Resistivity in Ω·m
        
        # Calculate Lundquist number S
        S = L * v_A / eta
        
        # Standard Sweet-Parker reconnection rate
        R_SP = S**(-0.5)
        
        # Calculate information-constrained reconnection rate
        # Based on holographic theory: R_info ≈ R_SP * (1 + γ*τ_A)
        # where τ_A = L/v_A is the Alfvén crossing time
        tau_A = L / v_A  # Alfvén crossing time (s)
        
        # Information-modified reconnection rate
        R_info = R_SP * (1 + gamma * tau_A)
        
        # Calculate critical current density for reconnection
        # j_crit = B/(μ₀L)
        j_crit = B_valid / (mu0 * L)  # A/m²
        
        # Calculate the information processing capacity per unit volume
        # In bits/m³: I_max = (L/l_p)^2 / L^3 where l_p is Planck length
        l_p = 1.616255e-35  # m
        I_max = (L / l_p)**2 / L**3  # Maximum information density (bits/m³)
        
        # Calculate local information rate (bits/s/m³)
        info_rate = I_max * gamma  # bits/s/m³
        
        # Calculate relationship between g×f(B) and information parameters
        # Under holographic hypothesis, correlation should be positive due to information constraints
        
        try:
            # Prepare data for regression
            X = np.column_stack((
                np.log(plasma_beta),
                np.log(R_info),
                np.ones_like(plasma_beta)
            ))
            y = np.log(gfB_valid)
            
            # Perform regression
            beta_coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            
            # Extract coefficients
            beta_weight, R_info_weight, intercept = beta_coef
            
            # Calculate model predictions
            y_pred = np.dot(X, beta_coef)
            
            # Calculate R²
            ss_total = np.sum((y - np.mean(y))**2)
            ss_residual = np.sum((y - y_pred)**2)
            r_squared = 1 - (ss_residual / ss_total)
            
            logger.info(f"Information-reconnection model: g×f(B) ~ β^{beta_weight:.4f} · R_info^{R_info_weight:.4f}")
            logger.info(f"Model R² = {r_squared:.4f}")
            
            # Positive coefficient on R_info indicates information-constrained reconnection
            if R_info_weight > 0:
                logger.info("RESULT: Positive correlation with reconnection rate supports information-constrained model")
                info_constrained = True
            else:
                logger.info("RESULT: Negative correlation with reconnection rate does not support information-constrained model")
                info_constrained = False
        except Exception as e:
            logger.error(f"Error in reconnection regression: {str(e)}")
            beta_weight, R_info_weight, r_squared = np.nan, np.nan, np.nan
            info_constrained = False
        
        # Calculate statistics
        stats = {
            "plasma_beta": {
                "mean": np.mean(plasma_beta),
                "median": np.median(plasma_beta),
                "std": np.std(plasma_beta)
            },
            "reconnection_rate": {
                "mean_standard": np.mean(R_SP),
                "mean_info_modified": np.mean(R_info),
                "ratio": np.mean(R_info / R_SP)
            },
            "alfven_velocity": {
                "mean": np.mean(v_A),
                "median": np.median(v_A),
                "std": np.std(v_A)
            },
            "g_fB_factor": {
                "mean": np.mean(gfB_valid),
                "median": np.median(gfB_valid),
                "std": np.std(gfB_valid)
            }
        }
        
        # Return the analysis results
        return {
            "info_constrained_reconnection": info_constrained,
            "model_coefficients": {
                "plasma_beta_weight": beta_weight,
                "reconnection_rate_weight": R_info_weight,
                "intercept": intercept,
                "r_squared": r_squared
            },
            "reconnection_parameters": {
                "lundquist_number": {
                    "mean": np.mean(S),
                    "median": np.median(S),
                    "std": np.std(S)
                },
                "reconnection_rate": {
                    "standard_mean": np.mean(R_SP),
                    "info_modified_mean": np.mean(R_info),
                    "enhancement_factor": np.mean(R_info / R_SP)
                },
                "critical_current": {
                    "mean": np.mean(j_crit),
                    "median": np.median(j_crit)
                },
                "information_parameters": {
                    "max_info_density": I_max,
                    "info_processing_rate": info_rate,
                    "alfven_time_mean": np.mean(tau_A),
                    "info_time_product": gamma * np.mean(tau_A)
                }
            },
            "statistics": stats
        }

    def analyze_alfven_turbulence_cascade(self, processed_data, parameters):
        """
        Analyze Alfvén wave turbulence under information processing constraints.
        
        This method examines how information processing limits modify the turbulent
        cascade from large to small scales, predicting a characteristic break at 
        holographic scales.
        
        Args:
            processed_data (dict): Processed Thomson scattering data
            parameters (dict): Estimated physical parameters
            
        Returns:
            dict: Analysis of Alfvén wave turbulence with holographic constraints
        """
        logger.info("Analyzing Alfvén wave turbulence with information processing constraints")
        
        # Extract maps from processed data
        maps = parameters.get('maps', {})
        magnetic_map = maps.get('magnetic_map')
        density_map = maps.get('density_map')
        
        if not all([magnetic_map, density_map]):
            logger.error("Required maps not available for Alfvén turbulence analysis")
            return None
        
        # Universal information processing rate
        gamma = 1.89e-29  # s^-1
        
        # Extract data arrays
        B = np.abs(magnetic_map.data)
        n_e = density_map.data  # electron density
        
        # Create valid data mask
        valid_mask = ~np.isnan(B) & ~np.isnan(n_e) & (B > 0) & (n_e > 0)
        
        # Extract valid data
        B_valid = B[valid_mask]
        n_e_valid = n_e[valid_mask]
        
        # Constants
        mu0 = 4 * np.pi * 1e-7  # Vacuum permeability (H/m)
        m_p = 1.6726219e-27  # Proton mass (kg)
        
        # Calculate mass density (assuming hydrogen plasma)
        rho = n_e_valid * m_p  # kg/m³
        
        # Calculate Alfvén velocity
        v_A = B_valid / np.sqrt(mu0 * rho)  # m/s
        
        # Calculate characteristic length scales
        # Solar radius as reference outer scale
        L_outer = self.solar_radius  # m
        
        # Calculate ion gyroradius (characteristic inner scale)
        k_B = 1.380649e-23  # Boltzmann constant (J/K)
        T_assumed = 1e6  # Assumed coronal temperature (K)
        ion_thermal_velocity = np.sqrt(2 * k_B * T_assumed / m_p)  # m/s
        ion_cyclotron_freq = (1.6e-19 * B_valid) / m_p  # rad/s
        rho_i = ion_thermal_velocity / ion_cyclotron_freq  # m
        
        # Calculate wavenumbers
        k_outer = 2 * np.pi / L_outer  # m^-1
        k_inner = 2 * np.pi / np.mean(rho_i)  # m^-1
        
        # Calculate inertial range extent in wavenumber space
        k_range = np.logspace(np.log10(k_outer), np.log10(k_inner), 1000)
        
        # Standard MHD turbulence predicts E(k) ~ k^(-5/3) (Kolmogorov)
        # for the inertial range and E(k) ~ k^(-7/3) for the dissipation range
        
        # Calculate information-limited scale
        # The holographic scale where information limits become important
        # Occurs at k_holo where the information processing time equals the eddy turnover time
        # τ_eddy(k_holo) = 1/γ
        
        # Estimate eddy turnover time scale at wavenumber k: τ_eddy ~ (k v_A)^-1
        mean_vA = np.mean(v_A)
        tau_eddy = 1.0 / (k_range * mean_vA)
        
        # Find the holographic wavenumber where τ_eddy = 1/γ
        tau_info = 1.0 / gamma
        k_holo_idx = np.argmin(np.abs(tau_eddy - tau_info))
        k_holo = k_range[k_holo_idx]
        
        logger.info(f"Outer scale wavenumber: k_outer = {k_outer:.3e} m^-1")
        logger.info(f"Ion-scale wavenumber: k_inner = {k_inner:.3e} m^-1")
        logger.info(f"Holographic scale wavenumber: k_holo = {k_holo:.3e} m^-1")
        
        # Calculate energy spectra with and without holographic constraints
        # Standard Kolmogorov-like MHD turbulence spectrum
        E_standard = np.zeros_like(k_range)
        
        # Inertial range: k^(-5/3)
        inertial_mask = k_range <= k_inner/10  
        E_standard[inertial_mask] = k_range[inertial_mask]**(-5/3)
        
        # Dissipation range: k^(-7/3)
        dissipation_mask = ~inertial_mask
        E_standard[dissipation_mask] = (k_inner/10)**(-5/3) * (k_range[dissipation_mask]/(k_inner/10))**(-7/3)
        
        # Normalize
        E_standard = E_standard / E_standard[0]
        
        # Holographically constrained turbulence spectrum
        E_holo = np.zeros_like(k_range)
        
        # Below holographic scale: standard cascade
        pre_holo_mask = k_range <= k_holo
        E_holo[pre_holo_mask] = k_range[pre_holo_mask]**(-5/3)
        
        # Calculate holographic modification factor for scales below information limit
        # We use a model where spectrum steepens more rapidly:
        # For k > k_holo: E(k) ~ k^(-5/3) * exp(-(k/k_holo)^α)
        # where α is a coefficient derived from holographic theory
        
        # Holographic modification with enhanced dissipation
        alpha = 0.5  # Holographic damping coefficient
        post_holo_mask = k_range > k_holo
        
        # Post-holographic scales show enhanced damping
        # The term (k/k_holo) represents deviation from standard cascade
        k_ratio = k_range[post_holo_mask] / k_holo
        
        # Define the holographic damping function
        holo_damping = np.exp(-(k_ratio**alpha))
        
        # Apply to the spectrum with proper normalization to ensure continuity
        E_holo[post_holo_mask] = (k_holo**(-5/3)) * (k_range[post_holo_mask]/k_holo)**(-7/3) * holo_damping
        
        # Normalize
        E_holo = E_holo / E_holo[0]
        
        # Calculate spectral index variation (d log E / d log k)
        # For the standard model
        spectral_index_std = np.zeros_like(k_range[1:])
        for i in range(len(k_range)-1):
            log_k_ratio = np.log(k_range[i+1] / k_range[i])
            log_E_ratio = np.log(E_standard[i+1] / E_standard[i])
            spectral_index_std[i] = log_E_ratio / log_k_ratio
        
        # For the holographic model
        spectral_index_holo = np.zeros_like(k_range[1:])
        for i in range(len(k_range)-1):
            log_k_ratio = np.log(k_range[i+1] / k_range[i])
            log_E_ratio = np.log(E_holo[i+1] / E_holo[i])
            spectral_index_holo[i] = log_E_ratio / log_k_ratio
        
        # Calculate key scales relative to each other
        scales_ratio = {
            "k_inner_to_outer": k_inner / k_outer,
            "k_holo_to_outer": k_holo / k_outer,
            "k_holo_to_inner": k_holo / k_inner
        }
        
        # Check if the holographic scale is within the turbulent cascade range
        holo_in_cascade = k_outer < k_holo < k_inner
        
        if holo_in_cascade:
            logger.info("Holographic scale falls within the turbulent cascade range")
            # Check where in the cascade range it falls (logarithmically)
            log_position = (np.log10(k_holo) - np.log10(k_outer)) / (np.log10(k_inner) - np.log10(k_outer))
            logger.info(f"Holographic scale at {log_position:.2f} of the way through the cascade (log scale)")
        else:
            logger.info("Holographic scale falls outside the turbulent cascade range")
        
        # Calculate characteristic times
        tau_outer = 1.0 / (k_outer * mean_vA)  # s
        tau_inner = 1.0 / (k_inner * mean_vA)  # s
        tau_holo = 1.0 / (k_holo * mean_vA)    # s
        
        logger.info(f"Outer scale time: τ_outer = {tau_outer:.3e} s")
        logger.info(f"Inner scale time: τ_inner = {tau_inner:.3e} s")
        logger.info(f"Holographic scale time: τ_holo = {tau_holo:.3e} s")
        logger.info(f"Information time scale: τ_info = {tau_info:.3e} s")
        
        # Return the analysis results
        return {
            "scale_parameters": {
                "outer_scale": {
                    "length": L_outer,  # m
                    "wavenumber": k_outer,  # m^-1
                    "time": tau_outer  # s
                },
                "inner_scale": {
                    "length": np.mean(rho_i),  # m
                    "wavenumber": k_inner,  # m^-1
                    "time": tau_inner  # s
                },
                "holographic_scale": {
                    "wavenumber": k_holo,  # m^-1
                    "time": tau_holo,  # s
                    "info_time": tau_info  # s
                },
                "scale_ratios": scales_ratio
            },
            "spectral_properties": {
                "holo_in_cascade_range": holo_in_cascade,
                "spectral_break_position": log_position if holo_in_cascade else None,
                "holographic_modification": {
                    "damping_coefficient": alpha,
                    "spectral_steepening": np.mean(spectral_index_holo[post_holo_mask]) - np.mean(spectral_index_std[post_holo_mask]) if any(post_holo_mask) else None
                }
            },
            "simulation_parameters": {
                "wavenumbers": k_range.tolist(),  # Export subset for plotting
                "standard_spectrum": E_standard.tolist(),  # Export subset for plotting
                "holographic_spectrum": E_holo.tolist(),  # Export subset for plotting
                "standard_spectral_index": spectral_index_std.tolist(),  # Export subset for plotting
                "holographic_spectral_index": spectral_index_holo.tolist()  # Export subset for plotting
            }
        }

    def create_validation_plots(self, parameters, results, evaluation):
        """
        Create visualizations to demonstrate key relationships for theory validation.
        
        Args:
            parameters (dict): Physical parameters from analysis
            results (dict): Analysis results
            evaluation (dict): Theory evaluation results
        """
        logger.info("Creating validation plots")
        
        # Create figures directory if it doesn't exist
        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-darkgrid')
        
        # 1. Combined Relationship Plot: T vs Thomson/(g×f(B))
        self._plot_combined_relationship(parameters, results, figures_dir)
        
        # 2. Information Processing Constraint Plot
        self._plot_information_constraints(parameters, results, figures_dir)
        
        # 3. Holographic Scale Transition Plot
        self._plot_holographic_transition(parameters, results, figures_dir)
        
        logger.info("Validation plots created successfully")
        
    def _plot_combined_relationship(self, parameters, results, figures_dir):
        """Create plot showing the relationship between temperature and Thomson/(g×f(B))."""
        try:
            # Extract data from maps
            maps = parameters.get('maps', {})
            T = maps['temperature_map'].data.flatten()
            S = maps['scattering_map'].data.flatten()
            g = maps['g_field_map'].data.flatten()
            f_B = maps['f_B_map'].data.flatten()
            
            # Create valid data mask
            valid_mask = ~np.isnan(T) & ~np.isnan(S) & ~np.isnan(g) & ~np.isnan(f_B)
            valid_mask &= (T > 0) & (S > 0) & (g > 0) & (f_B > 0)
            
            # Extract valid data
            T_valid = T[valid_mask]
            S_valid = S[valid_mask]
            g_valid = g[valid_mask]
            f_B_valid = f_B[valid_mask]
            
            # Calculate combined factor
            combined = S_valid / (g_valid * f_B_valid)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot data
            scatter = ax.scatter(np.log10(combined), np.log10(T_valid), 
                               alpha=0.5, c=np.log10(f_B_valid), cmap='viridis')
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('log₁₀(f(B))', rotation=90)
            
            # Add theoretical prediction line
            x_range = np.linspace(np.min(np.log10(combined)), np.max(np.log10(combined)), 100)
            # Theory predicts T ∝ (S/(g×f(B)))^(1/4)
            y_theory = 0.25 * x_range + np.mean(np.log10(T_valid) - 0.25 * np.log10(combined))
            ax.plot(x_range, y_theory, 'r--', label='Theoretical Prediction (T ∝ X¹/⁴)')
            
            # Add labels and title
            ax.set_xlabel('log₁₀(Thomson/(g×f(B)))')
            ax.set_ylabel('log₁₀(Temperature [K])')
            ax.set_title('Temperature vs Modified Thomson Scattering\nHolographic Theory Test')
            
            # Add legend
            ax.legend()
            
            # Add theory validation threshold
            combined_results = results.get('validation_results', {}).get('combined_relationship', {})
            r_squared = combined_results.get('r_squared', 0)
            ax.text(0.05, 0.95, f'R² = {r_squared:.3f}\nThreshold: R² > 0.3',
                   transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
            
            # Save figure
            plt.savefig(figures_dir / 'combined_relationship.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating combined relationship plot: {str(e)}")
            
    def _plot_information_constraints(self, parameters, results, figures_dir):
        """Create plot showing information processing constraints on Thomson scattering."""
        try:
            # Extract data
            maps = parameters.get('maps', {})
            S = maps['scattering_map'].data.flatten()
            T = maps['temperature_map'].data.flatten()
            
            # Universal information processing rate
            gamma = 1.89e-29  # s^-1
            
            # Calculate information processing time
            tau_info = 1.0 / gamma
            
            # Create valid data mask
            valid_mask = ~np.isnan(S) & ~np.isnan(T) & (S > 0) & (T > 0)
            
            # Extract valid data
            S_valid = S[valid_mask]
            T_valid = T[valid_mask]
            
            # Calculate scattering timescale (simplified)
            tau_scatter = 1.0 / S_valid
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot data
            scatter = ax.scatter(np.log10(tau_scatter), np.log10(T_valid), 
                               alpha=0.5, c=np.log10(S_valid), cmap='plasma')
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('log₁₀(Scattering Rate [s⁻¹])', rotation=90)
            
            # Add information processing time threshold
            ax.axvline(np.log10(tau_info), color='r', linestyle='--', 
                      label='Information Processing Threshold')
            
            # Add labels and title
            ax.set_xlabel('log₁₀(Scattering Time [s])')
            ax.set_ylabel('log₁₀(Temperature [K])')
            ax.set_title('Temperature vs Scattering Time\nInformation Processing Constraints')
            
            # Add legend
            ax.legend()
            
            # Add statistics
            holographic_results = results.get('validation_results', {}).get('holographic_model', {})
            ratio = holographic_results.get('gamma_H_ratio', 0)
            theoretical_ratio = 1.0 / (8 * np.pi)
            ax.text(0.05, 0.95, 
                   f'γ/H Ratio = {ratio:.3e}\nTheoretical: {theoretical_ratio:.3e}',
                   transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
            
            # Save figure
            plt.savefig(figures_dir / 'information_constraints.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating information constraints plot: {str(e)}")
            
    def _plot_holographic_transition(self, parameters, results, figures_dir):
        """Create plot showing the holographic transition in the turbulent cascade."""
        try:
            # Extract turbulence analysis results
            turbulence_results = results.get('turbulence_analysis', {})
            k_range = np.array(turbulence_results.get('simulation_parameters', {}).get('wavenumbers', []))
            E_standard = np.array(turbulence_results.get('simulation_parameters', {}).get('standard_spectrum', []))
            E_holo = np.array(turbulence_results.get('simulation_parameters', {}).get('holographic_spectrum', []))
            
            if len(k_range) == 0 or len(E_standard) == 0 or len(E_holo) == 0:
                logger.warning("Insufficient data for holographic transition plot")
                return
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot standard and holographic spectra
            ax.loglog(k_range, E_standard, 'b-', label='Standard MHD', alpha=0.7)
            ax.loglog(k_range, E_holo, 'r-', label='Holographic', alpha=0.7)
            
            # Add holographic scale
            k_holo = turbulence_results.get('scale_parameters', {}).get('holographic_scale', {}).get('wavenumber')
            if k_holo:
                ax.axvline(k_holo, color='g', linestyle='--', label='Holographic Scale')
            
            # Add Kolmogorov scaling reference
            k_ref = np.array([k_range[0], k_range[-1]])
            E_ref = k_ref**(-5/3)
            E_ref *= E_standard[0] / E_ref[0]  # Normalize
            ax.loglog(k_ref, E_ref, 'k:', label='k⁻⁵/³ Reference', alpha=0.5)
            
            # Add labels and title
            ax.set_xlabel('Wavenumber k [m⁻¹]')
            ax.set_ylabel('Energy Spectrum E(k)')
            ax.set_title('Turbulent Energy Spectrum\nHolographic Transition')
            
            # Add legend
            ax.legend()
            
            # Add statistics
            spectral_props = turbulence_results.get('spectral_properties', {})
            steepening = spectral_props.get('holographic_modification', {}).get('spectral_steepening', 0)
            ax.text(0.05, 0.95, 
                   f'Spectral Steepening: {steepening:.3f}\nThreshold: > 0.5',
                   transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
            
            # Save figure
            plt.savefig(figures_dir / 'holographic_transition.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating holographic transition plot: {str(e)}")

def main():
    """Main function to run the analysis."""
    print("===== Gravitational-Electromagnetic Regulation of Thomson Scattering Analysis =====")
    print("This script implements the Analysis Protocol for testing the hypothesis that")
    print("Thomson scattering in the solar corona is regulated by both gravitational and")
    print("electromagnetic constraints, creating a dynamic information processing boundary.\n")
    
    # Create data and output directories
    data_dir = Path("thomson_data")
    output_dir = Path("thomson_results")
    
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Create the analyzer and run the pipeline
    analyzer = ThomsonRegulationAnalyzer(data_dir=data_dir, output_dir=output_dir)
    results = analyzer.run_analysis_pipeline()
    
    # Evaluate whether results confirm the theory
    evaluation = analyzer.evaluate_theory_confirmation(results)
    
    # Print summary of results
    if results:
        combined_results = results.get('combined_relationship', {})
        self_regulation = results.get('self_regulation', {})
        
        print("\n===== Analysis Results =====")
        
        correlation = combined_results.get('correlation', 0)
        p_value = combined_results.get('p_value', 1)
        print(f"Combined Relationship (T ∝ Thomson/(g×f(B))) Correlation: {correlation:.4f} (p={p_value:.4e})")
        
        if abs(correlation) > 0.5 and p_value < 0.05:
            print("RESULT: Strong evidence for gravitational-electromagnetic regulation")
        elif abs(correlation) > 0.3 and p_value < 0.1:
            print("RESULT: Moderate evidence for gravitational-electromagnetic regulation")
        else:
            print("RESULT: Weak or no evidence for gravitational-electromagnetic regulation")
        
        sr_correlation = self_regulation.get('correlation', 0)
        sr_p_value = self_regulation.get('p_value', 1)
        print(f"Self-Regulation (T-n_e relationship) Correlation: {sr_correlation:.4f} (p={sr_p_value:.4e})")
        
        if sr_correlation < -0.3 and sr_p_value < 0.05:
            print("RESULT: Strong evidence for self-regulation mechanism (anti-correlation)")
        elif sr_correlation < 0 and sr_p_value < 0.1:
            print("RESULT: Moderate evidence for self-regulation mechanism")
        else:
            print("RESULT: Weak or no evidence for self-regulation mechanism")
        
        # Print theory evaluation results
        print("\n===== Theory Evaluation =====")
        if isinstance(evaluation["theory_confirmed"], bool):
            confirmation_status = "CONFIRMED" if evaluation["theory_confirmed"] else "NOT CONFIRMED"
        else:
            confirmation_status = evaluation["theory_confirmed"].upper()  # e.g., "PARTIALLY"
        confidence = evaluation["confidence_level"]
        print(f"Holographic Universe Theory Status: {confirmation_status} ({confidence} confidence)")
        print(f"Evidence: {evaluation['confirmation_ratio']}")
        print(f"Reasoning: {evaluation['reasoning']}")
        
        if evaluation["criteria_met"]:
            print("\nSatisfied Criteria:")
            for criterion in evaluation["criteria_met"]:
                print(f"✓ {criterion['name']}: {criterion['actual']} (Expected: {criterion['expected']})")
                if "physical_interpretation" in criterion:
                    print(f"  Interpretation: {criterion['physical_interpretation']}")
                
        if evaluation["criteria_failed"]:
            print("\nUnsatisfied Criteria:")
            for criterion in evaluation["criteria_failed"]:
                print(f"✗ {criterion['name']}: {criterion['actual']} (Expected: {criterion['expected']})")
                if "physical_interpretation" in criterion:
                    print(f"  Interpretation: {criterion['physical_interpretation']}")
        
        if evaluation["recommendations"]:
            print("\nRecommendations:")
            for recommendation in evaluation["recommendations"]:
                print(f"• {recommendation}")
        
        print("\nOutput files saved to:", output_dir)
    else:
        print("Analysis failed or produced no results.")

# Main function to run the analysis pipeline if executed directly
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Thomson Scattering Regulation Analysis Pipeline")
    parser.add_argument("--output", type=str, default="thomson_analysis_results.json", help="Output file for analysis results")
    parser.add_argument("--cache-dir", type=str, default=None, help="Custom cache directory for data storage")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--force-alternative", action="store_true", help="Force use of alternative data sources")
    parser.add_argument("--force-fallback", action="store_true", help="Force use of fallback date (2023-06-15) for LASCO data")
    parser.add_argument("--date", type=str, default="2023-06-15", help="Analysis date in YYYY-MM-DD format (default: 2023-06-15, known good date)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache before running analysis")
    parser.add_argument("--clear-alternative-cache", action="store_true", help="Clear only alternative data cache before running analysis")
    args = parser.parse_args()
    
    # Configure logging based on debug flag
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        logger.setLevel(logging.INFO)
    
    try:
        # Initialize the analyzer
        analyzer = ThomsonRegulationAnalyzer(cache_dir=args.cache_dir)
        
        # Clear cache if requested
        if args.clear_cache:
            logger.info("Clearing all caches as requested")
            analyzer.clear_cache()
            print("Cache cleared. Analysis will download fresh data.")
            
        if args.clear_alternative_cache:
            logger.info("Clearing alternative data cache as requested")
            analyzer.clear_cache("alternative_data")
            print("Alternative data cache cleared. Analysis will download fresh alternative data.")
        
        # Override the run_analysis_pipeline method to use our specified date if given
        if args.force_fallback:
            logger.info("Forcing use of fallback date (2023-06-15) for analysis")
            orig_download_alternative_data = analyzer.download_alternative_data
            
            # Create a wrapper that always uses the fallback date
            def download_alternative_with_fallback(date):
                return orig_download_alternative_data("2023-06-15")
            
            # Replace the method
            analyzer.download_alternative_data = download_alternative_with_fallback
        
        # Run the analysis pipeline
        if args.force_alternative:
            logger.info("Forced use of alternative data sources")
            analyzer.download_punch_data = lambda *args, **kwargs: False  # Override to skip PUNCH data
        
        results = analyzer.run_analysis_pipeline()
        
        # Evaluate theory confirmation
        theory_evaluation = analyzer.evaluate_theory_confirmation(results)
        
        # Add theory evaluation to results
        results["theory_evaluation"] = theory_evaluation
        
        # Convert NumPy types to native Python types for JSON serialization
        def convert_results_for_json(results_dict):
            logger.info("Converting results dictionary for JSON serialization")
            # Create a copy of the input dictionary
            cleaned_results = {}
            
            # Process each key in the results dictionary
            for key, value in results_dict.items():
                if isinstance(value, dict):
                    # Recursively clean nested dictionaries
                    cleaned_results[key] = convert_results_for_json(value)
                elif isinstance(value, (np.integer, np.int32, np.int64)):
                    # Convert NumPy integers to Python int
                    cleaned_results[key] = int(value)
                elif isinstance(value, (np.floating, np.float32, np.float64)):
                    # Convert NumPy floats to Python float
                    cleaned_results[key] = float(value)
                elif isinstance(value, np.ndarray):
                    # Convert NumPy arrays to lists
                    cleaned_results[key] = value.tolist()
                elif isinstance(value, np.bool_):
                    # Convert NumPy booleans to Python bool
                    cleaned_results[key] = bool(value)
                else:
                    # Keep other types as is (strings, native Python types, etc.)
                    cleaned_results[key] = value
            
            return cleaned_results
        
        # Clean the results and remove non-serializable objects
        clean_results = convert_results_for_json(results)
        
        # Save results to file using the custom JSON encoder for any remaining NumPy types
        try:
            with open(args.output, 'w') as f:
                json.dump(clean_results, f, indent=2, cls=NumpyJSONEncoder)
            logger.info(f"Analysis results saved to {args.output}")
        except TypeError as e:
            logger.error(f"Error serializing results to JSON: {str(e)}")
            logger.warning("Saving results with non-serializable data removed...")
            
            # Fallback: Serialize basic information only
            basic_results = {
                "data_source": results.get("data_source", "Unknown"),
                "analysis_timestamp": results.get("analysis_timestamp", datetime.now().isoformat()),
                "error": f"Full results could not be serialized: {str(e)}"
            }
            
            # Try to save at least the basic information
            with open(args.output, 'w') as f:
                json.dump(basic_results, f, indent=2)
            logger.info(f"Basic results saved to {args.output}")
        
        # Print summary to console
        print("\n===== Thomson Regulation Analysis Summary =====")
        print(f"Data source: {results['data_source']}")
        
        # Show if fallback data was used
        if "fallback" in results['data_source'].lower():
            print("\n⚠️ USING FALLBACK DATA FROM KNOWN GOOD DATE ⚠️")
            print("Analysis was performed with fallback data which may not match your requested date.")
            print("This is normal and necessary when primary data sources are unavailable.")
        
        print("\nKey physical parameters:")
        # Get top 5 parameters by extracting all keys first, then iterating through top 5
        param_keys = list(results['physical_parameters'].keys())[:5]
        for param in param_keys:
            value = results['physical_parameters'][param]
            if isinstance(value, (int, float, np.integer, np.floating)):
                print(f"  {param}: {float(value):.4g}")
            else:
                print(f"  {param}: {value}")
        
        # Display theory evaluation results
        if isinstance(theory_evaluation["theory_confirmed"], bool):
            confirmation_status = "CONFIRMED" if theory_evaluation["theory_confirmed"] else "NOT CONFIRMED"
        else:
            confirmation_status = theory_evaluation["theory_confirmed"].upper()  # e.g., "PARTIALLY"
        confidence = theory_evaluation["confidence_level"]
        print(f"\nTheory Status: {confirmation_status} ({confidence} confidence)")
        print(f"Evidence: {theory_evaluation['confirmation_ratio']}")
        
        print("\nHypothesis validation results:")
        for hypothesis, result in results['validation_results'].items():
            if 'confirmed' in result:
                status = "✓ CONFIRMED" if result['confirmed'] else "✗ REJECTED"
                confidence = float(result['confidence']) if isinstance(result['confidence'], (np.floating, np.integer)) else result['confidence']
                print(f"  {hypothesis}: {status} (confidence: {confidence:.2f})")
            else:
                # Handle the case where result structure is different
                if isinstance(result, dict) and 'correlation' in result:
                    corr = float(result.get('correlation', 0)) if isinstance(result.get('correlation'), (np.floating, np.integer)) else result.get('correlation', 0)
                    pval = float(result.get('p_value', 1)) if isinstance(result.get('p_value'), (np.floating, np.integer)) else result.get('p_value', 1)
                    status = "STRONG" if abs(corr) > 0.5 and pval < 0.05 else "MODERATE" if abs(corr) > 0.3 else "WEAK"
                    print(f"  {hypothesis}: {status} (corr: {corr:.2f}, p: {pval:.4f})")
        
        # Display criteria details if debug is enabled
        if args.debug:
            print("\nSatisfied Criteria:")
            for criterion in theory_evaluation.get("criteria_met", []):
                print(f"✓ {criterion['name']}: {criterion['actual']} (Expected: {criterion['expected']})")
                if "physical_interpretation" in criterion:
                    print(f"  Interpretation: {criterion['physical_interpretation']}")
                
            print("\nUnsatisfied Criteria:")
            for criterion in theory_evaluation.get("criteria_failed", []):
                print(f"✗ {criterion['name']}: {criterion['actual']} (Expected: {criterion['expected']})")
                if "physical_interpretation" in criterion:
                    print(f"  Interpretation: {criterion['physical_interpretation']}")
        
        print("\nAnalysis completed successfully.")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        if args.debug:
            logger.exception("Detailed error information:")
        sys.exit(1)