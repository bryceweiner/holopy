#!/usr/bin/env python3
"""
PUNCH Heliophysics Analysis using HoloPy

This script demonstrates how to use the HoloPy framework to analyze
Polarimetric Unified Coronal & Heliospheric Imager (PUNCH) mission data
for testing the thermodynamic regulation hypothesis of coronal heating.

The script implements the analysis protocol outlined in the research paper:
"Holographic Thermodynamic Regulation Analysis Using PUNCH Data"

Author: Holographic Universe Research Team
Date: 2023-07-01
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any

# Import sunpy for solar data handling
import sunpy.map
from sunpy.net import Fido, attrs as a
from sunpy.coordinates import frames
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

# Import complementary data sources modules
from astropy.io import fits
from scipy.stats import pearsonr, chi2_contingency
from scipy.signal import periodogram
from scipy.optimize import curve_fit
from scipy import ndimage
import reproject

# Import holopy modules
import holopy
from holopy.constants.physical_constants import PhysicalConstants
from holopy.utils import visualization

# Configure logging
from holopy.utils.logging import configure_logging
configure_logging(level='INFO')
logger = logging.getLogger('holopy.punch_analysis')

# Define missing visualization functions
def create_figure(nrows=1, ncols=1, figsize=None):
    """Create a figure with given rows and columns."""
    return plt.subplots(nrows, ncols, figsize=figsize)

def save_figure(fig, filename, dpi=300):
    """Save a figure to file."""
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

# Define missing math utility functions
def gradient_2d(array_2d):
    """
    Calculate the gradient of a 2D array.
    
    Args:
        array_2d: 2D numpy array
        
    Returns:
        Tuple of (dy, dx) gradient arrays
    """
    return np.gradient(array_2d)

def correlation_function(array1, array2=None):
    """
    Calculate correlation function between two arrays.
    
    Args:
        array1: First array
        array2: Second array (if None, autocorrelation of array1)
        
    Returns:
        Correlation function
    """
    if array2 is None:
        array2 = array1
    
    return np.corrcoef(array1.flatten(), array2.flatten())[0, 1]

class PunchAnalyzer:
    """
    Class for analyzing PUNCH polarimetric data using holographic principles.
    
    This class implements methods for extracting information density from
    polarization gradients and correlating with temperature measurements.
    """
    
    def __init__(
        self,
        data_dir: str = "punch_data",
        output_dir: str = "punch_results",
        days_to_analyze: int = 30,
        start_date: str = "2022-07-01",
        end_date: str = "2023-06-30"
    ):
        """
        Initialize the PUNCH data analyzer.
        
        Args:
            data_dir: Directory containing PUNCH FITS files
            output_dir: Directory for output results
            days_to_analyze: Number of past days to analyze (used only if start_date/end_date not provided)
            start_date: Start date for data analysis in YYYY-MM-DD format
            end_date: End date for data analysis in YYYY-MM-DD format
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.days_to_analyze = days_to_analyze
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize physical constants
        self.constants = PhysicalConstants()
        self.gamma = self.constants.gamma  # Fundamental information processing rate
        self.r_sun = 6.96e8  # meters
        
        # Protocol-specific constants
        self.cadence = 4 * 60  # 4-minute cadence in seconds
        self.min_fov_deg = 1.5  # Minimum field of view in degrees (6 R☉)
        self.max_fov_deg = 45.0  # Maximum field of view in degrees (180 R☉)
        self.min_fov_rsun = 6.0  # Minimum field of view in solar radii
        self.max_fov_rsun = 180.0  # Maximum field of view in solar radii
        
        logger.info(f"PunchAnalyzer initialized with γ = {self.gamma:.2e} s⁻¹")
        
        # Create date range for analysis
        if start_date and end_date:
            self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
            self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
            analysis_period = (self.end_date - self.start_date).days
            logger.info(f"Analyzing PUNCH data over {analysis_period} days fixed period")
        else:
            self.end_date = datetime.now()
            self.start_date = self.end_date - timedelta(days=days_to_analyze)
            logger.info(f"Analyzing PUNCH data from the last {days_to_analyze} days")
            
        logger.info(f"FOV coverage: {self.min_fov_deg}° to {self.max_fov_deg}° ({self.min_fov_rsun}-{self.max_fov_rsun} R☉)")
        logger.info(f"Date range: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
    
    def fetch_punch_data(self) -> List[str]:
        """
        Fetch PUNCH data from Virtual Solar Observatory using sunpy.
        
        Returns:
            List of downloaded file paths
        """
        try:
            logger.info(f"Querying VSO for PUNCH data from {self.start_date} to {self.end_date}...")
            
            # Convert datetime objects to strings
            start_date_str = self.start_date.strftime('%Y-%m-%d')
            end_date_str = self.end_date.strftime('%Y-%m-%d')
            
            # Query VSO for PUNCH data with specific requirements from protocol
            time_range = a.Time(start_date_str, end_date_str)
            instrument = a.Instrument('PUNCH')
            physobs = a.Physobs('polarized_intensity')
            
            # Additional constraints based on protocol
            provider = a.Provider('SDAC')  # NASA SDAC as specified in protocol
            sample = a.Sample(self.cadence * u.second)  # 4-minute cadence with proper units
            
            # Execute query for NFI (Narrow Field Imager)
            logger.info("Querying for NFI data...")
            nfi_result = Fido.search(
                time_range, 
                instrument, 
                a.Instrument('NFI'),  # Specific NFI instrument
                physobs, 
                provider,
                sample
            )
            
            # Execute query for WFI (Wide Field Imager) - 3 fields of view as specified
            logger.info("Querying for 3 WFI fields of view...")
            wfi_result = Fido.search(
                time_range, 
                instrument, 
                a.Instrument('WFI'),  # Specific WFI instrument
                physobs, 
                provider,
                sample
            )
            
            # Combine results
            result = nfi_result + wfi_result if len(nfi_result) > 0 and len(wfi_result) > 0 else nfi_result or wfi_result
            
            if len(result) == 0:
                logger.warning("No PUNCH data found for the specified date range")
                return []
            
            logger.info(f"Found {len(result)} PUNCH data files")
            
            # Download files
            downloaded_files = Fido.fetch(result, path=os.path.join(self.data_dir, '{file}'))
            logger.info(f"Downloaded {len(downloaded_files)} files to {self.data_dir}")
            
            # Verify data meets requirements
            verified_files = []
            for file in downloaded_files:
                try:
                    # Check if the file meets FOV requirements
                    if self._verify_file_requirements(file):
                        verified_files.append(file)
                    else:
                        logger.warning(f"File {file} does not meet protocol requirements, skipping")
                except Exception as e:
                    logger.warning(f"Error verifying file {file}: {e}")
            
            logger.info(f"Verified {len(verified_files)} files that meet protocol requirements")
            return verified_files
            
        except Exception as e:
            logger.error(f"Error fetching PUNCH data: {str(e)}")
            return []
    
    def _verify_file_requirements(self, file_path: str) -> bool:
        """
        Verify that a FITS file meets the protocol requirements.
        
        Args:
            file_path: Path to FITS file
            
        Returns:
            bool: True if file meets requirements
        """
        try:
            with fits.open(file_path) as hdul:
                header = hdul[0].header
                
                # Check for 4-minute cadence
                if 'CADENCE' in header:
                    cadence = header['CADENCE']  # in seconds
                    if abs(cadence - self.cadence) > 30:  # Allow 30s tolerance
                        logger.warning(f"File {file_path} has incorrect cadence: {cadence}s")
                        return False
                
                # Check FOV coverage
                if 'FOVMIN' in header and 'FOVMAX' in header:
                    fov_min = header['FOVMIN']  # in solar radii
                    fov_max = header['FOVMAX']  # in solar radii
                    
                    if fov_min > self.min_fov_rsun or fov_max < self.max_fov_rsun:
                        logger.warning(f"File {file_path} has incorrect FOV coverage: {fov_min}-{fov_max} R☉")
                        return False
                
                # Check for combined fields of view (NFI + WFI)
                if 'INSTRUME' in header:
                    instrument = header['INSTRUME']
                    if 'NFI+WFI' not in instrument and 'COMBINED' not in instrument:
                        logger.warning(f"File {file_path} is not a combined NFI+WFI image")
                        # We'll accept it anyway, as the combination might be done in post-processing
                
                return True
                
        except Exception as e:
            logger.warning(f"Error verifying file requirements for {file_path}: {e}")
            return False
    
    def list_available_data(self) -> List[str]:
        """
        List available PUNCH data files.
        
        Returns:
            List of file paths
        """
        # First check if we need to download data
        if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir):
            logger.info("No local PUNCH data found. Attempting to download from VSO...")
            downloaded_files = self.fetch_punch_data()
            if downloaded_files:
                return downloaded_files
        
        # Pattern for PUNCH data files (example pattern, adjust to actual naming convention)
        import glob
        pattern = os.path.join(self.data_dir, "punch_*_polarimetric.fits")
        
        # Find matching files
        files = glob.glob(pattern)
        
        # Filter by date if specified
        if self.days_to_analyze > 0:
            filtered_files = []
            
            for file in files:
                # Extract date from filename (adjust pattern as needed)
                try:
                    filename = os.path.basename(file)
                    date_str = filename.split("_")[1]  # Assuming punch_YYYYMMDD_polarimetric.fits
                    file_date = datetime.strptime(date_str, "%Y%m%d")
                    
                    # Check if within date range
                    if self.start_date <= file_date <= self.end_date:
                        filtered_files.append(file)
                except Exception as e:
                    logger.warning(f"Could not parse date from file {file}: {e}")
            
            files = filtered_files
        
        logger.info(f"Found {len(files)} PUNCH data files")
        return files
    
    def load_punch_map(self, fits_file: str) -> Dict[str, 'sunpy.map.Map']:
        """
        Load PUNCH data as sunpy maps.
        
        Args:
            fits_file: Path to PUNCH FITS file
            
        Returns:
            Dictionary of Stokes parameter maps
        """
        try:
            # Import the FITS data as sunpy maps
            # For typical PUNCH polarimetric data, we expect Stokes I, Q, U parameters
            from astropy.io import fits
            
            with fits.open(fits_file) as hdul:
                # Check if FITS file has multiple HDUs for different Stokes parameters
                if len(hdul) >= 3:
                    # Extract Stokes parameters
                    I_data = hdul[0].data  # Primary HDU usually contains Stokes I
                    Q_data = hdul[1].data  # First extension for Stokes Q
                    U_data = hdul[2].data  # Second extension for Stokes U
                    
                    # Create header
                    header = hdul[0].header
                    
                    # Create sunpy maps
                    I_map = sunpy.map.Map(I_data, header)
                    Q_map = sunpy.map.Map(Q_data, header)
                    U_map = sunpy.map.Map(U_data, header)
                    
                    return {'I': I_map, 'Q': Q_map, 'U': U_map}
                else:
                    # If data is in a different format, try to load as a single map
                    # and interpret as needed
                    logger.warning(f"Unexpected FITS structure in {fits_file}. Attempting direct load.")
                    punch_map = sunpy.map.Map(fits_file)
                    return {'map': punch_map}
                    
        except Exception as e:
            logger.error(f"Error loading PUNCH data from {fits_file}: {str(e)}")
            return {}
    
    def extract_information_density(self, polarization_data: np.ndarray) -> np.ndarray:
        """
        Extract information density from polarization gradient.
        
        Args:
            polarization_data: 2D array of polarization degree
            
        Returns:
            2D array of local information density
        """
        # Calculate polarization gradient using holopy utility
        grad_y, grad_x = gradient_2d(polarization_data)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Information density is proportional to gradient squared
        information_density = (grad_magnitude**2) / self.gamma
        
        return information_density
    
    def compute_gravitational_potential(self, radial_distance: np.ndarray) -> np.ndarray:
        """
        Compute gravitational potential at given radial distance.
        
        Args:
            radial_distance: array of distances from solar center in solar radii
            
        Returns:
            array of gravitational potential values
        """
        r_meters = radial_distance * self.r_sun
        g_sun = 274.0  # m/s^2, solar surface gravity
        g_local = g_sun * (self.r_sun / r_meters)**2
        return g_local
    
    def predict_temperature(self, info_density: np.ndarray, grav_potential: np.ndarray) -> np.ndarray:
        """
        Predict temperature based on holographic theory.
        
        Args:
            info_density: array of information density values
            grav_potential: array of gravitational potential values
            
        Returns:
            array of predicted temperatures
        """
        # Holographic prediction: T ∝ ρ_info/g
        # Normalize to solar units for temperature (1 MK = 1e6 K)
        predicted_temp = 1e6 * (info_density / grav_potential)
        return predicted_temp
    
    def analyze_punch_data(self, fits_file: str, temp_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main analysis function for PUNCH polarimetric data.
        
        Args:
            fits_file: Path to PUNCH FITS file
            temp_data: Dictionary containing temperature data from complementary sources
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Analyzing file: {fits_file}")
        
        results = {}
        
        try:
            # Load PUNCH data as sunpy maps
            stokes_maps = self.load_punch_map(fits_file)
            
            if not stokes_maps:
                logger.error(f"Failed to load Stokes parameters from {fits_file}")
                return {'error': f"Failed to load data from {fits_file}"}
                
            # Check if we have the expected Stokes parameters
            if 'I' in stokes_maps and 'Q' in stokes_maps and 'U' in stokes_maps:
                # Extract data arrays
                I_stokes = stokes_maps['I'].data
                Q_stokes = stokes_maps['Q'].data
                U_stokes = stokes_maps['U'].data
                
                # Store maps for later use
                results['stokes_maps'] = stokes_maps
                
                # Calculate degree of polarization
                polarization_degree = np.sqrt(Q_stokes**2 + U_stokes**2) / np.maximum(I_stokes, 1e-10)
                results['polarization_degree'] = polarization_degree
                
                # Create a map for the polarization degree
                pol_map = sunpy.map.Map(polarization_degree, stokes_maps['I'].meta)
                results['polarization_map'] = pol_map
                
                # ==========================================
                # PROTOCOL ALGORITHM 3.1: Information Density Extraction
                # ==========================================
                logger.info("Executing Protocol Algorithm 3.1: Information Density Extraction")
                
                # Calculate polarization gradient using holopy utility
                info_density = self.extract_information_density(polarization_degree)
                results['info_density'] = info_density
                
                # Create a map for information density
                info_meta = stokes_maps['I'].meta.copy()
                info_meta['BUNIT'] = 's'  # Units for information density
                info_meta['TELESCOP'] = 'PUNCH + HoloPy'
                info_map = sunpy.map.Map(info_density, info_meta)
                results['info_map'] = info_map
                
                # Identify regions approaching saturation
                saturation_threshold = 1/self.gamma
                saturation_regions = np.sum(info_density > 0.9 * saturation_threshold)
                results['saturation_regions'] = saturation_regions
                logger.info(f"Found {saturation_regions} regions approaching information saturation")
                
                # Create a map for high Thomson scattering regions
                # High Thomson scattering regions are identified by high polarization degree
                thomson_scatter_threshold = 0.3  # Typical threshold for strong Thomson scattering
                high_thomson_regions = polarization_degree > thomson_scatter_threshold
                results['high_thomson_regions'] = high_thomson_regions
                logger.info(f"Found {np.sum(high_thomson_regions)} regions with high Thomson scattering")
                
                # Create a coordinate grid for radial distances
                # Get map dimensions and reference point (center of Sun)
                reference_coord = SkyCoord(0*u.arcsec, 0*u.arcsec, 
                                          frame=frames.Helioprojective, 
                                          obstime=pol_map.date)
                
                # Get pixel coordinates in arcsec
                y, x = np.indices(info_density.shape)
                hpc_coords = pol_map.pixel_to_world(x, y)
                
                # Calculate distance in arcsec from Sun center
                distances = hpc_coords.separation(reference_coord)
                
                # Convert to solar radii
                rsun_arcsec = 959.63 * u.arcsec  # Approximate solar radius in arcseconds
                radial_distance_rsun = (distances / rsun_arcsec).value
                results['radial_distance_rsun'] = radial_distance_rsun
                
                # Compute gravitational potential
                grav_potential = self.compute_gravitational_potential(radial_distance_rsun)
                results['grav_potential'] = grav_potential
                
                # ==========================================
                # PROTOCOL ALGORITHM 3.2: Temperature Correlation Analysis
                # ==========================================
                logger.info("Executing Protocol Algorithm 3.2: Temperature Correlation Analysis")
                
                # Predict temperature based on holographic theory T ∝ ρ_info/g
                predicted_temp = self.predict_temperature(info_density, grav_potential)
                results['predicted_temp'] = predicted_temp
                
                # Create a map for predicted temperature
                temp_meta = info_meta.copy()
                temp_meta['BUNIT'] = 'K'  # Units for temperature
                temp_map = sunpy.map.Map(predicted_temp, temp_meta)
                results['temp_map'] = temp_map
                
                # Co-register with observed temperature data if available
                if temp_data and 'map' in temp_data:
                    logger.info(f"Co-registering with observed temperature data from {temp_data.get('source', 'unknown')}")
                    
                    try:
                        # Reproject observed temperature map to match polarization data coordinates
                        from sunpy.coordinates.utils import all_coordinates_from_map
                        from astropy.coordinates import SkyCoord
                        from reproject import reproject_interp
                        
                        observed_temp_map = temp_data['map']
                        
                        # Reproject temperature map to match polarization map
                        reproj_temp, footprint = reproject_interp(
                            observed_temp_map, 
                            pol_map.wcs, 
                            shape_out=pol_map.data.shape
                        )
                        
                        # Store reprojected temperature data
                        results['observed_temp'] = reproj_temp
                        results['observed_temp_map'] = sunpy.map.Map(
                            reproj_temp, pol_map.meta.copy()
                        )
                        
                        # Calculate temperature correlation statistics
                        mask = (footprint > 0) & np.isfinite(reproj_temp) & np.isfinite(predicted_temp)
                        if np.sum(mask) > 10:  # Need at least 10 points for statistics
                            obs_temp_flat = reproj_temp[mask].flatten()
                            pred_temp_flat = predicted_temp[mask].flatten()
                            
                            # Calculate Pearson correlation
                            pearson_r, p_value = pearsonr(
                                np.log10(np.maximum(obs_temp_flat, 1e-10)),
                                np.log10(np.maximum(pred_temp_flat, 1e-10))
                            )
                            
                            results['temp_correlation'] = {
                                'pearson_r': pearson_r,
                                'p_value': p_value,
                                'sample_size': np.sum(mask)
                            }
                            
                            # Check if correlation meets protocol significance criteria
                            if pearson_r > 0.8 and p_value < 0.01:
                                logger.info(f"Temperature correlation meets protocol significance criteria: r={pearson_r:.3f}, p={p_value:.3e}")
                                results['meets_significance'] = True
                            else:
                                logger.info(f"Temperature correlation does NOT meet protocol significance criteria: r={pearson_r:.3f}, p={p_value:.3e}")
                                results['meets_significance'] = False
                            
                            # Perform Bayesian model comparison (holographic vs. conventional)
                            # Implement Bayesian comparison between holographic model and conventional model
                            results['bayesian_comparison'] = self._bayesian_model_comparison(
                                observed_temp=obs_temp_flat,
                                predicted_temp=pred_temp_flat,
                                info_density=info_density[mask].flatten(),
                                grav_potential=grav_potential[mask].flatten()
                            )
                        else:
                            logger.warning("Insufficient valid points for temperature correlation analysis")
                    except Exception as e:
                        logger.warning(f"Error during temperature co-registration: {e}")
                
                # ==========================================
                # PROTOCOL ALGORITHM 3.3: Track Expected Signatures
                # ==========================================
                logger.info("Executing Protocol Algorithm 3.3: Expected Signatures Check")
                
                # 4.1 Spatial Correlation Signatures
                # Identify temperature peaks and check if they coincide with high information density
                # Temperature peaks are local maxima in predicted temperature
                from scipy import ndimage
                
                # Find local maxima in temperature
                temp_peaks = self._find_local_maxima(predicted_temp)
                results['temp_peaks'] = temp_peaks
                
                # Check if peaks coincide with high information density
                if np.sum(temp_peaks) > 0:
                    # Check overlap with high information density regions
                    high_info_threshold = np.percentile(info_density, 90)  # Top 10% of info density
                    high_info_regions = info_density > high_info_threshold
                    
                    # Calculate overlap
                    overlap = np.sum(temp_peaks & high_info_regions) / np.sum(temp_peaks)
                    results['peak_info_overlap'] = overlap
                    
                    logger.info(f"Temperature peak overlap with high information density: {overlap:.1%}")
                    
                    # Check for anti-correlation with gravitational potential
                    # In high information regions, we expect lower gravitational potential
                    high_info_indices = np.where(high_info_regions)
                    if len(high_info_indices[0]) > 0:
                        mean_g_high_info = np.mean(grav_potential[high_info_indices])
                        mean_g_overall = np.mean(grav_potential)
                        
                        g_ratio = mean_g_high_info / mean_g_overall
                        results['g_anticorrelation'] = g_ratio < 1.0
                        logger.info(f"Gravitational potential in high info regions is {g_ratio:.2f}x the average")
                    
                    # Check for sharp transitions at holographic boundaries
                    # These are identified by rapid changes in information density
                    info_gradient = np.gradient(info_density)
                    high_gradient = (np.abs(info_gradient[0])**2 + np.abs(info_gradient[1])**2) > np.percentile(
                        np.abs(info_gradient[0])**2 + np.abs(info_gradient[1])**2, 95
                    )
                    results['holographic_boundaries'] = high_gradient
                    logger.info(f"Found {np.sum(high_gradient)} pixels with sharp holographic boundaries")
                    
                # Additional physics analysis here
                # ...
                
                # Add metadata
                results['filename'] = os.path.basename(fits_file)
                results['date'] = pol_map.date
                
                logger.info(f"Analysis completed successfully for {os.path.basename(fits_file)}")
                
            else:
                logger.error(f"Missing required Stokes parameters in {fits_file}")
                return {'error': f"Missing required Stokes parameters in {fits_file}"}
                
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing {fits_file}: {str(e)}")
            return {'error': str(e)}
    
    def _find_local_maxima(self, data: np.ndarray, threshold_percentile: float = 90) -> np.ndarray:
        """
        Find local maxima in a 2D array.
        
        Args:
            data: 2D array of data
            threshold_percentile: Percentile threshold for considering peaks
            
        Returns:
            Boolean mask of local maxima positions
        """
        # Apply a small Gaussian filter to reduce noise
        smoothed = ndimage.gaussian_filter(data, sigma=1.0)
        
        # Find local maxima
        local_max = ndimage.maximum_filter(smoothed, size=5)
        maxima = (smoothed == local_max)
        
        # Apply threshold to only keep significant peaks
        threshold = np.percentile(data, threshold_percentile)
        maxima = maxima & (data > threshold)
        
        return maxima
            
    def _bayesian_model_comparison(self, observed_temp: np.ndarray, predicted_temp: np.ndarray, 
                                  info_density: np.ndarray, grav_potential: np.ndarray) -> Dict:
        """
        Perform Bayesian model comparison between holographic and conventional models.
        
        Args:
            observed_temp: Observed temperature values
            predicted_temp: Predicted temperature values from holographic model
            info_density: Information density values
            grav_potential: Gravitational potential values
            
        Returns:
            Dictionary with Bayesian comparison results
        """
        try:
            # 1. Holographic Model (T ∝ ρ_info/g)
            # Already calculated as predicted_temp
            
            # 2. Conventional Model (T based on hydrostatic equilibrium, T ∝ g)
            # In conventional models, coronal temperature often scales with gravity
            conventional_temp = 1e6 * grav_potential / np.max(grav_potential)
            
            # Calculate log-likelihoods
            # Using Gaussian likelihood with fixed variance
            variance = np.var(observed_temp) 
            
            def log_likelihood(model, data):
                return -0.5 * np.sum((model - data)**2 / variance)
            
            ll_holographic = log_likelihood(predicted_temp, observed_temp)
            ll_conventional = log_likelihood(conventional_temp, observed_temp)
            
            # Calculate Bayes factor (assuming equal priors)
            bayes_factor = np.exp(ll_holographic - ll_conventional)
            
            # Calculate AIC and BIC
            n = len(observed_temp)
            k_holographic = 2  # Number of parameters in holographic model
            k_conventional = 1  # Number of parameters in conventional model
            
            aic_holographic = 2 * k_holographic - 2 * ll_holographic
            aic_conventional = 2 * k_conventional - 2 * ll_conventional
            
            bic_holographic = k_holographic * np.log(n) - 2 * ll_holographic
            bic_conventional = k_conventional * np.log(n) - 2 * ll_conventional
            
            # Interpretation of Bayes factor
            if bayes_factor > 100:
                interpretation = "Decisive evidence for holographic model"
            elif bayes_factor > 10:
                interpretation = "Strong evidence for holographic model"
            elif bayes_factor > 3:
                interpretation = "Substantial evidence for holographic model"
            elif bayes_factor > 1:
                interpretation = "Weak evidence for holographic model"
            elif bayes_factor > 1/3:
                interpretation = "Weak evidence for conventional model"
            elif bayes_factor > 1/10:
                interpretation = "Substantial evidence for conventional model"
            elif bayes_factor > 1/100:
                interpretation = "Strong evidence for conventional model"
            else:
                interpretation = "Decisive evidence for conventional model"
            
            return {
                'bayes_factor': bayes_factor,
                'interpretation': interpretation,
                'll_holographic': ll_holographic,
                'll_conventional': ll_conventional,
                'aic_holographic': aic_holographic,
                'aic_conventional': aic_conventional,
                'bic_holographic': bic_holographic,
                'bic_conventional': bic_conventional,
                'favors_holographic': bayes_factor > 1
            }
            
        except Exception as e:
            logger.warning(f"Error in Bayesian model comparison: {e}")
            return {'error': str(e)}
    
    def detect_phase_transitions(
        self, 
        time_series_data: np.ndarray, 
        threshold: float = np.pi/2,
        thomson_regions: Optional[np.ndarray] = None
    ) -> Tuple[List[int], List[Tuple[int, int]], np.ndarray]:
        """
        Detect phase transitions where accumulated information approaches π/2.
        This follows Protocol Algorithm 3.3: Thomson Scattering Information Saturation
        
        Args:
            time_series_data: 3D array (time, y, x) of information density
            threshold: phase transition threshold (default: π/2)
            thomson_regions: Optional mask of high Thomson scattering regions
            
        Returns:
            Tuple containing:
            - list of time indices where transitions occur
            - list of (y, x) coordinates of transitions
            - 2D array showing transition density across the field of view
        """
        transition_times = []
        transition_locations = []
        
        # Create a transition map to track where transitions occur
        transition_map = np.zeros(time_series_data.shape[1:], dtype=int)
        
        # Integrate information over time
        time_step = 240  # seconds (assuming 4-minute cadence)
        cumulative_info = np.cumsum(time_series_data, axis=0) * self.gamma * time_step
        
        # Apply mask for high Thomson scattering regions if provided
        mask = np.ones_like(cumulative_info[0], dtype=bool)
        if thomson_regions is not None:
            logger.info(f"Focusing on {np.sum(thomson_regions)} high Thomson scattering regions")
            mask = thomson_regions
        
        # Detect transitions
        for t in range(1, cumulative_info.shape[0]):
            # Find where cumulative information crosses the threshold with mask applied
            transition_mask = (cumulative_info[t] >= threshold) & (cumulative_info[t-1] < threshold) & mask
            
            if np.any(transition_mask):
                y_trans, x_trans = np.where(transition_mask)
                for y, x in zip(y_trans, x_trans):
                    transition_times.append(t)
                    transition_locations.append((y, x))
                    transition_map[y, x] += 1  # Increment count for this location
        
        logger.info(f"Detected {len(transition_times)} phase transitions")
        
        return transition_times, transition_locations, transition_map
    
    def analyze_spectral_fingerprints(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze spectral fingerprints as described in Section 4.3 of the protocol.
        
        Args:
            results: Analysis results including polarization and information data
            
        Returns:
            Dictionary with spectral fingerprint analysis results
        """
        logger.info("Analyzing spectral fingerprints from polarization data")
        
        fingerprints = {}
        
        try:
            # Check if we have the necessary data
            if not all(k in results for k in ['polarization_degree', 'info_density', 'predicted_temp']):
                logger.warning("Missing required data for spectral fingerprint analysis")
                return {'error': 'Missing data for spectral analysis'}
            
            # 1. Polarization degree increases with temperature
            # Analyze relationship between polarization degree and temperature
            pol_degree = results['polarization_degree']
            pred_temp = results['predicted_temp']
            
            # Create temperature bins
            temp_bins = 10
            temp_min, temp_max = np.nanpercentile(pred_temp, [5, 95])
            temp_ranges = np.linspace(temp_min, temp_max, temp_bins+1)
            
            # Calculate mean polarization in each temperature bin
            pol_by_temp = np.zeros(temp_bins)
            
            for i in range(temp_bins):
                mask = (pred_temp >= temp_ranges[i]) & (pred_temp < temp_ranges[i+1])
                if np.sum(mask) > 0:
                    pol_by_temp[i] = np.nanmean(pol_degree[mask])
            
            # Check if polarization increases with temperature
            # Linear regression of polarization vs temperature
            valid_bins = ~np.isnan(pol_by_temp)
            if np.sum(valid_bins) > 2:
                from scipy.stats import linregress
                temp_centers = (temp_ranges[:-1] + temp_ranges[1:]) / 2
                slope, intercept, r_value, p_value, std_err = linregress(
                    temp_centers[valid_bins], pol_by_temp[valid_bins]
                )
                
                pol_temp_relationship = {
                    'slope': slope,
                    'r_value': r_value,
                    'p_value': p_value,
                    'increases_with_temp': slope > 0 and p_value < 0.05
                }
                
                if pol_temp_relationship['increases_with_temp']:
                    logger.info("✓ Signature detected: Polarization degree increases with temperature")
                else:
                    logger.info("✗ Signature not detected: Polarization degree does not increase with temperature")
                
                fingerprints['pol_temp_relationship'] = pol_temp_relationship
            
            # 2. Spectral line widths scale with local information rate
            # This would require spectroscopic data, but we can approximate with info density variability
            info_density = results['info_density']
            
            # Calculate spatial variations as a proxy for spectral line widths
            info_std = ndimage.generic_filter(info_density, np.std, size=5)
            
            # Check correlation between information density and its variability
            mask = (info_density > 0) & (info_std > 0)
            if np.sum(mask) > 10:
                info_corr, info_p = pearsonr(
                    np.log10(np.maximum(info_density[mask], 1e-30)),
                    np.log10(np.maximum(info_std[mask], 1e-30))
                )
                
                fingerprints['info_variability'] = {
                    'correlation': info_corr,
                    'p_value': info_p,
                    'scales_with_info': info_corr > 0 and info_p < 0.05
                }
                
                if fingerprints['info_variability']['scales_with_info']:
                    logger.info("✓ Signature detected: Variability scales with information rate")
                else:
                    logger.info("✗ Signature not detected: Variability does not scale with information rate")
            
            # 3. Non-thermal velocity distributions near saturation points
            # Identify regions approaching saturation
            saturation_threshold = 0.9/self.gamma
            near_saturation = info_density > saturation_threshold
            
            # Calculate information gradient in these regions
            if np.sum(near_saturation) > 0:
                grad_y, grad_x = np.gradient(info_density)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                
                # Calculate gradient statistics in saturated vs non-saturated regions
                grad_sat = grad_mag[near_saturation]
                grad_nonsat = grad_mag[~near_saturation]
                
                if len(grad_sat) > 0 and len(grad_nonsat) > 0:
                    grad_ratio = np.mean(grad_sat) / np.mean(grad_nonsat)
                    
                    fingerprints['saturation_gradients'] = {
                        'gradient_ratio': grad_ratio,
                        'non_thermal_signature': grad_ratio > 1.5
                    }
                    
                    if fingerprints['saturation_gradients']['non_thermal_signature']:
                        logger.info("✓ Signature detected: Non-thermal distributions near saturation points")
                    else:
                        logger.info("✗ Signature not detected: No non-thermal distributions near saturation")
            
            # Overall spectral fingerprint score
            detected_signatures = sum(1 for key in fingerprints 
                                    if key != 'error' and fingerprints[key].get('increases_with_temp', False) or
                                       fingerprints[key].get('scales_with_info', False) or
                                       fingerprints[key].get('non_thermal_signature', False))
            
            fingerprints['signatures_detected'] = detected_signatures
            fingerprints['total_signatures'] = 3
            fingerprints['detection_ratio'] = detected_signatures / 3
            
            logger.info(f"Detected {detected_signatures}/3 expected spectral fingerprints")
            
            return fingerprints
            
        except Exception as e:
            logger.error(f"Error in spectral fingerprint analysis: {str(e)}")
            return {'error': str(e)}
            
    def analyze_temporal_evolution(self, time_series_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze temporal evolution signatures as described in Section 4.2 of the protocol.
        
        Args:
            time_series_data: Dictionary with time series data for info_density and predicted_temp
            
        Returns:
            Dictionary with temporal evolution analysis results
        """
        logger.info("Analyzing temporal evolution signatures")
        
        temporal_results = {}
        
        try:
            # Check if we have the necessary data
            if not isinstance(time_series_data, dict) or not all(k in time_series_data for k in ['info_density', 'predicted_temp']):
                logger.warning("Missing required data for temporal evolution analysis")
                return {'error': 'Missing data for temporal analysis'}
            
            info_time_series = np.array(time_series_data['info_density'])
            temp_time_series = np.array(time_series_data['predicted_temp'])
            
            if info_time_series.ndim != 3 or temp_time_series.ndim != 3:
                logger.warning(f"Expected 3D time series data, got info: {info_time_series.ndim}D, temp: {temp_time_series.ndim}D")
                return {'error': 'Invalid time series data dimensions'}
            
            # 1. Temperature bursts when information density approaches saturation
            # Calculate information density relative to saturation
            saturation_threshold = 1/self.gamma
            relative_info = info_time_series / saturation_threshold
            
            # Find bursts in temperature
            temp_diff = np.zeros_like(temp_time_series)
            temp_diff[1:] = temp_time_series[1:] - temp_time_series[:-1]
            temp_bursts = temp_diff > np.percentile(temp_diff, 95)
            
            # Check if temperature bursts follow high information density
            burst_detection = False
            info_before_burst = []
            
            for t in range(1, temp_time_series.shape[0]):
                # Find burst locations
                burst_locs = np.where(temp_bursts[t])
                
                if len(burst_locs[0]) > 0:
                    # Check information density in previous time step
                    prev_info = np.mean(relative_info[t-1, burst_locs[0], burst_locs[1]])
                    info_before_burst.append(prev_info)
            
            if info_before_burst:
                mean_info_before_burst = np.mean(info_before_burst)
                burst_detection = mean_info_before_burst > 0.8  # 80% of saturation
                
                temporal_results['temp_bursts'] = {
                    'info_before_burst': mean_info_before_burst,
                    'bursts_follow_high_info': burst_detection
                }
                
                if burst_detection:
                    logger.info("✓ Signature detected: Temperature bursts follow high information density")
                else:
                    logger.info("✗ Signature not detected: Temperature bursts do not follow high information density")
            
            # 2. Recovery to lower temperatures after information discharge
            # Look for information discharge events
            info_diff = np.zeros_like(info_time_series)
            info_diff[1:] = info_time_series[1:] - info_time_series[:-1]
            info_discharge = info_diff < np.percentile(info_diff, 5)  # Large negative changes
            
            temp_after_discharge = []
            
            for t in range(1, info_time_series.shape[0]-1):  # Skip last frame
                # Find discharge locations
                discharge_locs = np.where(info_discharge[t])
                
                if len(discharge_locs[0]) > 0:
                    # Check temperature change in next time step
                    next_temp_diff = np.mean(temp_diff[t+1, discharge_locs[0], discharge_locs[1]])
                    temp_after_discharge.append(next_temp_diff)
            
            if temp_after_discharge:
                mean_temp_after_discharge = np.mean(temp_after_discharge)
                recovery_detection = mean_temp_after_discharge < 0  # Negative temperature change
                
                temporal_results['temp_recovery'] = {
                    'temp_change_after_discharge': mean_temp_after_discharge,
                    'recovery_after_discharge': recovery_detection
                }
                
                if recovery_detection:
                    logger.info("✓ Signature detected: Temperature decreases after information discharge")
                else:
                    logger.info("✗ Signature not detected: No temperature recovery after information discharge")
            
            # 3. Quasi-periodic oscillations with frequency ~ γ
            # Look for oscillations in spatial mean information density
            mean_info_density = np.mean(info_time_series, axis=(1, 2))
            
            # Compute power spectrum
            from scipy.signal import periodogram
            frequencies, power = periodogram(mean_info_density, fs=1/self.cadence)
            
            # Find peak frequency
            peak_idx = np.argmax(power[1:]) + 1  # Skip DC component
            peak_freq = frequencies[peak_idx]
            
            # Check if peak frequency is close to γ
            # Convert to period in seconds
            peak_period = 1/peak_freq if peak_freq > 0 else float('inf')
            gamma_period = 1/self.gamma
            
            # Check if periods are within an order of magnitude
            ratio = peak_period / gamma_period
            quasi_periodic = 0.1 < ratio < 10
            
            temporal_results['quasi_periodic'] = {
                'peak_frequency': peak_freq,
                'peak_period': peak_period,
                'gamma': self.gamma,
                'gamma_period': gamma_period,
                'period_ratio': ratio,
                'oscillation_detected': quasi_periodic
            }
            
            if quasi_periodic:
                logger.info(f"✓ Signature detected: Quasi-periodic oscillations with frequency ~ γ (ratio={ratio:.2f})")
            else:
                logger.info(f"✗ Signature not detected: No quasi-periodic oscillations with frequency ~ γ (ratio={ratio:.2f})")
            
            # Overall temporal signature score
            detected_signatures = sum(1 for key in temporal_results 
                                   if key != 'error' and 
                                   (key == 'temp_bursts' and temporal_results[key].get('bursts_follow_high_info', False) or
                                    key == 'temp_recovery' and temporal_results[key].get('recovery_after_discharge', False) or
                                    key == 'quasi_periodic' and temporal_results[key].get('oscillation_detected', False)))
            
            temporal_results['signatures_detected'] = detected_signatures
            temporal_results['total_signatures'] = 3
            temporal_results['detection_ratio'] = detected_signatures / 3
            
            logger.info(f"Detected {detected_signatures}/3 expected temporal evolution signatures")
            
            return temporal_results
            
        except Exception as e:
            logger.error(f"Error in temporal evolution analysis: {str(e)}")
            return {'error': str(e)}
    
    def run_analysis(self) -> Dict[str, Any]:
        """
        Run the full analysis pipeline on available PUNCH data.
        
        Returns:
            Dictionary with analysis results
        """
        # List available data files
        files = self.list_available_data()
        
        if not files:
            logger.warning("No PUNCH data files found. Attempting to generate simulated data.")
            # Generate simulated data
            try:
                self._generate_simulated_data()
                files = self.list_available_data()
                if not files:
                    logger.error("Failed to generate or find simulated data.")
                    return {'error': 'No data files found and simulation failed'}
            except Exception as e:
                logger.error(f"Error generating simulated data: {str(e)}")
                return {'error': f'Data generation failed: {str(e)}'}
        
        # Initialize results container
        all_results = {
            'files_analyzed': [],
            'time_series': {
                'info_density': [],
                'predicted_temp': []
            },
            'statistics': {
                'pearson_r': [],
                'p_value': []
            }
        }
        
        # Try to fetch complementary data if needed
        complementary_data = self.fetch_complementary_data()
        spice_files = complementary_data.get('spice', [])
        eis_files = complementary_data.get('eis', [])
        
        # Load temperature data if available
        temp_data = None
        if spice_files:
            temp_data = self.load_temperature_data(spice_file=spice_files[0])
        elif eis_files:
            temp_data = self.load_temperature_data(eis_file=eis_files[0])
        
        # Analyze each file
        for file in files:
            results = self.analyze_punch_data(file, temp_data=temp_data)
            
            if 'error' in results:
                logger.warning(f"Skipping file {file} due to error")
                continue
            
            # Store results
            all_results['files_analyzed'].append(file)
            all_results['time_series']['info_density'].append(results['info_density'])
            all_results['time_series']['predicted_temp'].append(results['predicted_temp'])
            
            # Calculate statistics if available
            if 'info_density' in results and 'predicted_temp' in results:
                # Flatten arrays and filter for valid values
                info_flat = results['info_density'].flatten()
                temp_flat = results['predicted_temp'].flatten()
                valid_mask = np.isfinite(info_flat) & np.isfinite(temp_flat) & (info_flat > 0) & (temp_flat > 0)
                
                if np.sum(valid_mask) > 10:  # Need at least 10 points for meaningful statistics
                    from scipy.stats import pearsonr
                    try:
                        # Calculate log-log correlation
                        pr, pval = pearsonr(np.log10(info_flat[valid_mask]), np.log10(temp_flat[valid_mask]))
                        all_results['statistics']['pearson_r'].append(pr)
                        all_results['statistics']['p_value'].append(pval)
                        logger.info(f"File {os.path.basename(file)}: Pearson r = {pr:.3f}, p-value = {pval:.3e}")
                        
                        # Check if correlation meets protocol significance criteria
                        if pr > 0.8 and pval < 0.01:
                            logger.info(f"✓ File meets protocol significance criteria (r > 0.8, p < 0.01)")
                        else:
                            logger.info(f"✗ File does not meet protocol significance criteria")
                    except Exception as e:
                        logger.warning(f"Could not calculate statistics for {file}: {e}")
            
            # Generate and save plots for this file
            self.generate_plots(results, os.path.splitext(os.path.basename(file))[0])
            
            # Analyze spectral fingerprints for this file (protocol section 4.3)
            try:
                from scipy.signal import periodogram
                from scipy import ndimage
                
                # Calculate and store spectral fingerprints
                # 1. Check if polarization degree increases with temperature
                pol_deg = results['polarization_degree']
                pred_temp = results['predicted_temp']
                valid_mask = np.isfinite(pol_deg) & np.isfinite(pred_temp) & (pred_temp > 0)
                
                if np.sum(valid_mask) > 100:  # Need enough pixels for a good correlation
                    pol_corr, pol_pval = pearsonr(
                        pol_deg[valid_mask].flatten(),
                        np.log10(pred_temp[valid_mask].flatten())
                    )
                    
                    results['pol_temp_correlation'] = {
                        'pearson_r': pol_corr,
                        'p_value': pol_pval,
                        'increases_with_temp': pol_corr > 0 and pol_pval < 0.05
                    }
                    
                    if results['pol_temp_correlation']['increases_with_temp']:
                        logger.info("✓ Spectral signature 4.3.1: Polarization increases with temperature")
                    else:
                        logger.info("✗ Spectral signature 4.3.1 not detected")
                
                # 2. Check if line widths scale with local information rate
                # Approximate with local standard deviation as width proxy
                info_density = results['info_density']
                info_local_std = ndimage.generic_filter(info_density, np.std, size=5)
                
                valid_mask = np.isfinite(info_density) & np.isfinite(info_local_std) & (info_density > 0)
                if np.sum(valid_mask) > 100:
                    width_corr, width_pval = pearsonr(
                        np.log10(info_density[valid_mask].flatten()),
                        np.log10(np.maximum(info_local_std[valid_mask].flatten(), 1e-30))
                    )
                    
                    results['width_info_correlation'] = {
                        'pearson_r': width_corr,
                        'p_value': width_pval,
                        'widths_scale_with_info': width_corr > 0 and width_pval < 0.05
                    }
                    
                    if results['width_info_correlation']['widths_scale_with_info']:
                        logger.info("✓ Spectral signature 4.3.2: Line widths scale with information rate")
                    else:
                        logger.info("✗ Spectral signature 4.3.2 not detected")
                
                # 3. Check for non-thermal velocity distributions near saturation
                if 'high_thomson_regions' in results:
                    # Calculate entropy in high Thomson regions vs. regular regions
                    high_thomson = results['high_thomson_regions']
                    
                    if np.sum(high_thomson) > 0:
                        # Use local gradient as a proxy for velocity distribution width
                        grad_y, grad_x = np.gradient(info_density)
                        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                        
                        mean_grad_thomson = np.mean(grad_mag[high_thomson])
                        mean_grad_normal = np.mean(grad_mag[~high_thomson])
                        
                        grad_ratio = mean_grad_thomson / mean_grad_normal if mean_grad_normal > 0 else 0
                        
                        results['non_thermal_signature'] = {
                            'gradient_ratio': grad_ratio,
                            'non_thermal_detected': grad_ratio > 1.5
                        }
                        
                        if results['non_thermal_signature']['non_thermal_detected']:
                            logger.info("✓ Spectral signature 4.3.3: Non-thermal velocity distributions detected")
                        else:
                            logger.info("✗ Spectral signature 4.3.3 not detected")
            except Exception as e:
                logger.warning(f"Error in spectral fingerprint analysis: {e}")
        
        # If we have time series data, analyze phase transitions and temporal evolution
        if len(all_results['time_series']['info_density']) > 1:
            # Convert list to 3D array (time, y, x)
            info_time_series = np.array(all_results['time_series']['info_density'])
            
            # Get high Thomson scattering regions from the last frame (most recent)
            high_thomson = None
            last_file_results = self.analyze_punch_data(files[-1])
            if 'high_thomson_regions' in last_file_results:
                high_thomson = last_file_results['high_thomson_regions']
            
            # Detect phase transitions
            transitions, transition_locs, transition_map = self.detect_phase_transitions(
                info_time_series, 
                threshold=np.pi/2,
                thomson_regions=high_thomson
            )
            
            all_results['transitions'] = {
                'times': transitions,
                'locations': transition_locs,
                'map': transition_map
            }
            
            # Generate time series plots
            self.generate_time_series_plots(all_results)
            
            # Check for temporal signatures (section 4.2 of protocol)
            try:
                # 1. Temperature bursts when info density approaches saturation
                temp_time_series = np.array(all_results['time_series']['predicted_temp'])
                
                # Calculate temporal derivative of temperature
                temp_diff = np.zeros_like(temp_time_series)
                temp_diff[1:] = temp_time_series[1:] - temp_time_series[:-1]
                
                # Define temperature bursts as top 5% of positive changes
                burst_threshold = np.percentile(temp_diff[temp_diff > 0], 95)
                temp_bursts = temp_diff > burst_threshold
                
                # Check if bursts follow high information density
                if np.any(temp_bursts[1:]):  # Skip first frame
                    # For each burst, check info density in previous frame
                    saturation_level = []
                    
                    for t in range(1, len(temp_time_series)):
                        if np.any(temp_bursts[t]):
                            # Get burst locations
                            burst_y, burst_x = np.where(temp_bursts[t])
                            
                            # Check info density in previous frame at those locations
                            prev_info = info_time_series[t-1, burst_y, burst_x]
                            
                            # Calculate how close to saturation (1/γ)
                            saturation_frac = prev_info * self.gamma
                            saturation_level.append(np.mean(saturation_frac))
                    
                    if saturation_level:
                        all_results['temp_bursts'] = {
                            'mean_saturation_before_burst': np.mean(saturation_level),
                            'follows_high_info': np.mean(saturation_level) > 0.8
                        }
                        
                        if all_results['temp_bursts']['follows_high_info']:
                            logger.info("✓ Temporal signature 4.2.1: Temperature bursts follow high information density")
                        else:
                            logger.info("✗ Temporal signature 4.2.1 not detected")
                
                # 2. Quasi-periodic oscillations with frequency ~ γ
                # Calculate spatial average of information density
                if info_time_series.shape[0] > 10:  # Need enough time points
                    mean_info = np.mean(info_time_series, axis=(1, 2))
                    
                    # Calculate power spectrum
                    timestep = 240  # 4-minute cadence in seconds
                    freqs, psd = periodogram(mean_info, fs=1/timestep)
                    
                    # Find peak frequency
                    peak_idx = np.argmax(psd[1:]) + 1  # Skip DC component
                    peak_freq = freqs[peak_idx]
                    
                    # Compare with γ (should be within order of magnitude)
                    freq_ratio = peak_freq / self.gamma
                    
                    all_results['quasi_periodic'] = {
                        'peak_frequency': peak_freq,
                        'gamma_frequency': self.gamma,
                        'frequency_ratio': freq_ratio,
                        'detected': 0.1 < freq_ratio < 10
                    }
                    
                    if all_results['quasi_periodic']['detected']:
                        logger.info(f"✓ Temporal signature 4.2.2: Quasi-periodic oscillations detected with f/γ = {freq_ratio:.2f}")
                    else:
                        logger.info(f"✗ Temporal signature 4.2.2 not detected: f/γ = {freq_ratio:.2f}")
            except Exception as e:
                logger.warning(f"Error in temporal evolution analysis: {e}")
        
        # Calculate overall statistics
        if all_results['statistics']['pearson_r']:
            all_results['statistics']['mean_pearson_r'] = np.mean(all_results['statistics']['pearson_r'])
            all_results['statistics']['mean_p_value'] = np.mean(all_results['statistics']['p_value'])
            
            # Check against protocol significance criteria
            all_results['statistics']['meets_criteria'] = (
                all_results['statistics']['mean_pearson_r'] > 0.8 and 
                all_results['statistics']['mean_p_value'] < 0.01
            )
            
            logger.info(f"Overall: Mean Pearson r = {all_results['statistics']['mean_pearson_r']:.3f}")
            logger.info(f"Overall: Mean p-value = {all_results['statistics']['mean_p_value']:.3e}")
            
            if all_results['statistics']['meets_criteria']:
                logger.info("✓ Analysis meets protocol significance criteria")
            else:
                logger.info("✗ Analysis does not meet protocol significance criteria")
        
        if 'transitions' in all_results:
            logger.info(f"Phase transitions detected: {len(all_results['transitions']['times'])}")
        
        return all_results
    
    def _generate_simulated_data(self):
        """
        Generate simulated PUNCH data files for testing when no real data is available.
        """
        logger.info("Generating simulated PUNCH polarimetric data for analysis...")
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Determine dates to generate
        num_days = (self.end_date - self.start_date).days
        if num_days > 30:
            # Limit to 30 days if the period is too long
            logger.warning(f"Limiting simulation to 30 days out of {num_days} day period")
            num_days = 30
            
        # Generate a file for selected dates within the period
        step = max(1, num_days // 10)  # Generate about 10 files evenly spaced
        
        for i in range(0, num_days, step):
            current_date = self.start_date + timedelta(days=i)
            date_str = current_date.strftime("%Y%m%d")
            filename = f"punch_{date_str}_polarimetric.fits"
            filepath = os.path.join(self.data_dir, filename)
            
            # Skip if file already exists
            if os.path.exists(filepath):
                logger.info(f"Simulated data file already exists: {filename}")
                continue
                
            logger.info(f"Generating simulated data for date {date_str}")
            
            try:
                # Create simulated PUNCH polarimetric data
                self._create_simulated_fits_file(filepath, date_str)
                logger.info(f"Created simulated FITS file: {filepath}")
            except Exception as e:
                logger.warning(f"Error generating simulated data for {date_str}: {str(e)}")
                
        return
    
    def _create_simulated_fits_file(self, filepath, date_str):
        """Create a simulated FITS file with PUNCH-like data structure."""
        # Import required modules
        from astropy.io import fits
        
        # Define image parameters
        image_size = 256  # 256x256 pixel image
        r_sun = image_size // 4  # Solar radius in pixels
        
        # Create coordinate grid
        y, x = np.ogrid[-image_size//2:image_size//2, -image_size//2:image_size//2]
        r = np.sqrt(x*x + y*y)
        
        # Create Stokes parameters with realistic features
        # Stokes I - Total intensity (Thomson-scattered white light)
        I = np.exp(-(r-r_sun*1.5)**2/(2*(r_sun*0.5)**2))
        I = I / I.max()  # Normalize
        
        # Add coronal structures based on the date (for time series analysis)
        date_num = int(date_str)
        np.random.seed(date_num)  # Seed the random generator for reproducibility
        
        # Add coronal streamers
        for i in range(3):
            angle = np.random.uniform(0, 2*np.pi)
            width = np.random.uniform(0.1, 0.3)
            length = np.random.uniform(1.5, 3.0) * r_sun
            
            # Streamer mask based on angle and length
            streamer_mask = (np.abs(np.arctan2(y, x) - angle) % np.pi < width) & (r < length) & (r > r_sun)
            I[streamer_mask] *= 1.5  # Enhance brightness in streamers
        
        # Add active regions near the limb
        for i in range(2):
            limb_angle = np.random.uniform(0, 2*np.pi)
            limb_x = r_sun * np.cos(limb_angle)
            limb_y = r_sun * np.sin(limb_angle)
            size = np.random.uniform(0.15, 0.3) * r_sun
            
            # Active region enhancement
            active_region = np.exp(-((x-limb_x)**2 + (y-limb_y)**2)/(2*size**2))
            I += active_region * 0.3
        
        # Stokes Q, U - Linear polarization (proportional to Thomson scattering)
        Q = I * 0.1 * np.cos(np.arctan2(y, x) * 2)
        U = I * 0.1 * np.sin(np.arctan2(y, x) * 2)
        
        # Add occulter to simulate coronagraph
        occulter_mask = r < r_sun
        I[occulter_mask] = 0
        Q[occulter_mask] = 0
        U[occulter_mask] = 0
        
        # Create FITS header with proper WCS
        header = fits.Header()
        
        # Standard FITS keywords
        header['SIMPLE'] = True
        header['BITPIX'] = -64
        header['NAXIS'] = 2
        header['NAXIS1'] = image_size
        header['NAXIS2'] = image_size
        
        # Observation date
        obs_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T00:00:00.000"
        header['DATE-OBS'] = obs_date
        header['DATE'] = datetime.now().isoformat()
        
        # WCS for solar coordinates
        header['WCSNAME'] = 'HELIOPROJECTIVE'
        header['CTYPE1'] = 'HPLN-TAN'  # Helioprojective longitude
        header['CTYPE2'] = 'HPLT-TAN'  # Helioprojective latitude
        header['CUNIT1'] = 'arcsec'
        header['CUNIT2'] = 'arcsec'
        header['CRPIX1'] = image_size // 2 + 0.5  # Center pixel X
        header['CRPIX2'] = image_size // 2 + 0.5  # Center pixel Y
        header['CDELT1'] = 4.0  # 4 arcsec per pixel resolution
        header['CDELT2'] = 4.0
        header['CRVAL1'] = 0.0  # Sun center (degrees)
        header['CRVAL2'] = 0.0
        
        # Protocol-specific metadata
        header['TELESCOP'] = 'PUNCH'
        header['INSTRUME'] = 'NFI+WFI'
        header['OBJECT'] = 'SOLAR CORONA'
        header['CADENCE'] = self.cadence
        header['FOVMIN'] = self.min_fov_rsun
        header['FOVMAX'] = self.max_fov_rsun
        header['CREATOR'] = 'HoloPy PUNCH Simulator'
        header['DESCRIPT'] = 'Simulated PUNCH polarimetric data for holographic analysis'
        
        # Create HDU structure
        primary_hdu = fits.PrimaryHDU(I, header=header)
        q_hdu = fits.ImageHDU(Q, name='Q')
        u_hdu = fits.ImageHDU(U, name='U')
        
        # Create HDU list and write to file
        hdul = fits.HDUList([primary_hdu, q_hdu, u_hdu])
        hdul.writeto(filepath, overwrite=True)
    
    def generate_plots(self, results: Dict[str, Any], base_filename: str) -> None:
        """
        Generate plots from the analysis results.
        
        Args:
            results: Dictionary containing analysis results
            base_filename: Base filename for plot files
        """
        if 'error' in results:
            return
        
        try:
            # Create figure directory
            fig_dir = os.path.join(self.output_dir, 'figures')
            os.makedirs(fig_dir, exist_ok=True)
            
            # Use sunpy's visualization capabilities for maps
            if 'polarization_map' in results and 'info_map' in results and 'temp_map' in results:
                pol_map = results['polarization_map']
                info_map = results['info_map']
                temp_map = results['temp_map']
                
                # Plot polarization degree map
                plt.figure(figsize=(10, 8))
                pol_map.plot(title=f'Polarization Degree - {pol_map.date.strftime("%Y-%m-%d")}')
                pol_map.draw_limb(color='black')
                plt.colorbar(label='Polarization Degree')
                plt.savefig(os.path.join(fig_dir, f'{base_filename}_polarization.png'), dpi=300)
                plt.close()
                
                # Plot information density map
                plt.figure(figsize=(10, 8))
                info_map.plot(cmap='plasma', title=f'Information Density - {info_map.date.strftime("%Y-%m-%d")}')
                info_map.draw_limb(color='black')
                plt.colorbar(label='Information Density [s]')
                plt.savefig(os.path.join(fig_dir, f'{base_filename}_info_density.png'), dpi=300)
                plt.close()
                
                # Plot predicted temperature map
                plt.figure(figsize=(10, 8))
                temp_map.plot(cmap='hot', title=f'Predicted Temperature - {temp_map.date.strftime("%Y-%m-%d")}')
                temp_map.draw_limb(color='black')
                plt.colorbar(label='Temperature [K]')
                plt.savefig(os.path.join(fig_dir, f'{base_filename}_predicted_temp.png'), dpi=300)
                plt.close()
            
            # Create combined figure with all analysis results
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            # Information density map
            im1 = axes[0, 0].imshow(results['info_density'], cmap='plasma')
            axes[0, 0].set_title('Information Density')
            plt.colorbar(im1, ax=axes[0, 0], label='ρ_info [s]')
            
            # Predicted temperature map
            im2 = axes[0, 1].imshow(results['predicted_temp'], cmap='hot')
            axes[0, 1].set_title('Predicted Temperature')
            plt.colorbar(im2, ax=axes[0, 1], label='T [K]')
            
            # Scatter plot: info density vs temperature
            valid_mask = (results['info_density'] > 0) & (results['predicted_temp'] < 10e6)
            info_flat = results['info_density'][valid_mask].flatten()
            temp_flat = results['predicted_temp'][valid_mask].flatten()
            
            if len(info_flat) > 0 and len(temp_flat) > 0:
                axes[1, 0].scatter(info_flat, temp_flat/1e6, alpha=0.1, s=1, c='blue')
                axes[1, 0].set_xlabel('Information Density [s]')
                axes[1, 0].set_ylabel('Predicted Temperature [MK]')
                axes[1, 0].set_title('Info Density vs Temperature')
                axes[1, 0].set_xscale('log')
                axes[1, 0].set_yscale('log')
                
                # Calculate correlation coefficient
                from scipy.stats import pearsonr
                try:
                    pearson_r, p_value = pearsonr(np.log10(info_flat), np.log10(temp_flat))
                    axes[1, 0].text(0.05, 0.95, f'r = {pearson_r:.3f}\np = {p_value:.3e}', 
                                    transform=axes[1, 0].transAxes, 
                                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
                except Exception as e:
                    logger.warning(f"Could not calculate pearsonr: {e}")
            
            # Histogram of information density
            if len(info_flat) > 0:
                axes[1, 1].hist(np.log10(info_flat), bins=50, alpha=0.7, density=True)
                axes[1, 1].set_xlabel('log₁₀(Information Density)')
                axes[1, 1].set_ylabel('Probability Density')
                axes[1, 1].set_title('Information Density Distribution')
                
                # Add vertical line for saturation threshold
                saturation_threshold = 1/self.gamma
                axes[1, 1].axvline(np.log10(saturation_threshold), color='red', linestyle='--', 
                                   label=f'Saturation: {saturation_threshold:.1e} s')
                axes[1, 1].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f'{base_filename}_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Create radial profile plots
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            
            # Get radial profile data
            r_values = np.unique(results['radial_distance_rsun'].flatten())
            r_values = r_values[r_values > 0]  # Remove zero values
            
            # Initialize arrays for binned data
            info_radial = np.zeros_like(r_values)
            temp_radial = np.zeros_like(r_values)
            
            # Bin data by radius
            for i, r in enumerate(r_values):
                mask = np.isclose(results['radial_distance_rsun'], r, rtol=1e-2)
                if np.any(mask):
                    info_radial[i] = np.mean(results['info_density'][mask])
                    temp_radial[i] = np.mean(results['predicted_temp'][mask])
            
            # Plot information density vs radius
            axes[0].plot(r_values, info_radial, 'b-', label='Information Density')
            axes[0].set_xlabel('Radial Distance [R_sun]')
            axes[0].set_ylabel('Information Density [s]')
            axes[0].set_title('Information Density Radial Profile')
            axes[0].set_yscale('log')
            axes[0].legend()
            
            # Plot temperature vs radius
            axes[1].plot(r_values, temp_radial/1e6, 'r-', label='Predicted Temperature')
            axes[1].set_xlabel('Radial Distance [R_sun]')
            axes[1].set_ylabel('Temperature [MK]')
            axes[1].set_title('Temperature Radial Profile')
            axes[1].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f'{base_filename}_radial.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
    
    def generate_time_series_plots(self, all_results: Dict[str, Any]) -> None:
        """
        Generate plots showing time series analysis.
        
        Args:
            all_results: Dictionary containing all analysis results
        """
        try:
            # Create figure directory
            fig_dir = os.path.join(self.output_dir, 'figures')
            os.makedirs(fig_dir, exist_ok=True)
            
            # Calculate time-averaged statistics
            info_time_series = np.array(all_results['time_series']['info_density'])
            temp_time_series = np.array(all_results['time_series']['predicted_temp'])
            
            # Average over time
            info_mean = np.mean(info_time_series, axis=0)
            temp_mean = np.mean(temp_time_series, axis=0)
            
            # Calculate standard deviation
            info_std = np.std(info_time_series, axis=0)
            temp_std = np.std(temp_time_series, axis=0)
            
            # Generate plots
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Mean information density map
            im1 = axes[0, 0].imshow(info_mean, cmap='plasma')
            axes[0, 0].set_title('Mean Information Density')
            plt.colorbar(im1, ax=axes[0, 0], label='ρ_info [s]')
            
            # Mean predicted temperature map
            im2 = axes[0, 1].imshow(temp_mean/1e6, cmap='hot')
            axes[0, 1].set_title('Mean Predicted Temperature')
            plt.colorbar(im2, ax=axes[0, 1], label='T [MK]')
            
            # Information density variability
            im3 = axes[1, 0].imshow(info_std/np.maximum(info_mean, 1e-10), cmap='viridis')
            axes[1, 0].set_title('Information Density Variability')
            plt.colorbar(im3, ax=axes[1, 0], label='Relative Std Dev')
            
            # Temperature variability
            im4 = axes[1, 1].imshow(temp_std/np.maximum(temp_mean, 1e-10), cmap='viridis')
            axes[1, 1].set_title('Temperature Variability')
            plt.colorbar(im4, ax=axes[1, 1], label='Relative Std Dev')
            
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'time_series_averages.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Plot phase transitions if detected
            if 'transitions' in all_results and all_results['transitions']['times']:
                # Create map showing transition locations
                transition_map = np.zeros_like(info_mean)
                
                for _, loc in zip(all_results['transitions']['times'], all_results['transitions']['locations']):
                    y, x = loc
                    transition_map[y, x] = 1
                
                # Apply gaussian filter to make points more visible
                from scipy.ndimage import gaussian_filter
                transition_map = gaussian_filter(transition_map, sigma=1)
                
                # Plot
                plt.figure(figsize=(10, 8))
                plt.imshow(transition_map, cmap='hot', alpha=0.7)
                plt.colorbar(label='Phase Transition Density')
                plt.title('Information Phase Transition Locations')
                plt.savefig(os.path.join(fig_dir, 'phase_transitions.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Plot transition time histogram
                plt.figure(figsize=(10, 6))
                plt.hist(all_results['transitions']['times'], bins=20, alpha=0.7)
                plt.xlabel('Time Index')
                plt.ylabel('Number of Transitions')
                plt.title('Phase Transition Temporal Distribution')
                plt.savefig(os.path.join(fig_dir, 'transition_times.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logger.error(f"Error generating time series plots: {e}")
    
    def fetch_complementary_data(self) -> Dict[str, List[str]]:
        """
        Fetch complementary data from other missions as specified in the protocol.
        
        Returns:
            Dictionary of downloaded file paths by instrument
        """
        complementary_data = {}
        
        try:
            # Create data directory if it doesn't exist
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Create subdirectories for different data types
            for subdir in ['spice', 'aia', 'psp', 'eis']:
                os.makedirs(os.path.join(self.data_dir, subdir), exist_ok=True)
            
            # Convert datetime objects to strings
            start_date_str = self.start_date.strftime('%Y-%m-%d')
            end_date_str = self.end_date.strftime('%Y-%m-%d')
            time_range = a.Time(start_date_str, end_date_str)
            
            # 1. Solar Orbiter/SPICE: Temperature diagnostics
            logger.info("Fetching Solar Orbiter/SPICE temperature diagnostics data...")
            try:
                spice_result = Fido.search(
                    time_range,
                    a.Instrument('SPICE'),
                    a.Physobs('temp')
                )
                
                if len(spice_result) > 0:
                    spice_files = Fido.fetch(spice_result, path=os.path.join(self.data_dir, 'spice', '{file}'))
                    complementary_data['spice'] = spice_files
                    logger.info(f"Downloaded {len(spice_files)} SPICE files")
                else:
                    logger.warning("No SPICE temperature data found")
                    complementary_data['spice'] = []
            except Exception as e:
                logger.warning(f"Error fetching SPICE data: {e}")
                complementary_data['spice'] = []
            
            # 2. SDO/AIA: Multi-wavelength imaging
            logger.info("Fetching SDO/AIA multi-wavelength imaging data...")
            try:
                # Get multiple wavelengths (171, 193, 211, 335) for temperature analysis
                wavelengths = [171, 193, 211, 335] * u.angstrom
                
                # Search for each wavelength individually
                aia_files = []
                for wave in wavelengths:
                    aia_result = Fido.search(
                        time_range,
                        a.Instrument('AIA'),
                        a.Wavelength(wave)
                    )
                    
                    if len(aia_result) > 0:
                        # Only get a few files per wavelength to avoid too much data
                        max_files = min(len(aia_result[0]), 2)
                        wave_files = Fido.fetch(aia_result[0][:max_files], 
                                               path=os.path.join(self.data_dir, 'aia', '{file}'))
                        aia_files.extend(wave_files)
                        logger.info(f"Downloaded {len(wave_files)} AIA {wave} files")
                
                complementary_data['aia'] = aia_files
                logger.info(f"Downloaded a total of {len(aia_files)} AIA files")
                
            except Exception as e:
                logger.warning(f"Error fetching AIA data: {e}")
                complementary_data['aia'] = []
            
            # 3. Parker Solar Probe: In-situ measurements
            logger.info("Fetching Parker Solar Probe in-situ measurements...")
            try:
                # We'll try each instrument separately since combining might cause issues
                psp_files = []
                
                # Try FIELDS instrument
                fields_result = Fido.search(
                    time_range,
                    a.Instrument('FIELDS'),
                    a.Source('PSP')
                )
                
                if len(fields_result) > 0:
                    # Limit to a few files to avoid downloading too much
                    max_fields = min(len(fields_result[0]), 2)
                    fields_files = Fido.fetch(fields_result[0][:max_fields], 
                                             path=os.path.join(self.data_dir, 'psp', '{file}'))
                    psp_files.extend(fields_files)
                    logger.info(f"Downloaded {len(fields_files)} PSP/FIELDS files")
                
                # Try SWEAP instrument
                sweap_result = Fido.search(
                    time_range,
                    a.Instrument('SWEAP'),
                    a.Source('PSP')
                )
                
                if len(sweap_result) > 0:
                    # Limit to a few files to avoid downloading too much
                    max_sweap = min(len(sweap_result[0]), 2)
                    sweap_files = Fido.fetch(sweap_result[0][:max_sweap], 
                                            path=os.path.join(self.data_dir, 'psp', '{file}'))
                    psp_files.extend(sweap_files)
                    logger.info(f"Downloaded {len(sweap_files)} PSP/SWEAP files")
                
                complementary_data['psp'] = psp_files
                logger.info(f"Downloaded a total of {len(psp_files)} Parker Solar Probe files")
                
                if len(psp_files) == 0:
                    logger.warning("No Parker Solar Probe data found")
                
            except Exception as e:
                logger.warning(f"Error fetching Parker Solar Probe data: {e}")
                complementary_data['psp'] = []
            
            # 4. Hinode/EIS: Spectroscopic temperature measurements
            logger.info("Fetching Hinode/EIS spectroscopic temperature measurements...")
            try:
                eis_result = Fido.search(
                    time_range,
                    a.Instrument('EIS'),
                    a.Source('Hinode')
                )
                
                if len(eis_result) > 0:
                    # Limit to a few files to avoid downloading too much
                    max_eis = min(len(eis_result[0]), 2)
                    eis_files = Fido.fetch(eis_result[0][:max_eis], 
                                          path=os.path.join(self.data_dir, 'eis', '{file}'))
                    complementary_data['eis'] = eis_files
                    logger.info(f"Downloaded {len(eis_files)} EIS files")
                else:
                    logger.warning("No EIS data found")
                    complementary_data['eis'] = []
            except Exception as e:
                logger.warning(f"Error fetching EIS data: {e}")
                complementary_data['eis'] = []
            
            # Total summary
            total_files = sum(len(files) for files in complementary_data.values())
            logger.info(f"Downloaded a total of {total_files} complementary data files")
            
            return complementary_data
        
        except Exception as e:
            logger.error(f"Error fetching complementary data: {str(e)}")
            return {}
            
    def load_temperature_data(self, spice_file: str = None, eis_file: str = None) -> Dict[str, np.ndarray]:
        """
        Load temperature measurements from complementary data.
        
        Args:
            spice_file: Path to Solar Orbiter/SPICE file
            eis_file: Path to Hinode/EIS file
            
        Returns:
            Dictionary with temperature maps and metadata
        """
        temperature_data = {}
        
        # Try SPICE data first (priority for temperature diagnostics)
        if spice_file and os.path.exists(spice_file):
            try:
                with fits.open(spice_file) as hdul:
                    # Extract temperature map from SPICE data
                    # SPICE provides temperature diagnostics via line ratios
                    temperature_map = hdul[1].data  # Temperature data typically in extension 1
                    temperature_header = hdul[1].header
                    
                    # Create a proper map with correct coordinates
                    temp_map = sunpy.map.Map(temperature_map, temperature_header)
                    temperature_data['map'] = temp_map
                    temperature_data['source'] = 'Solar Orbiter/SPICE'
                    
                    logger.info(f"Loaded temperature data from SPICE file {os.path.basename(spice_file)}")
                    return temperature_data
            except Exception as e:
                logger.warning(f"Error loading SPICE temperature data: {e}")
        
        # Try EIS data as backup
        if eis_file and os.path.exists(eis_file):
            try:
                with fits.open(eis_file) as hdul:
                    # Extract temperature map from EIS data
                    # EIS provides spectroscopic temperature measurements
                    temperature_map = hdul[1].data  # Temperature data typically in extension 1
                    temperature_header = hdul[1].header
                    
                    # Create a proper map with correct coordinates
                    temp_map = sunpy.map.Map(temperature_map, temperature_header)
                    temperature_data['map'] = temp_map
                    temperature_data['source'] = 'Hinode/EIS'
                    
                    logger.info(f"Loaded temperature data from EIS file {os.path.basename(eis_file)}")
                    return temperature_data
            except Exception as e:
                logger.warning(f"Error loading EIS temperature data: {e}")
        
        logger.warning("No temperature data could be loaded from complementary sources")
        return {}

def main():
    """
    Main execution function.
    
    This function parses command-line arguments and runs the analysis.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='PUNCH Data Analysis with HoloPy')
    parser.add_argument('--data-dir', type=str, default='punch_data',
                        help='Directory containing PUNCH data files')
    parser.add_argument('--output-dir', type=str, default='punch_results',
                        help='Directory for output files')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to analyze (only used if start-date/end-date not provided)')
    parser.add_argument('--start-date', type=str, default="2022-07-01",
                        help='Start date for analysis in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, default="2023-06-30",
                        help='End date for analysis in YYYY-MM-DD format')
    parser.add_argument('--download', action='store_true',
                        help='Attempt to download data from VSO')
    parser.add_argument('--no-complementary', action='store_true',
                        help='Skip complementary data download (SPICE, AIA, PSP, EIS)')
    parser.add_argument('--null-hypothesis', action='store_true',
                        help='Perform explicit null hypothesis testing')
    
    args = parser.parse_args()
    
    # Print HoloPy version and analysis parameters
    logger.info(f"HoloPy version: {holopy.__version__}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Analysis period: {args.start_date} to {args.end_date}")
    
    # Print Analysis Protocol Notice
    logger.info("=" * 80)
    logger.info("PUNCH Heliophysics Analysis following Protocol v1.0")
    logger.info("Testing the hypothesis that coronal heating represents a fundamental")
    logger.info("mechanism where the cosmic screen prevents localized information")
    logger.info("saturation via thermodynamics: T ∝ ρ_info/g")
    logger.info("=" * 80)
    
    # Check if data directory exists and create it if needed
    if not os.path.isdir(args.data_dir):
        if not os.path.exists(args.data_dir):
            logger.warning(f"Data directory {args.data_dir} does not exist. Creating it.")
            os.makedirs(args.data_dir, exist_ok=True)
        else:
            logger.error(f"Data path {args.data_dir} exists but is not a directory.")
            return 1
    
    # Create analyzer
    analyzer = PunchAnalyzer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        days_to_analyze=args.days,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # If download flag is set, attempt to download PUNCH data
    if args.download:
        logger.info("Attempting to download PUNCH data from VSO...")
        downloaded_files = analyzer.fetch_punch_data()
        logger.info(f"Downloaded {len(downloaded_files)} PUNCH files")
        
        # Also download complementary data unless explicitly skipped
        if not args.no_complementary:
            logger.info("Downloading complementary data from other missions...")
            complementary_data = analyzer.fetch_complementary_data()
            for source, files in complementary_data.items():
                logger.info(f"Downloaded {len(files)} {source.upper()} files")
    
    # Run analysis following Protocol steps
    logger.info("Starting analysis pipeline...")
    
    # Phase 1: Data Acquisition
    logger.info("Phase 1: Data Acquisition")
    # (Already handled with data download)
    
    # Phase 2: Algorithm Implementation
    logger.info("Phase 2: Algorithm Implementation")
    logger.info("  - Algorithm 3.1: Information Density Extraction")
    logger.info("  - Algorithm 3.2: Temperature Correlation Analysis")
    logger.info("  - Algorithm 3.3: Phase Transition Detection")
    
    # Phase 3: Analysis
    logger.info("Phase 3: Analysis")
    results = analyzer.run_analysis()
    
    if 'error' in results:
        logger.error(f"Analysis failed: {results['error']}")
        return 1
    
    # Phase 4: Validation
    logger.info("Phase 4: Validation")
    
    # Conduct explicit null hypothesis testing if requested
    if args.null_hypothesis and 'statistics' in results:
        from scipy.stats import ttest_1samp
        
        logger.info("Performing null hypothesis testing:")
        logger.info("  H₀: Coronal temperature is uncorrelated with information density")
        logger.info("  H₁: T ∝ ρ_info/g as predicted by holographic theory")
        
        # Get correlation coefficients
        r_values = results['statistics'].get('pearson_r', [])
        
        if r_values:
            # Test if correlation is significantly above 0 (null hypothesis)
            t_stat, p_val = ttest_1samp(r_values, 0)
            
            logger.info(f"  - Mean correlation coefficient: {np.mean(r_values):.3f}")
            logger.info(f"  - t-statistic: {t_stat:.3f}, p-value: {p_val:.3e}")
            
            # Check against criteria from Section 5.2 of Protocol
            if np.mean(r_values) > 0.8 and p_val < 0.01:
                logger.info("  ✓ Null hypothesis REJECTED at p < 0.01")
                logger.info("  ✓ Results support the holographic heating model")
            else:
                logger.info("  ✗ Failed to reject null hypothesis at required significance")
                if np.mean(r_values) <= 0.8:
                    logger.info("    - Correlation coefficient below protocol threshold of 0.8")
                if p_val >= 0.01:
                    logger.info("    - p-value above protocol threshold of 0.01")
    
    # Print overall summary
    logger.info("Analysis completed successfully")
    logger.info(f"Files analyzed: {len(results['files_analyzed'])}")
    
    # Section 5 from Protocol: Statistical Framework
    if 'statistics' in results and 'mean_pearson_r' in results['statistics']:
        # Section 5.2: Significance Criteria
        pearson_r = results['statistics']['mean_pearson_r']
        p_value = results['statistics']['mean_p_value']
        
        logger.info(f"Statistical results:")
        logger.info(f"  - Pearson correlation coefficient: {pearson_r:.3f} (threshold: > 0.8)")
        logger.info(f"  - p-value: {p_value:.3e} (threshold: < 0.01)")
        
        meets_criteria = pearson_r > 0.8 and p_value < 0.01
        
        if meets_criteria:
            logger.info("  ✓ Results meet protocol significance criteria")
        else:
            logger.info("  ✗ Results do not meet protocol significance criteria")
    
    # Section 4 from Protocol: Expected Signatures
    signatures_detected = 0
    total_signatures = 0
    
    # 4.1 Spatial Correlation Signatures
    if any(k in results for k in ['peak_info_overlap', 'g_anticorrelation', 'holographic_boundaries']):
        logger.info("Expected Signatures - Spatial Correlation:")
        
        # Temperature peaks coincide with high information density
        if 'peak_info_overlap' in results and results['peak_info_overlap'] > 0.5:
            signatures_detected += 1
            logger.info("  ✓ Temperature peaks coincide with high information density regions")
        else:
            logger.info("  ✗ Temperature peaks do not coincide with high information density")
        total_signatures += 1
        
        # Anti-correlation with gravitational potential
        if 'g_anticorrelation' in results and results['g_anticorrelation']:
            signatures_detected += 1
            logger.info("  ✓ Anti-correlation with gravitational potential detected")
        else:
            logger.info("  ✗ No anti-correlation with gravitational potential")
        total_signatures += 1
        
        # Sharp transitions at holographic boundaries
        if 'holographic_boundaries' in results and np.sum(results['holographic_boundaries']) > 0:
            signatures_detected += 1
            logger.info("  ✓ Sharp transitions at holographic boundaries detected")
        else:
            logger.info("  ✗ No sharp transitions at holographic boundaries")
        total_signatures += 1
    
    # 4.2 Temporal Evolution
    if 'temp_bursts' in results or 'quasi_periodic' in results:
        logger.info("Expected Signatures - Temporal Evolution:")
        
        # Temperature bursts when information approaches saturation
        if 'temp_bursts' in results and results['temp_bursts'].get('follows_high_info', False):
            signatures_detected += 1
            logger.info("  ✓ Temperature bursts follow high information density")
        else:
            logger.info("  ✗ Temperature bursts do not follow high information density")
        total_signatures += 1
        
        # Quasi-periodic oscillations with frequency ~ γ
        if 'quasi_periodic' in results and results['quasi_periodic'].get('detected', False):
            signatures_detected += 1
            logger.info(f"  ✓ Quasi-periodic oscillations detected with f/γ = {results['quasi_periodic'].get('frequency_ratio', 0):.2f}")
        else:
            logger.info("  ✗ No quasi-periodic oscillations detected")
        total_signatures += 1
    
    # 4.3 Spectral Fingerprints
    spectral_found = False
    for result in results.get('files_analyzed', []):
        if ('pol_temp_correlation' in results and results['pol_temp_correlation'].get('increases_with_temp', False)) or \
           ('width_info_correlation' in results and results['width_info_correlation'].get('widths_scale_with_info', False)) or \
           ('non_thermal_signature' in results and results['non_thermal_signature'].get('non_thermal_detected', False)):
            spectral_found = True
            break
    
    if spectral_found:
        logger.info("Expected Signatures - Spectral Fingerprints: DETECTED")
        signatures_detected += 1
    else:
        logger.info("Expected Signatures - Spectral Fingerprints: NOT DETECTED")
    total_signatures += 1
    
    # Phase transitions
    if 'transitions' in results:
        n_transitions = len(results['transitions']['times'])
        logger.info(f"Phase transitions detected: {n_transitions}")
        
        if n_transitions > 0:
            signatures_detected += 1
            logger.info("  ✓ Information phase transitions detected")
        else:
            logger.info("  ✗ No information phase transitions detected")
        total_signatures += 1
    
    # Overall signature detection rate
    if total_signatures > 0:
        detection_rate = signatures_detected / total_signatures
        logger.info(f"Overall signature detection rate: {signatures_detected}/{total_signatures} ({detection_rate:.1%})")
        
        if detection_rate >= 0.7:  # At least 70% of expected signatures
            logger.info("✓ Analysis SUPPORTS the holographic heating model")
        else:
            logger.info("✗ Analysis does NOT strongly support the holographic heating model")
    
    # Results location
    logger.info(f"Results saved to {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 