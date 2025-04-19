# PowerShell script to download and prepare PUNCH data for holographic analysis
# This script downloads sample PUNCH polarimetric data for testing the analysis script

# Configuration
$DATA_DIR = "$PWD\punch_data"
$LOG_FILE = "$DATA_DIR\download.log"
$START_DATE = (Get-Date).AddDays(-30) # Last 30 days
$END_DATE = Get-Date

# Create directories
New-Item -ItemType Directory -Path $DATA_DIR -Force | Out-Null

# Function to log messages
function Write-Log {
    param (
        [string]$Message
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp $Message" | Out-File -Append -FilePath $LOG_FILE
    Write-Host "$timestamp $Message"
}

# Check dependencies - Verify PowerShell version and modules
function Test-Dependencies {
    Write-Log "Checking dependencies..."
    
    # Check PowerShell version
    $PSVersion = $PSVersionTable.PSVersion.Major
    if ($PSVersion -lt 5) {
        Write-Log "ERROR: PowerShell 5.0 or higher is required. Current version: $PSVersion"
        exit 1
    }
    
    # Check for required modules
    if (-not (Get-Module -ListAvailable -Name Microsoft.PowerShell.Utility)) {
        Write-Log "ERROR: Required PowerShell module not available: Microsoft.PowerShell.Utility"
        exit 1
    }
    
    # Check for Python dependencies
    try {
        # Check if sunpy is available (required for VSO access)
        $sunpy = python -c "import sunpy; print('OK')" 2>$null
        if ($sunpy -eq "OK") {
            Write-Log "SunPy is available for VSO data access"
            $script:HAS_SUNPY = $true
        } else {
            Write-Log "ERROR: SunPy not available. Install with: pip install sunpy"
            Write-Log "SunPy is required for accessing PUNCH data via Virtual Solar Observatory"
            $script:HAS_SUNPY = $false
        }
        
        # Check for astropy
        $astropy = python -c "import astropy; print('OK')" 2>$null
        if ($astropy -eq "OK") {
            Write-Log "Astropy is available for FITS processing"
            $script:HAS_ASTROPY = $true
        } else {
            Write-Log "WARNING: Astropy not available. FITS file creation will be limited."
            $script:HAS_ASTROPY = $false
        }
        
        # Check for numpy
        $numpy = python -c "import numpy; print('OK')" 2>$null
        if ($numpy -eq "OK") {
            Write-Log "NumPy is available for data processing"
            $script:HAS_NUMPY = $true
        } else {
            Write-Log "WARNING: NumPy not available. Simulated data generation will be limited."
            $script:HAS_NUMPY = $false
        }
    } catch {
        Write-Log "ERROR: Python checks failed. Make sure Python with SunPy is installed."
        exit 1
    }
}

# Fetch PUNCH data from Virtual Solar Observatory using SunPy
function Get-PunchData {
    Write-Log "Downloading PUNCH data using SunPy Fido client..."
    
    # Format dates for Python
    $start_date_str = $START_DATE.ToString("yyyy-MM-dd")
    $end_date_str = $END_DATE.ToString("yyyy-MM-dd")
    
    # Create Python script for VSO access
    $tempScript = New-TemporaryFile
    $tempScript = Rename-Item -Path $tempScript -NewName "$($tempScript.Name).py" -PassThru
    
    # Write Python script content
    @"
import os
import sys
from datetime import datetime
from sunpy.net import Fido, attrs as a

# Output directory
data_dir = r"$DATA_DIR"
os.makedirs(data_dir, exist_ok=True)

# Configure date range
start_date = "$start_date_str"
end_date = "$end_date_str"

print(f"Searching for PUNCH data from {start_date} to {end_date}")

try:
    # Query VSO for PUNCH data
    time_range = a.Time(start_date, end_date)
    instrument = a.Instrument('PUNCH')  # Adjust if needed
    physobs = a.Physobs('polarized_intensity')  # Adjust if needed
    
    # Execute query
    print("Querying Virtual Solar Observatory...")
    result = Fido.search(time_range, instrument, physobs)
    
    if len(result) == 0:
        print("No PUNCH data found for the specified date range")
        print("This is expected if searching for data before the PUNCH mission launch")
        print("Generating simulated data instead")
        sys.exit(0)
    
    print(f"Found {len(result)} PUNCH data files")
    
    # Download files
    print("Downloading files...")
    downloaded_files = Fido.fetch(result, path=os.path.join(data_dir, '{file}'))
    print(f"Downloaded {len(downloaded_files)} files to {data_dir}")
    
    # List downloaded files
    print("\nDownloaded files:")
    for file in downloaded_files:
        print(f"  - {os.path.basename(file)}")
    
except Exception as e:
    print(f"Error accessing VSO: {str(e)}")
    print("This may be due to PUNCH data not yet being available in VSO")
    print("Will generate simulated data instead")
    sys.exit(1)
"@ | Out-File -FilePath $tempScript.FullName -Encoding utf8

    # Execute the Python script
    try {
        Write-Log "Executing SunPy data retrieval script..."
        $output = python $tempScript.FullName 2>&1
        $output | ForEach-Object { Write-Log $_ }
        
        # Check if any files were downloaded
        $downloaded = $output -match "Downloaded .* files"
        if ($downloaded) {
            Write-Log "Successfully downloaded PUNCH data using SunPy"
            $success = $true
        } else {
            Write-Log "No PUNCH data was downloaded. VSO search returned no results."
            $success = $false
        }
    } catch {
        Write-Log "Error executing SunPy data retrieval: $_"
        $success = $false
    }
    
    # Clean up temporary script
    Remove-Item -Path $tempScript.FullName -Force
    
    return $success
}

# Generate sample simulated data for testing
function New-SimulatedData {
    Write-Log "Generating simulated PUNCH polarimetric data for testing..."
    
    # Loop through each day in the date range
    $current_date = $START_DATE
    while ($current_date -le $END_DATE) {
        $date_str = $current_date.ToString("yyyyMMdd")
        $filename = "punch_${date_str}_polarimetric.fits"
        $filepath = Join-Path -Path $DATA_DIR -ChildPath $filename
        
        # Create Python script to generate the file
        $tempScript = New-TemporaryFile
        $tempScript = Rename-Item -Path $tempScript -NewName "$($tempScript.Name).py" -PassThru
        
        # Write Python script content
        @"
import numpy as np
import os
from datetime import datetime

# Check if astropy is available for FITS creation
try:
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    HAS_FITS = True
except ImportError:
    HAS_FITS = False
    print("Astropy not available, creating NPY file instead")

# Create simulated PUNCH polarimetric data
date_str = "$date_str"
image_size = 256  # 256x256 pixel image
r_sun = 256 // 4  # Solar radius in pixels

# Create coordinate grid
y, x = np.ogrid[-image_size//2:image_size//2, -image_size//2:image_size//2]
r = np.sqrt(x*x + y*y)

# Create polarized intensity images (Stokes parameters)
# Stokes I - Total intensity
I = np.exp(-(r-r_sun*1.5)**2/(2*(r_sun*0.5)**2))
I = I / I.max()  # Normalize

# Add some random structures
np.random.seed(int(date_str[-6:]))  # Use date as seed for reproducibility
for i in range(5):
    # Random blobs to simulate structures
    cx = np.random.randint(-image_size//3, image_size//3)
    cy = np.random.randint(-image_size//3, image_size//3)
    size = np.random.randint(r_sun//4, r_sun//2)
    intensity = np.random.uniform(0.2, 0.5)
    
    # Add structure to image
    struct = intensity * np.exp(-((x-cx)**2 + (y-cy)**2)/(2*size**2))
    I += struct

# Stokes Q - Linear polarization
Q = I * 0.1 * np.cos(np.arctan2(y, x) * 2)

# Stokes U - Linear polarization
U = I * 0.1 * np.sin(np.arctan2(y, x) * 2)

# Add time-dependent variations (to allow for time-series analysis)
day_factor = float(date_str[-2:]) / 30.0  # Normalized day in month
time_variation = 0.2 * np.sin(r/r_sun * np.pi * day_factor)
I *= (1 + time_variation)
Q *= (1 + time_variation)
U *= (1 + time_variation)

# Ensure output directory exists
os.makedirs("$DATA_DIR", exist_ok=True)

if HAS_FITS:
    # Create FITS header with proper WCS for solar data
    hdr = fits.Header()
    # Standard FITS keywords
    hdr['SIMPLE'] = True
    hdr['BITPIX'] = -64
    hdr['NAXIS'] = 2
    hdr['NAXIS1'] = image_size
    hdr['NAXIS2'] = image_size
    
    # Date and observation information
    hdr['DATE-OBS'] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T00:00:00.000"
    hdr['DATE'] = datetime.now().isoformat()
    
    # WCS keywords for solar coordinates (helioprojective Cartesian)
    hdr['WCSNAME'] = 'HELIOPROJECTIVE'
    hdr['CTYPE1'] = 'HPLN-TAN'  # Helioprojective longitude
    hdr['CTYPE2'] = 'HPLT-TAN'  # Helioprojective latitude
    hdr['CUNIT1'] = 'arcsec'
    hdr['CUNIT2'] = 'arcsec'
    # Center of the sun in the image
    hdr['CRPIX1'] = image_size // 2 + 0.5  # Center pixel X
    hdr['CRPIX2'] = image_size // 2 + 0.5  # Center pixel Y
    hdr['CDELT1'] = 4.0  # 4 arcsec per pixel resolution
    hdr['CDELT2'] = 4.0
    hdr['CRVAL1'] = 0.0  # Sun center (degrees)
    hdr['CRVAL2'] = 0.0
    
    # Instrument-specific keywords
    hdr['TELESCOP'] = 'PUNCH'
    hdr['INSTRUME'] = 'NFI+WFI'
    hdr['OBJECT'] = 'SOLAR CORONA'
    hdr['DATATYPE'] = 'POLARIMETRIC'
    hdr['CREATOR'] = 'HoloPy PUNCH Data Simulator'
    hdr['CREATDAT'] = datetime.now().isoformat()
    hdr['DESCRIPT'] = 'Simulated PUNCH polarimetric data for holographic analysis'
    
    # Create primary HDU with Stokes I
    primary_hdu = fits.PrimaryHDU(I, header=hdr)
    
    # Create extensions for Stokes Q and U
    q_hdu = fits.ImageHDU(Q, name='Q')
    u_hdu = fits.ImageHDU(U, name='U')
    
    # Create HDU list and write to file
    hdul = fits.HDUList([primary_hdu, q_hdu, u_hdu])
    hdul.writeto("$filepath", overwrite=True)
    print(f"Created FITS file: $filepath with solar coordinate system")
else:
    # Save as NPY files instead
    np.save("${filepath}.I.npy", I)
    np.save("${filepath}.Q.npy", Q)
    np.save("${filepath}.U.npy", U)
    print(f"Created NPY files: ${filepath}.*.npy")
"@ | Out-File -FilePath $tempScript.FullName -Encoding utf8

        # Execute the Python script
        try {
            $output = python $tempScript.FullName 2>&1
            Write-Log "Generated file for date $date_str"
            Write-Log $output
        } catch {
            Write-Log "Error generating file for date $date_str : $_"
        }
        
        # Clean up temporary script
        Remove-Item -Path $tempScript.FullName -Force
        
        # Move to next day
        $current_date = $current_date.AddDays(1)
    }
}

# Convert NPY files to FITS if needed
function Convert-NpyToFits {
    Write-Log "Checking for NPY files to convert to FITS..."
    
    # Find all NPY files
    $npy_files = Get-ChildItem -Path $DATA_DIR -Filter "*.I.npy"
    
    if ($npy_files.Count -gt 0 -and $script:HAS_ASTROPY) {
        Write-Log "Found $($npy_files.Count) NPY files to convert to FITS"
        
        # Create Python script to convert the files
        $tempScript = New-TemporaryFile
        $tempScript = Rename-Item -Path $tempScript -NewName "$($tempScript.Name).py" -PassThru
        
        # Write Python script content
        @"
import numpy as np
import os
import glob
from astropy.io import fits
from datetime import datetime

# Find all I.npy files
i_files = glob.glob("$DATA_DIR/*.I.npy")

for i_file in i_files:
    try:
        # Get base name and date string
        base_name = i_file[:-6]  # Remove .I.npy
        date_str = os.path.basename(base_name).split('_')[1]
        
        # Load Stokes parameters
        I = np.load(f"{base_name}.I.npy")
        Q = np.load(f"{base_name}.Q.npy")
        U = np.load(f"{base_name}.U.npy")
        
        # Get image dimensions
        image_size = I.shape[0]
        
        # Create FITS header with proper WCS for solar data
        hdr = fits.Header()
        # Standard FITS keywords
        hdr['SIMPLE'] = True
        hdr['BITPIX'] = -64
        hdr['NAXIS'] = 2
        hdr['NAXIS1'] = image_size
        hdr['NAXIS2'] = image_size
        
        # Date and observation information
        hdr['DATE-OBS'] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T00:00:00.000"
        hdr['DATE'] = datetime.now().isoformat()
        
        # WCS keywords for solar coordinates (helioprojective Cartesian)
        hdr['WCSNAME'] = 'HELIOPROJECTIVE'
        hdr['CTYPE1'] = 'HPLN-TAN'  # Helioprojective longitude
        hdr['CTYPE2'] = 'HPLT-TAN'  # Helioprojective latitude
        hdr['CUNIT1'] = 'arcsec'
        hdr['CUNIT2'] = 'arcsec'
        # Center of the sun in the image
        hdr['CRPIX1'] = image_size // 2 + 0.5  # Center pixel X
        hdr['CRPIX2'] = image_size // 2 + 0.5  # Center pixel Y
        hdr['CDELT1'] = 4.0  # 4 arcsec per pixel resolution
        hdr['CDELT2'] = 4.0
        hdr['CRVAL1'] = 0.0  # Sun center (degrees)
        hdr['CRVAL2'] = 0.0
        
        # Instrument-specific keywords
        hdr['TELESCOP'] = 'PUNCH'
        hdr['INSTRUME'] = 'NFI+WFI'
        hdr['OBJECT'] = 'SOLAR CORONA'
        hdr['DATATYPE'] = 'POLARIMETRIC'
        hdr['CREATOR'] = 'HoloPy PUNCH Data Simulator'
        hdr['CREATDAT'] = datetime.now().isoformat()
        hdr['DESCRIPT'] = 'Simulated PUNCH polarimetric data for holographic analysis'
        
        # Create primary HDU with Stokes I
        primary_hdu = fits.PrimaryHDU(I, header=hdr)
        
        # Create extensions for Stokes Q and U
        q_hdu = fits.ImageHDU(Q, name='Q')
        u_hdu = fits.ImageHDU(U, name='U')
        
        # Create HDU list and write to file
        fits_file = f"{base_name}.fits"
        hdul = fits.HDUList([primary_hdu, q_hdu, u_hdu])
        hdul.writeto(fits_file, overwrite=True)
        print(f"Converted {i_file} to {fits_file}")
        
        # Remove the NPY files
        os.remove(f"{base_name}.I.npy")
        os.remove(f"{base_name}.Q.npy")
        os.remove(f"{base_name}.U.npy")
    except Exception as e:
        print(f"Error converting {i_file}: {str(e)}")
"@ | Out-File -FilePath $tempScript.FullName -Encoding utf8

        # Execute the Python script
        try {
            $output = python $tempScript.FullName 2>&1
            Write-Log "Conversion complete"
            Write-Log $output
        } catch {
            Write-Log "Error converting NPY files: $_"
        }
        
        # Clean up temporary script
        Remove-Item -Path $tempScript.FullName -Force
    } elseif ($npy_files.Count -gt 0) {
        Write-Log "NPY files found but Astropy is not available for conversion. Files will remain in NPY format."
    } else {
        Write-Log "No NPY files found for conversion."
    }
}

# Main function
function Main {
    Write-Log "Starting PUNCH data preparation..."
    
    # Check dependencies
    Test-Dependencies
    
    # Attempt to download data via SunPy/VSO
    if ($script:HAS_SUNPY) {
        $downloaded = Get-PunchData
    } else {
        Write-Log "SunPy not available. Cannot access VSO to download PUNCH data."
        $downloaded = $false
    }
    
    # If download failed or SunPy not available, generate simulated data
    if (-not $downloaded) {
        Write-Log "Using simulated data instead of real PUNCH observations"
        New-SimulatedData
        
        # Convert NPY to FITS if necessary
        Convert-NpyToFits
    }
    
    # Count and verify files
    $fits_files = Get-ChildItem -Path $DATA_DIR -Filter "*.fits" | Measure-Object | Select-Object -ExpandProperty Count
    $npy_files = Get-ChildItem -Path $DATA_DIR -Filter "*.I.npy" | Measure-Object | Select-Object -ExpandProperty Count
    
    Write-Log "Data preparation complete."
    Write-Log "FITS files prepared: $fits_files"
    if ($npy_files -gt 0) {
        Write-Log "NPY files prepared: $npy_files"
        Write-Log "Note: NPY files will need to be manually converted to FITS or the analysis script modified to handle them."
    }
    
    Write-Log "To analyze this data, run:"
    Write-Log "python punch_heliophysics_analysis.py --data-dir $DATA_DIR"
}

# Run the main function
Main 