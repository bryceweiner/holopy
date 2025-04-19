# PUNCH Heliophysics Analysis with HoloPy

This example script demonstrates how to use the HoloPy framework to analyze data from NASA's Polarimetric Unified Coronal and Heliospheric Imager (PUNCH) mission for testing the holographic thermodynamic regulation hypothesis of coronal heating.

## Overview

The PUNCH mission observes the Sun's outer corona and solar wind using polarized light, providing new insights into how the corona is heated and how the solar wind is accelerated. This example script implements the analysis protocol outlined in the research paper: "Holographic Thermodynamic Regulation Analysis Using PUNCH Data."

The script's holographic framework predicts that coronal heating represents a fundamental mechanism where the cosmic screen prevents localized information saturation via thermodynamics, following the relation:

```
T ∝ ρ_info/g
```

where T is the coronal temperature, ρ_info is the local information density derived from polarimetric data, and g is the gravitational potential.

## Requirements

- Python 3.6+
- HoloPy (latest version)
- sunpy 5.0+ (for solar data handling)
- astropy (for FITS file handling and coordinate calculations)
- NumPy
- Matplotlib
- SciPy

## Data Requirements

The script expects PUNCH polarimetric FITS files with the following characteristics:
- 4-minute cadence
- Background-subtracted
- Contains Stokes I, Q, and U parameters
- Covering 1.5° (6 R☉) to 45° (180 R☉)
- Combined NFI + WFI fields of view

## Usage

```powershell
# Navigate to the examples directory
cd examples

# Run the analysis with default settings (past 30 days data)
python punch_heliophysics_analysis.py

# Run with custom data directory and output directory
python punch_heliophysics_analysis.py --data-dir /path/to/punch/data --output-dir /path/to/results

# Analyze only the past 7 days
python punch_heliophysics_analysis.py --days 7

# Attempt to download data from VSO (Virtual Solar Observatory)
python punch_heliophysics_analysis.py --download
```

## Data Retrieval

The script can automatically retrieve PUNCH data from the Virtual Solar Observatory (VSO) using sunpy's Fido client:

```python
from sunpy.net import Fido, attrs as a

# Search for PUNCH data in a specific time range
time_range = a.Time('2023-01-01', '2023-01-31')
instrument = a.Instrument('PUNCH')
physobs = a.Physobs('polarized_intensity')

# Execute query
result = Fido.search(time_range, instrument, physobs)

# Download files
downloaded_files = Fido.fetch(result, path='punch_data/{file}')
```

## Data Directory Structure

If not retrieving directly from VSO, the script expects PUNCH data files in the following format:
- `punch_YYYYMMDD_polarimetric.fits`

Where `YYYYMMDD` is the date in ISO format.

## Output

The script generates the following outputs in the specified output directory:

### Per-file Outputs:
- `figures/{filename}_polarization.png`: Polarization degree map with solar limb
- `figures/{filename}_info_density.png`: Information density map
- `figures/{filename}_predicted_temp.png`: Predicted temperature map
- `figures/{filename}_analysis.png`: 2x2 panel showing:
  - Information density map
  - Predicted temperature map
  - Information density vs. temperature scatter plot
  - Information density distribution histogram
- `figures/{filename}_radial.png`: Radial profiles of information density and temperature

### Time Series Outputs:
- `figures/time_series_averages.png`: 2x2 panel showing:
  - Mean information density map
  - Mean predicted temperature map
  - Information density variability
  - Temperature variability
- `figures/phase_transitions.png`: Map of information phase transition locations
- `figures/transition_times.png`: Histogram of phase transition times

## Algorithm Description

The script implements the following key algorithms:

1. **Data Handling with sunpy**:
   - Retrieve PUNCH data using sunpy's Fido client
   - Load data as sunpy.map.Map objects for proper coordinate handling
   - Utilize SkyCoord for solar-specific coordinate transformations

2. **Information Density Extraction**:
   - Calculate degree of polarization p = √(Q²+U²)/I
   - Compute spatial polarization gradient ∇p
   - Derive information density ρ_info = |∇p|² / (γ)

3. **Temperature Correlation Analysis**:
   - Compute gravitational potential g(r) based on solar coordinates
   - Calculate predicted temperature T_pred = ρ_info/g
   - Generate correlation statistics

4. **Phase Transition Detection**:
   - Track regions of high Thomson scattering
   - Monitor information density evolution
   - Detect phase transitions when ρ_info · τ = nπ/2
   - Correlate with temperature variations

## References

1. NASA PUNCH Mission: [https://punch.space.swri.edu/](https://punch.space.swri.edu/)
2. Data access: [https://punch.space.swri.edu/data_access.php](https://punch.space.swri.edu/data_access.php)
3. SunPy Documentation: [https://docs.sunpy.org/](https://docs.sunpy.org/)
4. Virtual Solar Observatory: [https://sdac.virtualsolar.org/](https://sdac.virtualsolar.org/)

## Contributing

Feel free to contribute improvements to this example by submitting pull requests to the HoloPy repository. 