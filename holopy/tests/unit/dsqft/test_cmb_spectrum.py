"""
Tests for CMB power spectrum calculations.

These tests verify the physical correctness of the CMB power spectrum
calculations against known results from Callin (2006).
"""

import numpy as np
import pytest
from holopy.dsqft.correlation import ModifiedCorrelationFunction
from holopy.dsqft.dictionary import FieldOperatorDictionary, FieldType
from holopy.constants.physical_constants import PhysicalConstants
import logging

logger = logging.getLogger(__name__)

@pytest.fixture
def correlation():
    """Create a test instance of ModifiedCorrelationFunction."""
    pc = PhysicalConstants()
    dictionary = FieldOperatorDictionary(hubble_parameter=pc.hubble_parameter)
    dictionary.register_bulk_field('scalar', FieldType.SCALAR, mass=0.0)
    
    # Create cosmology
    from holopy.cosmology.expansion import HolographicExpansion
    cosmology = HolographicExpansion(
        omega_m=0.3,
        omega_r=9.0e-5,
        omega_lambda=0.7,
        h0=pc.hubble_parameter/100,  # Convert to h0 units
        info_constraint=True
    )
    
    return ModifiedCorrelationFunction(cosmology=cosmology)

def test_cmb_power_spectrum_physical_range(correlation):
    """Test CMB power spectrum over physically relevant multipole range."""
    # Test at key multipoles from Callin (2006)
    ell_values = np.array([2, 10, 100, 500, 1000, 2000])
    power_spectrum = correlation.cmb_power_spectrum('scalar', ell_values)
    
    # Basic physical requirements
    assert np.all(power_spectrum > 0), "Power spectrum must be positive"
    assert not np.any(np.isnan(power_spectrum)), "Power spectrum contains NaN values"
    assert not np.any(np.isinf(power_spectrum)), "Power spectrum contains infinite values"
    
    # Check COBE normalization at ℓ=10
    l10_idx = np.where(ell_values == 10)[0][0]
    C_10 = power_spectrum[l10_idx]
    assert np.isclose(C_10, 0.64575e-11, rtol=0.1), "COBE normalization incorrect at ℓ=10"

def test_cmb_power_spectrum_sachs_wolfe(correlation):
    """Test Sachs-Wolfe plateau at low multipoles."""
    # Test at low multipoles where Sachs-Wolfe effect dominates
    ell_low = np.linspace(2, 30, 10)
    power_low = correlation.cmb_power_spectrum('scalar', ell_low)
    
    # On large scales, C_ℓ should be approximately constant
    # Calculate variation in ℓ(ℓ+1)C_ℓ
    normalized_power = ell_low * (ell_low + 1) * power_low
    variation = np.std(normalized_power) / np.mean(normalized_power)
    
    # Allow for 10% variation in Sachs-Wolfe plateau
    assert variation < 0.1, "Excessive variation in Sachs-Wolfe plateau"

def test_cmb_power_spectrum_acoustic_peaks(correlation):
    """Test positions and heights of acoustic peaks."""
    # Generate fine grid around first peak
    ell_peak1 = np.linspace(180, 220, 50)
    power_peak1 = correlation.cmb_power_spectrum('scalar', ell_peak1)
    
    # First peak should be at ℓ ≈ 200
    peak1_pos = ell_peak1[np.argmax(power_peak1)]
    assert 190 <= peak1_pos <= 210, f"First peak position {peak1_pos} out of expected range"
    
    # Test peak height ratios from Callin (2006)
    ell_peaks = np.array([200, 525, 825])  # Positions of first three peaks
    power_peaks = correlation.cmb_power_spectrum('scalar', ell_peaks)
    
    # Second-to-first peak ratio should be ≈ 0.5
    ratio_21 = power_peaks[1] / power_peaks[0]
    assert 0.4 <= ratio_21 <= 0.6, f"Second-to-first peak ratio {ratio_21} out of expected range"
    
    # Third-to-first peak ratio should be ≈ 0.3
    ratio_31 = power_peaks[2] / power_peaks[0]
    assert 0.2 <= ratio_31 <= 0.4, f"Third-to-first peak ratio {ratio_31} out of expected range"

def test_cmb_power_spectrum_silk_damping(correlation):
    """Test Silk damping at high multipoles."""
    # Test at high multipoles where Silk damping dominates
    ell_high = np.linspace(1000, 2000, 50)
    power_high = correlation.cmb_power_spectrum('scalar', ell_high)
    
    # Silk damping causes exponential suppression
    # Test by fitting log(C_ℓ) vs ℓ²
    log_power = np.log(power_high)
    ell_squared = ell_high**2
    slope, _ = np.polyfit(ell_squared, log_power, 1)
    
    # Slope should be negative (damping)
    assert slope < 0, "No evidence of Silk damping at high-ℓ"
    
    # Check exponential suppression rate
    # Silk scale is around ℓ_s ≈ 1500
    damping_scale = -1.0 / slope
    assert 1e6 <= damping_scale <= 3e6, f"Silk damping scale {damping_scale} out of expected range"

def test_cmb_transfer_function_physical_accuracy(correlation):
    """
    Test the physical accuracy of the CMB transfer function.
    
    Verifies:
    1. First acoustic peak at ℓ ≈ 220
    2. Correct peak height ratios from baryon loading
    3. Proper Silk damping envelope
    4. Reionization bump at low ℓ
    """
    # Only calculate at key multipoles that test important physical features
    ell = np.array([
        5,  # Reionization bump
        10,  # COBE normalization
        200,  # First acoustic peak
        525,  # Second acoustic peak
        825,  # Third acoustic peak
        1500,  # Silk damping scale
        2000   # High-ℓ damping
    ])
    
    # Calculate power spectrum
    power_spectrum = correlation.cmb_power_spectrum("scalar", ell)
    
    # Debug: Check if we're getting reasonable values
    logger.info(f"Power spectrum min: {np.min(power_spectrum)}, max: {np.max(power_spectrum)}")
    
    # Check for NaN or inf values
    if np.any(np.isnan(power_spectrum)) or np.any(np.isinf(power_spectrum)):
        logger.error("Found NaN or inf values in power spectrum")
        nan_indices = np.where(np.isnan(power_spectrum))[0]
        inf_indices = np.where(np.isinf(power_spectrum))[0]
        if len(nan_indices) > 0:
            logger.error(f"NaN values at ℓ = {ell[nan_indices]}")
        if len(inf_indices) > 0:
            logger.error(f"Inf values at ℓ = {ell[inf_indices]}")
        pytest.fail("Power spectrum contains NaN or inf values")
    
    # 1. First acoustic peak position
    # Find the peak around ℓ ≈ 200
    peak_idx = np.argmax(power_spectrum[2:5]) + 2  # Look in range of first three peaks
    first_peak_ell = ell[peak_idx]
    logger.info(f"First peak found at ℓ = {first_peak_ell:.1f}")
    
    # Verify first peak position with 10% tolerance
    assert np.isclose(
        first_peak_ell,
        200.0,
        atol=20.0,  # 10% tolerance
        msg=f"First acoustic peak should be at ℓ ≈ 200, found at ℓ = {first_peak_ell:.1f}"
    ), f"First acoustic peak position {first_peak_ell:.1f} out of expected range"
    
    # 2. Peak height ratios from baryon loading
    peak_heights = power_spectrum[2:5]  # First three peaks
    ratio_21 = peak_heights[1] / peak_heights[0]  # Second to first
    ratio_31 = peak_heights[2] / peak_heights[0]  # Third to first
    
    logger.info(f"Peak height ratios: 2/1 = {ratio_21:.3f}, 3/1 = {ratio_31:.3f}")
    
    # From Planck 2018 results
    assert np.isclose(
        ratio_21,
        0.45,
        atol=0.05,
        msg=f"Second peak suppression incorrect: expected 0.45±0.05, got {ratio_21:.3f}"
    ), f"Second peak suppression incorrect: expected 0.45±0.05, got {ratio_21:.3f}"
    assert np.isclose(
        ratio_31,
        0.35,
        atol=0.05,
        msg=f"Third peak ratio incorrect: expected 0.35±0.05, got {ratio_31:.3f}"
    ), f"Third peak ratio incorrect: expected 0.35±0.05, got {ratio_31:.3f}"
    
    # 3. Silk damping envelope
    # Test at ℓ = 1500 and 2000
    high_ell_power = power_spectrum[5:]
    high_ell = ell[5:]
    
    # Calculate damping ratio
    damping_ratio = high_ell_power[1] / high_ell_power[0]
    expected_ratio = np.exp(-(high_ell[1]**2 - high_ell[0]**2) / (1400.0**2))
    
    logger.info(f"Silk damping ratio: {damping_ratio:.3f}, expected: {expected_ratio:.3f}")
    
    # Allow for 20% variation in damping ratio
    assert np.isclose(
        damping_ratio,
        expected_ratio,
        atol=0.2 * expected_ratio,
        msg=f"Silk damping ratio incorrect: expected {expected_ratio:.3f}±{0.2*expected_ratio:.3f}, got {damping_ratio:.3f}"
    ), f"Silk damping ratio incorrect: expected {expected_ratio:.3f}±{0.2*expected_ratio:.3f}, got {damping_ratio:.3f}"
    
    # 4. Reionization bump
    # Compare power at ℓ = 5 with first peak
    bump_amplitude = power_spectrum[0] / power_spectrum[peak_idx]
    expected_amplitude = 0.054  # From Planck 2018
    
    logger.info(f"Reionization bump amplitude: {bump_amplitude:.3f}")
    
    assert np.isclose(
        bump_amplitude,
        expected_amplitude,
        atol=0.01,
        msg=f"Reionization bump amplitude incorrect: expected {expected_amplitude:.3f}±0.01, got {bump_amplitude:.3f}"
    ), f"Reionization bump amplitude incorrect: expected {expected_amplitude:.3f}±0.01, got {bump_amplitude:.3f}" 