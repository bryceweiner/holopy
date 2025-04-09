import sys
sys.path.append('.')

from holopy.cosmology.expansion import HolographicExpansion
import numpy as np
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(name)s:%(message)s'
)

def test_growth_factor():
    """Test the linear growth factor calculation with physical requirements."""
    
    # Create cosmology with physically motivated parameters
    cosmo = HolographicExpansion(
        omega_m=0.3,  # Matter density
        omega_r=9.0e-5,  # Radiation density (matches CMB temperature)
        omega_lambda=0.7,  # Dark energy density
        h0=0.7  # Hubble parameter
    )
    
    logger = logging.getLogger("test_growth")
    logger.info("Testing growth factor with cosmology: Ωm=0.3, Ωr=9.0e-5, ΩΛ=0.7, h0=0.7")
    
    # Test at physically significant redshifts
    z_test = np.array([
        0.0,    # Today
        0.5,    # Dark energy domination
        1.0,    # Matter-dark energy transition
        5.0,    # Matter domination
        1100.0, # Recombination
        3400.0  # Matter-radiation equality
    ])
    
    logger.debug(f"Testing redshifts: {z_test}")
    
    # Calculate growth factors
    D = cosmo.growth_factor(z_test)
    
    print("\nGrowth factors D(z):")
    for zi, Di in zip(z_test, D):
        print(f"z = {zi:.1f}: D = {Di:.6f}")
    
    print("\nVerifying physical requirements:")
    
    # 1. Normalization: D(z=0) = 1 exactly
    print(f"D(z=0) = {D[0]:.6f} (should be 1.0)")
    norm_error = np.abs(D[0] - 1.0)
    logger.debug(f"Normalization error: {norm_error:.6e}")
    assert norm_error < 1e-10, "Growth factor not properly normalized"
    
    # 2. Monotonic decrease with redshift (structure grows over time)
    diffs = np.diff(D)
    monotonic = all(diffs < 0)
    print("Growth factor decreases with z:", monotonic)
    
    # Print detailed monotonicity information
    for i in range(len(z_test)-1):
        ratio = D[i+1]/D[i]
        logger.debug(f"D(z={z_test[i+1]:.1f}) / D(z={z_test[i]:.1f}) = {ratio:.6f} ({'Decreasing' if ratio < 1 else 'Increasing!'})")
        if ratio >= 1:
            logger.warning(f"Non-monotonic behavior between z={z_test[i]:.1f} and z={z_test[i+1]:.1f}")
    
    # Verify monotonicity
    try:
        assert monotonic, "Growth factor not monotonically decreasing with z"
    except AssertionError as e:
        logger.error(f"Monotonicity test failed: {e}")
        # Find where monotonicity is violated
        non_monotonic_indices = np.where(diffs >= 0)[0]
        for idx in non_monotonic_indices:
            logger.error(f"Non-monotonic at indices {idx}-{idx+1}: z={z_test[idx]:.1f}-{z_test[idx+1]:.1f}, D={D[idx]:.6f}-{D[idx+1]:.6f}")
        raise
    
    # 3. Matter domination scaling
    z_md = 5.0  # Deep in matter era
    D_md = cosmo.growth_factor(z_md)
    a_md = 1.0/(1.0 + z_md)
    ratio_md = D_md/a_md
    
    print(f"\nMatter domination (z=5):")
    print(f"D(z) = {D_md:.6f}")
    print(f"a(z) = {a_md:.6f}")
    print(f"D/a ratio = {ratio_md:.6f}")
    
    # Should follow D ∝ a but with some suppression from dark energy
    try:
        assert 0.8 < ratio_md < 1.0, "Incorrect matter era scaling"
        logger.info(f"Matter era scaling verified: D/a = {ratio_md:.6f}")
    except AssertionError as e:
        logger.error(f"Matter era scaling test failed: D/a = {ratio_md:.6f}")
        raise
    
    # 4. Radiation era suppression
    z_rad = 3400.0  # Matter-radiation equality
    D_rad = cosmo.growth_factor(z_rad)
    a_rad = 1.0/(1.0 + z_rad)
    ratio_rad = D_rad/a_rad
    
    print(f"\nRadiation era (z=3400):")
    print(f"D(z) = {D_rad:.6f}")
    print(f"a(z) = {a_rad:.6f}")
    print(f"D/a ratio = {ratio_rad:.6f}")
    print(f"Logarithmic threshold = {0.1:.6f}")
    
    # Growth should be logarithmic: D ∝ ln(a/a_eq)
    # This means D(z) << a(z) in radiation era
    try:
        assert D_rad < 0.1*a_rad, "Missing logarithmic suppression in radiation era"
        logger.info(f"Radiation era suppression verified: D/a = {ratio_rad:.6f} < 0.1")
    except AssertionError as e:
        logger.error(f"Radiation era test failed: D/a = {ratio_rad:.6f} >= 0.1")
        raise
    
    # 5. Holographic modification
    if cosmo.info_constraint:
        # Growth should be suppressed by information processing constraint
        gamma = cosmo.gamma
        h0 = cosmo.h0_si
        gamma_over_h = gamma/h0
        
        # Test holographic suppression factor at z=0
        expected_suppression = np.exp(-gamma/(2*h0))
        D_ratio = D[0]/expected_suppression
        
        print(f"\nHolographic modification:")
        print(f"γ/H ratio = {gamma_over_h:.6e} (should be ≈ 1/8π = {1/(8*np.pi):.6e})")
        print(f"Suppression factor = {expected_suppression:.6f}")
        print(f"D(z=0)/suppression = {D_ratio:.6f}")
        
        # Should match holographic suppression after normalization
        try:
            assert np.abs(D_ratio - 1.0) < 0.1, "Incorrect holographic modification"
            logger.info(f"Holographic modification verified: ratio = {D_ratio:.6f}")
        except AssertionError as e:
            logger.error(f"Holographic test failed: ratio = {D_ratio:.6f}")
            raise
    
    print("\nAll physical tests passed!")

if __name__ == "__main__":
    test_growth_factor() 