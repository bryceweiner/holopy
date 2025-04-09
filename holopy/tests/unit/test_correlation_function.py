"""
Simple test script for the ModifiedCorrelationFunction class.
"""

import numpy as np
from holopy.cosmology.expansion import HolographicExpansion
from holopy.dsqft.correlation import ModifiedCorrelationFunction

def main():
    """Test the ModifiedCorrelationFunction with the sound_horizon method."""
    # Create a standard expansion model
    expansion = HolographicExpansion(
        omega_m=0.3, 
        omega_r=9.0e-5,
        omega_lambda=0.7,
        omega_k=0.0,
        h0=0.7,
        info_constraint=True
    )
    
    # Create a ModifiedCorrelationFunction instance
    correlation = ModifiedCorrelationFunction(
        cosmology=expansion,
        n_s=0.96,
        amplitude=2.1e-9,
        k_pivot=0.05
    )
    
    # Test transfer function calculation
    k_values = np.logspace(-3, 1, 5)  # Test with a few k values
    
    print("Testing transfer function computation:")
    for k in k_values:
        T = correlation.compute_transfer_function(k)
        print(f"  T(k={k:.6e}) = {T:.6e}")
    
    # Test power spectrum calculation
    print("\nTesting power spectrum computation:")
    z = 0.0  # Redshift
    for k in k_values:
        P = correlation.compute_power_spectrum(k, z)
        print(f"  P(k={k:.6e}, z={z}) = {P:.6e}")

if __name__ == "__main__":
    main() 