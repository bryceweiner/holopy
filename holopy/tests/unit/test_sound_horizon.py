"""
Simple test script for the sound_horizon method.
"""

from holopy.cosmology.expansion import HolographicExpansion

def main():
    """Test the sound_horizon method."""
    # Create a standard expansion model
    expansion = HolographicExpansion(
        omega_m=0.3, 
        omega_r=9.0e-5,
        omega_lambda=0.7,
        omega_k=0.0,
        h0=0.7,
        info_constraint=True
    )
    
    # Calculate sound horizon
    r_s = expansion.sound_horizon()
    print(f"Sound horizon (with info constraints): {r_s:.6f} Mpc")
    
    # Test with information constraints disabled
    expansion_no_info = HolographicExpansion(
        omega_m=0.3, 
        omega_r=9.0e-5,
        omega_lambda=0.7,
        omega_k=0.0,
        h0=0.7,
        info_constraint=False
    )
    
    # Calculate sound horizon without info constraints
    r_s_no_info = expansion_no_info.sound_horizon()
    print(f"Sound horizon (without info constraints): {r_s_no_info:.6f} Mpc")
    
    # Test with different cosmological parameters
    expansion_alt = HolographicExpansion(
        omega_m=0.25, 
        omega_r=8.0e-5,
        omega_lambda=0.75,
        omega_k=0.0,
        h0=0.73,
        info_constraint=True
    )
    
    # Calculate sound horizon with alternative parameters
    r_s_alt = expansion_alt.sound_horizon()
    print(f"Sound horizon (alt params): {r_s_alt:.6f} Mpc")
    
    # Test with custom recombination redshift
    r_s_custom_z = expansion.sound_horizon(z_recomb=1100.0)
    print(f"Sound horizon (z_recomb=1100): {r_s_custom_z:.6f} Mpc")

if __name__ == "__main__":
    main() 