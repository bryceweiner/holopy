"""
Tests for the HolographicExpansion class.
"""

import unittest
import numpy as np
from holopy.cosmology.expansion import HolographicExpansion

class TestHolographicExpansion(unittest.TestCase):
    """Test suite for HolographicExpansion class."""
    
    def test_sound_horizon(self):
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
        
        # Check that sound horizon has a reasonable value based on actual observed results
        self.assertGreater(r_s, 1.0, "Sound horizon is too small")
        self.assertLess(r_s, 10.0, "Sound horizon is too large")
        
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
        
        # Check that sound horizon has a reasonable value
        self.assertGreater(r_s_no_info, 1.0, "Sound horizon (no info) is too small")
        self.assertLess(r_s_no_info, 10.0, "Sound horizon (no info) is too large")
        
        # With information constraints, sound horizon should be smaller
        self.assertLessEqual(r_s, r_s_no_info, "Information constraints should reduce sound horizon")
        
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
        
        # Check that sound horizon has a reasonable value
        self.assertGreater(r_s_alt, 1.0, "Sound horizon (alt params) is too small")
        self.assertLess(r_s_alt, 10.0, "Sound horizon (alt params) is too large")
        
        # Test with custom recombination redshift
        r_s_custom_z = expansion.sound_horizon(z_recomb=1100.0)
        
        # Should be similar but not identical to default
        self.assertLess(abs(r_s - r_s_custom_z), 0.5, 
                        "Sound horizon shouldn't change dramatically with small z_recomb change")

if __name__ == '__main__':
    unittest.main() 