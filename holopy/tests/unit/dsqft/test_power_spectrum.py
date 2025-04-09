import numpy as np
import matplotlib.pyplot as plt
from holopy.dsqft.correlation import ModifiedCorrelationFunction
from holopy.dsqft.dictionary import FieldOperatorDictionary, FieldType
from holopy.constants.physical_constants import PhysicalConstants
from holopy.cosmology.expansion import HolographicExpansion

# Set constants to more appropriate testing values
pc = PhysicalConstants()
pc.hubble_parameter = 1.0  # Set H=1 for testing
pc.gamma = 0.01  # Set γ to small but significant value

# Create dictionary with a massless field for simplicity
dictionary = FieldOperatorDictionary(hubble_parameter=1.0)
dictionary.register_bulk_field('scalar', FieldType.SCALAR, mass=0.0)  # Massless field

# Create cosmology
cosmology = HolographicExpansion(
    omega_m=0.3,
    omega_r=9.0e-5,
    omega_lambda=0.7,
    h0=1.0,
    info_constraint=True
)

# Create correlation function
correlation = ModifiedCorrelationFunction(cosmology=cosmology)

print("Testing power spectrum")
# Test power spectrum at a single k value
k = 1.0
power = correlation.compute_power_spectrum(k)
print(f"Power spectrum at k={k}: {power}")

# Expected result for massless field:
# P(k) = A_s * (k/k_0)^(n_s-1) * T(k)^2 * D(z=0)^2
k_0 = 0.05  # Pivot scale
n_s = 0.96  # Spectral index
A_s = 2.1e-9  # Amplitude
expected = A_s * (k/k_0)**(n_s-1)  # Just the primordial part
print(f"Expected primordial value: {expected}")
print(f"Ratio (actual/primordial): {power/expected:.8f}")

# Test k dependence with an array of k values
k_values = np.logspace(-1, 1, 5)  # Test 5 values from 0.1 to 10
powers = correlation.compute_power_spectrum(k_values)

print("\nTesting k dependence:")
print(f"k values: {k_values}")
print(f"Power spectrum values: {powers}")

# For massless field, should scale as k^(n_s-1) at large k
if isinstance(powers, np.ndarray) and len(powers) > 1:
    ratio_high_k = powers[-2] / powers[-1]
    expected_ratio = (k_values[-1]/k_values[-2])**(1-n_s)
    print(f"\nLarge k scaling - Ratio of powers for k={k_values[-2]} and k={k_values[-1]}: {ratio_high_k:.4f}")
    print(f"Expected ratio for k^(n_s-1) scaling: {expected_ratio:.4f}")

print("\nAll tests completed") 