import numpy as np
from holopy.dsqft.correlation import ModifiedCorrelationFunction
from holopy.dsqft.dictionary import FieldOperatorDictionary, FieldType
from holopy.constants.physical_constants import PhysicalConstants
from holopy.constants.dsqft_constants import DSQFTConstants
from holopy.cosmology.expansion import HolographicExpansion

# Use smaller numbers that won't trigger numerical instability
# and set constants to more appropriate testing values
pc = PhysicalConstants()
pc.hubble_parameter = 1.0  # Set H=1 for testing
pc.gamma = 0.01  # Set information rate to small but significant value

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

print("Testing correlation function")

# Test at three points
r1 = 0.0
r2 = 0.5
r3 = 1.0

# Calculate correlation function
correlation_values = correlation.compute_correlation_function(np.array([r1, r2, r3]))

print("Testing bulk_three_point_function")
eta = -1.0  # Conformal time
x1 = np.array([0.0, 0.0, 0.0])
x2 = np.array([0.5, 0.0, 0.0])
x3 = np.array([0.0, 0.5, 0.0])

# Use correct method signature with field_names as a list
field_names = ['scalar', 'scalar', 'scalar']
three_point = correlation.bulk_three_point_function(
    field_names,
    eta, x1, eta, x2, eta, x3
)

print(f"Three-point function value: {three_point}")

# Test symmetry by permuting points
three_point_perm = correlation.bulk_three_point_function(
    field_names,
    eta, x2, eta, x3, eta, x1
)

print(f"Three-point function with permuted points: {three_point_perm}")
print(f"Symmetry ratio: {three_point/three_point_perm:.8f}")
print("Should be close to 1.0 for proper symmetry")

# Test scaling with distance
x2_scaled = np.array([1.0, 0.0, 0.0])  # Double the distance
three_point_scaled = correlation.bulk_three_point_function(
    field_names,
    eta, x1, eta, x2_scaled, eta, x3
)

print(f"\nThree-point at original distance: {three_point}")
print(f"Three-point at doubled distance: {three_point_scaled}")
ratio = three_point / three_point_scaled
print(f"Ratio (should decrease with distance): {ratio:.8f}") 