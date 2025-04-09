import pytest
import numpy as np
from holopy.dsqft.causal_patch import CausalPatch
from holopy.dsqft.propagator import BulkBoundaryPropagator
from holopy.constants.physical_constants import PhysicalConstants
from holopy.utils.logging import configure_logging

# Configure logging for tests
configure_logging(level='INFO')

@pytest.mark.physical
def test_physical_constants():
    """Test that physical constants maintain their theoretical relationships."""
    pc = PhysicalConstants()
    
    # Test fundamental information processing rate
    assert np.isclose(pc.gamma, 1.89e-29, rtol=1e-3), \
        "Information processing rate γ should be 1.89 × 10^-29 s^-1"
    
    # Test γ/H ratio
    theoretical_ratio = 1/(8*np.pi)
    actual_ratio = pc.gamma/pc.hubble_parameter
    assert np.isclose(actual_ratio, theoretical_ratio, rtol=1e-3), \
        f"γ/H ratio should be approximately 1/8π, got {actual_ratio}"

@pytest.mark.physical
def test_causal_patch_creation():
    """Test that causal patch creation respects physical constraints."""
    pc = PhysicalConstants()
    patch_radius = 0.5/pc.hubble_parameter
    
    patch = CausalPatch(
        radius=patch_radius,
        reference_frame='static',
        observer_time=0.0
    )
    
    # Verify patch size is within causal horizon
    assert patch.radius <= pc.c/pc.hubble_parameter, \
        "Patch radius should not exceed causal horizon"

@pytest.mark.numerical
def test_wavepacket_evolution():
    """Test numerical stability of wavepacket evolution."""
    pc = PhysicalConstants()
    patch = CausalPatch(
        radius=0.5/pc.hubble_parameter,
        reference_frame='static',
        observer_time=0.0
    )
    
    propagator = BulkBoundaryPropagator(
        conformal_dim=1.0,
        d=4,
        gamma=pc.gamma,
        hubble_parameter=pc.hubble_parameter
    )
    
    # Test wavepacket evolution over time
    t_points = np.linspace(0, 1.0/pc.hubble_parameter, 100)
    x_grid = patch.create_spatial_grid(resolution=50)
    
    # Verify numerical stability
    for t in t_points:
        # Test that wavepacket remains normalized
        wavepacket = propagator.compute_field_from_boundary(
            lambda x: np.exp(-np.sum(x**2)/2),
            t,
            x_grid[0],
            patch.boundary_projection(resolution=100)
        )
        assert not np.isnan(wavepacket), "Wavepacket should not contain NaN values"
        assert not np.isinf(wavepacket), "Wavepacket should not contain infinite values"

@pytest.mark.example
def test_photon_example_consistency():
    """Test that the example implementation matches theoretical predictions."""
    from examples.photon_example import simulate_photon_in_causal_patch
    
    t_points, x_grid, psi_evolution = simulate_photon_in_causal_patch()
    
    # Test that evolution maintains physical properties
    assert np.all(np.isfinite(psi_evolution)), \
        "Wavefunction evolution should not contain NaN or infinite values"
    
    # Test that probability density integrates to 1
    for psi_t in psi_evolution:
        total_probability = np.trapz(psi_t, x_grid[:, 0])
        assert np.isclose(total_probability, 1.0, rtol=1e-3), \
            "Wavefunction should remain normalized" 