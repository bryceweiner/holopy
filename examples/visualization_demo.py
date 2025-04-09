"""
HoloPy Visualization Demo

This script demonstrates the visualization capabilities of the HoloPy package,
showcasing various plots and visualizations for holographic physics simulations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Import holopy visualization utilities
from holopy.utils.visualization import (
    visualize_e8_projection,
    plot_wavefunction_evolution,
    plot_cmb_power_spectrum,
    plot_cosmic_evolution,
    plot_decoherence_rates,
    plot_early_universe,
    set_default_plotting_style
)

# Create output directory for saving figures
os.makedirs("figures", exist_ok=True)

# Set default style for all plots
set_default_plotting_style()

def demo_e8_projections():
    """Demonstrate E8 root system visualization."""
    print("Generating E8 projection visualizations...")
    
    # Generate some simulated E8 root vectors
    # In reality, these would come from proper E8 calculations
    np.random.seed(42)  # For reproducibility
    
    # Generate 240 random vectors in 8D space
    # Normalize them to have similar norms for visualization purposes
    root_vectors = np.random.randn(240, 8)
    norms = np.linalg.norm(root_vectors, axis=1, keepdims=True)
    root_vectors = root_vectors / norms * np.random.uniform(0.9, 1.1, size=(240, 1))
    
    # Visualize in 3D
    fig_3d = visualize_e8_projection(
        root_vectors, 
        dimension=3,
        save_path="figures/e8_projection_3d.png"
    )
    
    # Visualize in 2D with different projection
    # Create a custom projection matrix that projects onto a specific plane
    custom_proj = np.zeros((8, 2))
    custom_proj[2, 0] = 0.7
    custom_proj[3, 0] = 0.7
    custom_proj[4, 1] = 0.7
    custom_proj[5, 1] = 0.7
    
    fig_2d = visualize_e8_projection(
        root_vectors, 
        dimension=2, 
        projection_matrix=custom_proj,
        marker_size=30,
        save_path="figures/e8_projection_2d.png"
    )
    
    print("E8 projections saved to 'figures/e8_projection_3d.png' and 'figures/e8_projection_2d.png'")

def demo_wavefunction_evolution():
    """Demonstrate quantum wavefunction evolution visualization."""
    print("Generating wavefunction evolution visualizations...")
    
    # Generate sample quantum evolution data
    # In a real scenario, this would come from the quantum simulation modules
    
    # Create a grid in position space
    x_min, x_max = -10, 10
    n_points = 200
    x = np.linspace(x_min, x_max, n_points)
    
    # Time evolution parameters
    n_times = 50
    t = np.linspace(0, 5, n_times)
    
    # Create a Gaussian wavepacket that disperses and oscillates over time
    psi = np.zeros((n_times, n_points), dtype=complex)
    
    # Initial width of the wavepacket
    sigma = 1.0
    
    # Wave number (momentum)
    k = 2.0
    
    # Mass parameter for time evolution
    mass = 0.5
    
    # Generate the evolving wavefunction
    for i, t_i in enumerate(t):
        # Time-dependent width
        sigma_t = sigma * np.sqrt(1 + (t_i / (mass * sigma**2))**2)
        
        # Normalization factor
        norm = (2 * np.pi * sigma_t**2)**(-0.25)
        
        # Phase factor
        phase = k * x - 0.5 * k**2 * t_i / mass
        
        # Additional phase from dispersion
        disp_phase = -0.5 * np.arctan(t_i / (mass * sigma**2))
        
        # Construct the wavefunction
        psi[i] = norm * np.exp(-(x**2) / (4 * sigma_t**2)) * np.exp(1j * (phase + disp_phase))
    
    # Add a complexity measure (representing holographic effects on the wavefunction)
    complexity = np.zeros((n_times, n_points))
    for i, t_i in enumerate(t):
        # Example complexity measure - highest in regions of large gradients and oscillations
        psi_grad = np.gradient(np.abs(psi[i]))
        complexity[i] = np.abs(psi_grad) * (1 + 0.5 * t_i) * np.exp(-(x**2) / 25.0)
    
    # Prepare the evolution data dictionary
    evolution_data = {
        't': t,
        'psi': psi,
        'complexity': complexity
    }
    
    # Plot wavefunction at specific times
    selected_times = [0.0, 1.0, 3.0]  # Initial, middle, and later times
    fig = plot_wavefunction_evolution(
        evolution_data,
        times=selected_times,
        x_range=(x_min, x_max),
        n_points=n_points,
        show_decoherence=True,
        save_path="figures/wavefunction_evolution.png"
    )
    
    print("Wavefunction evolution saved to 'figures/wavefunction_evolution.png'")

def demo_cmb_power_spectrum():
    """Demonstrate CMB power spectrum visualization."""
    print("Generating CMB power spectrum visualizations...")
    
    # Generate sample CMB power spectrum data
    # In a real scenario, this would come from the CMB simulation modules
    
    # Create multipole values (l)
    l_min, l_max = 2, 2500
    n_l = 500
    l = np.logspace(np.log10(l_min), np.log10(l_max), n_l).astype(int)
    
    # Create TT spectrum (temperature-temperature)
    # Using a simplified model based on ΛCDM with acoustic peaks
    # Real data would come from a proper Boltzmann code
    tt_spectrum = np.zeros(n_l)
    for i, l_i in enumerate(l):
        # Base spectrum with declining power law
        base = 1000 * (l_i/100)**(-0.5) * np.exp(-l_i/1000)
        
        # Add acoustic peaks
        peaks = 0
        for peak_l, amp in [(220, 1.0), (540, 0.5), (800, 0.3), (1100, 0.2)]:
            peaks += amp * np.exp(-(np.log(l_i/peak_l))**2 * 10)
        
        # Add holographic E8×E8 effects as small modulations
        e8_effect = 1.0 + 0.05 * np.sin(l_i / 100)
        
        tt_spectrum[i] = base * (1 + peaks) * e8_effect
    
    # Create EE spectrum (polarization E-mode)
    ee_spectrum = tt_spectrum * 0.1 * (l/500)**0.5 * (1.0 + 0.03 * np.cos(l / 80))
    
    # Create TE spectrum (temperature-polarization cross-correlation)
    te_spectrum = np.sqrt(tt_spectrum * ee_spectrum) * 0.8 * np.sin(l/200) * (1.0 + 0.02 * np.sin(l / 150))
    
    # Prepare the CMB data dictionary
    cmb_data = {
        'l': l,
        'TT': tt_spectrum,
        'EE': ee_spectrum,
        'TE': te_spectrum
    }
    
    # Plot the TT spectrum
    fig_tt = plot_cmb_power_spectrum(
        cmb_data,
        spectrum_types=['TT'],
        show_conventional=True,
        save_path="figures/cmb_spectrum_TT.png"
    )
    
    # Plot all spectra
    fig_all = plot_cmb_power_spectrum(
        cmb_data,
        spectrum_types=['TT', 'EE', 'TE'],
        show_conventional=True,
        save_path="figures/cmb_spectrum_all.png"
    )
    
    print("CMB power spectra saved to 'figures/cmb_spectrum_TT.png' and 'figures/cmb_spectrum_all.png'")

def demo_cosmic_evolution():
    """Demonstrate cosmic evolution visualization."""
    print("Generating cosmic evolution visualizations...")
    
    # Generate sample cosmic evolution data
    # In a real scenario, this would come from the HolographicExpansion class
    
    # Create logarithmically spaced time values from early universe to present
    t_min, t_max = 1e-36, 4.35e17  # Planck time to 13.8 billion years in seconds
    n_t = 1000
    t = np.logspace(np.log10(t_min), np.log10(t_max), n_t)
    
    # Create scale factor evolution (normalized to a=1 at present)
    # Simple model with radiation, matter, and dark energy eras
    a = np.zeros(n_t)
    for i, t_i in enumerate(t):
        # Define transition times
        t_eq = 5e11  # Radiation-matter equality at ~50,000 years
        t_acc = 2e16  # Acceleration begins at ~7 billion years
        
        # Different growth rates in different eras
        if t_i < t_eq:
            # Radiation dominated: a ∝ t^(1/2)
            a[i] = 1e-12 * (t_i/t_min)**0.5
        elif t_i < t_acc:
            # Matter dominated: a ∝ t^(2/3)
            a[i] = 1e-4 * (t_i/t_eq)**0.667
        else:
            # Dark energy dominated: a ∝ exp(H₀t)
            h0 = 2.2e-18  # Hubble constant in s^-1
            a[i] = 0.3 * np.exp(h0 * (t_i - t_acc))
    
    # Normalize to a=1 at present time
    a = a / a[-1]
    
    # Calculate Hubble parameter H = ȧ/a
    log_a = np.log(a)
    dlog_a_dt = np.gradient(log_a, t)
    h = dlog_a_dt
    
    # Prepare the expansion data dictionary
    expansion_data = {
        't': t,
        'a': a,
        'h': h
    }
    
    # Plot scale factor evolution
    fig_scale = plot_cosmic_evolution(
        expansion_data,
        plot_type='scale_factor',
        save_path="figures/cosmic_scale_factor.png"
    )
    
    # Plot Hubble parameter evolution
    fig_hubble = plot_cosmic_evolution(
        expansion_data,
        plot_type='hubble',
        hubble_tension=True,
        save_path="figures/cosmic_hubble.png"
    )
    
    # Plot deceleration parameter evolution
    fig_decel = plot_cosmic_evolution(
        expansion_data,
        plot_type='deceleration',
        save_path="figures/cosmic_deceleration.png"
    )
    
    print("Cosmic evolution figures saved to 'figures/cosmic_*.png'")

def demo_decoherence_rates():
    """Demonstrate decoherence rates visualization."""
    print("Generating decoherence rates visualization...")
    
    # Generate sample decoherence rate data
    # In a real scenario, this would come from quantum simulations
    
    # Create system sizes
    system_sizes = np.logspace(1, 3, 20)  # Sizes from 10 to 1000
    
    # Generate decoherence rates with theoretical L^-2 scaling plus noise
    coeff = 0.05
    rates = coeff / system_sizes**2
    
    # Add some random noise to simulate measurement uncertainty
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, 0.1, size=len(rates))
    rates = rates * (1 + noise)
    
    # Ensure all rates are positive
    rates = np.abs(rates)
    
    # Plot decoherence rates
    fig_rates = plot_decoherence_rates(
        system_sizes,
        rates,
        theoretical_curve=True,
        save_path="figures/decoherence_rates.png"
    )
    
    print("Decoherence rates figure saved to 'figures/decoherence_rates.png'")

def demo_early_universe():
    """Demonstrate early universe simulation visualization."""
    print("Generating early universe simulation visualizations...")
    
    # Generate sample early universe simulation data
    # In a real scenario, this would come from the early_universe module
    
    # Inflation data
    n_infl = 500
    t_infl = np.logspace(-36, -32, n_infl)  # Inflation time range
    phi = np.zeros(n_infl)
    
    # Simple slow-roll inflation model
    phi_start = 3.0  # Initial inflaton field value
    for i, t_i in enumerate(t_infl):
        # Exponential decay of the inflaton
        phi[i] = phi_start * np.exp(-1e32 * (t_i - t_infl[0]))
    
    # Reheating data
    n_reh = 500
    t_reh = np.logspace(-32, -25, n_reh)  # Reheating time range
    
    # Energy densities during reheating
    rho_phi = np.zeros(n_reh)  # Inflaton energy density
    rho_r = np.zeros(n_reh)   # Radiation energy density
    T = np.zeros(n_reh)       # Temperature
    
    # Simple reheating model with energy transfer
    rho_phi_start = 1e64  # Initial energy density in inflaton
    for i, t_i in enumerate(t_reh):
        # Decay rate of inflaton to radiation
        gamma = 1e27  # Decay rate
        
        # Inflaton energy decays exponentially
        rho_phi[i] = rho_phi_start * np.exp(-gamma * (t_i - t_reh[0]))
        
        # Radiation builds up and then dilutes due to expansion
        # Simple model: first increases as inflaton decays, then decreases as a^-4
        buildup = 1 - np.exp(-gamma * (t_i - t_reh[0]))
        dilution = (t_reh[0] / t_i)**(2/3)  # Assumes matter-dominated expansion
        rho_r[i] = rho_phi_start * buildup * dilution
        
        # Temperature is related to radiation energy density
        # T ∝ ρ_r^(1/4)
        T[i] = 1e15 * (rho_r[i] / rho_phi_start)**(1/4)  # In GeV
    
    # Prepare the simulation results dictionary
    simulation_results = {
        'inflation': {
            't': t_infl,
            'phi': phi
        },
        'reheating': {
            't': t_reh,
            'rho_phi': rho_phi,
            'rho_r': rho_r,
            'T': T
        }
    }
    
    # Define critical transitions
    critical_transitions = [
        {'name': 'End of Inflation', 'time': 1e-32},
        {'name': 'End of Reheating', 'time': 1e-26}
    ]
    
    # Plot energy densities during reheating
    fig_energy = plot_early_universe(
        simulation_results,
        plot_type='energy_densities',
        critical_transitions=critical_transitions,
        save_path="figures/early_universe_energy.png"
    )
    
    # Plot temperature evolution
    fig_temp = plot_early_universe(
        simulation_results,
        plot_type='temperature',
        critical_transitions=critical_transitions,
        save_path="figures/early_universe_temperature.png"
    )
    
    # Plot inflaton field evolution
    fig_infl = plot_early_universe(
        simulation_results,
        plot_type='inflaton',
        save_path="figures/early_universe_inflaton.png"
    )
    
    print("Early universe simulation figures saved to 'figures/early_universe_*.png'")

def main():
    """Run all visualization demos."""
    print("Running HoloPy visualization demonstrations...")
    
    # Run each demo function
    demo_e8_projections()
    demo_wavefunction_evolution()
    demo_cmb_power_spectrum()
    demo_cosmic_evolution()
    demo_decoherence_rates()
    demo_early_universe()
    
    print("\nAll visualizations completed and saved to the 'figures' directory.")

if __name__ == "__main__":
    main() 