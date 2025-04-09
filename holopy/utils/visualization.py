"""
Visualization Utilities for HoloPy.

This module provides visualization functions for plotting E8 projections,
quantum states, and cosmological data in the holographic framework.
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Union, Dict, List, Tuple, Callable

# Setup logging
logger = logging.getLogger(__name__)

def set_default_plotting_style():
    """
    Set default matplotlib style for HoloPy plots.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['lines.linewidth'] = 2
    
    logger.debug("Set default HoloPy plotting style")

def visualize_e8_projection(
    root_vectors: np.ndarray,
    dimension: int = 3,
    projection_matrix: Optional[np.ndarray] = None,
    show_labels: bool = False,
    marker_size: int = 20,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize projection of E8 root system into lower dimensions.
    
    Args:
        root_vectors (np.ndarray): E8 root vectors, shape (240, 8)
        dimension (int, optional): Dimension of projection (2 or 3)
        projection_matrix (np.ndarray, optional): Custom projection matrix
        show_labels (bool, optional): Whether to show root labels
        marker_size (int, optional): Size of markers in the plot
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Figure object
    """
    # Check inputs
    if dimension not in [2, 3]:
        raise ValueError("Dimension must be 2 or 3")
    
    # Set default plotting style
    set_default_plotting_style()
    
    # Create default projection matrix if none provided
    if projection_matrix is None:
        if dimension == 2:
            # Project onto first two dimensions
            projection_matrix = np.zeros((8, 2))
            projection_matrix[0, 0] = 1.0
            projection_matrix[1, 1] = 1.0
        else:
            # Project onto first three dimensions
            projection_matrix = np.zeros((8, 3))
            projection_matrix[0, 0] = 1.0
            projection_matrix[1, 1] = 1.0
            projection_matrix[2, 2] = 1.0
    
    # Project root vectors
    projected_roots = root_vectors @ projection_matrix
    
    # Create plot
    fig = plt.figure(figsize=(12, 10))
    
    if dimension == 2:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(
            projected_roots[:, 0],
            projected_roots[:, 1],
            c=np.linalg.norm(root_vectors, axis=1),
            cmap=cm.viridis,
            s=marker_size,
            alpha=0.7
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Root norm')
        
        # Set labels
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title('2D Projection of E8 Root System')
        
        # Add labels if requested
        if show_labels:
            for i, (x, y) in enumerate(projected_roots):
                ax.annotate(str(i), (x, y), fontsize=8)
    
    else:  # 3D plot
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            projected_roots[:, 0],
            projected_roots[:, 1],
            projected_roots[:, 2],
            c=np.linalg.norm(root_vectors, axis=1),
            cmap=cm.viridis,
            s=marker_size,
            alpha=0.7
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Root norm')
        
        # Set labels
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title('3D Projection of E8 Root System')
        
        # Add labels if requested
        if show_labels:
            for i, (x, y, z) in enumerate(projected_roots):
                ax.text(x, y, z, str(i), fontsize=8)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved E8 projection visualization to {save_path}")
    
    logger.debug(f"Visualized E8 projection in {dimension}D")
    return fig

def plot_wavefunction_evolution(
    evolution_data: Dict[str, np.ndarray],
    times: List[float],
    x_range: Tuple[float, float],
    n_points: int = 100,
    show_decoherence: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the evolution of a wavefunction over time.
    
    Args:
        evolution_data (Dict[str, np.ndarray]): Evolution data from ModifiedSchrodinger
        times (List[float]): Times at which to plot the wavefunction
        x_range (Tuple[float, float]): Range of x values to plot
        n_points (int, optional): Number of points for plotting
        show_decoherence (bool, optional): Whether to show decoherence effects
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Figure object
    """
    # Set default plotting style
    set_default_plotting_style()
    
    # Extract data
    t = evolution_data.get('t', None)
    psi = evolution_data.get('psi', None)
    complexity = evolution_data.get('complexity', None)
    
    if t is None or psi is None:
        raise ValueError("Evolution data must contain 't' and 'psi' arrays")
    
    # Create x points for evaluation
    x = np.linspace(x_range[0], x_range[1], n_points)
    
    # Create figure
    fig, axes = plt.subplots(len(times), 2, figsize=(12, 4 * len(times)))
    
    # Handle single time case
    if len(times) == 1:
        axes = axes.reshape(1, 2)
    
    # Plot for each time
    for i, plot_time in enumerate(times):
        # Find closest time in the data
        time_idx = np.argmin(np.abs(t - plot_time))
        actual_time = t[time_idx]
        
        # Get wavefunction at this time
        psi_t = psi[time_idx]
        
        # Evaluate wavefunction at the x points
        # This would normally require the full wavefunction object
        # For now, we assume psi_t contains the values at x points
        psi_values = psi_t
        
        # Calculate probability density
        prob_density = np.abs(psi_values)**2
        
        # Plot probability density
        axes[i, 0].plot(x, prob_density, 'b-', label='Probability density')
        axes[i, 0].set_xlabel('Position (x)')
        axes[i, 0].set_ylabel('|ψ|²')
        axes[i, 0].set_title(f'Probability density at t = {actual_time:.3f}')
        
        # Plot real and imaginary parts
        axes[i, 1].plot(x, np.real(psi_values), 'r-', label='Re(ψ)')
        axes[i, 1].plot(x, np.imag(psi_values), 'g-', label='Im(ψ)')
        
        # Also plot decoherence effects if available and requested
        if show_decoherence and complexity is not None:
            complexity_t = complexity[time_idx]
            # Scale for better visualization
            scaled_complexity = 0.5 * complexity_t / np.max(complexity_t)
            axes[i, 1].plot(x, scaled_complexity, 'k--', label='Complexity (scaled)')
        
        axes[i, 1].set_xlabel('Position (x)')
        axes[i, 1].set_ylabel('Amplitude')
        axes[i, 1].set_title(f'Wavefunction components at t = {actual_time:.3f}')
        axes[i, 1].legend()
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved wavefunction evolution plot to {save_path}")
    
    logger.debug(f"Plotted wavefunction evolution at {len(times)} time points")
    return fig

def plot_cmb_power_spectrum(
    cmb_data: Dict[str, np.ndarray],
    spectrum_types: List[str] = ['TT'],
    include_e8_effects: bool = True,
    show_conventional: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the CMB power spectrum with and without E8×E8 effects.
    
    Args:
        cmb_data (Dict[str, np.ndarray]): CMB power spectrum data
        spectrum_types (List[str], optional): Which spectra to plot
        include_e8_effects (bool, optional): Whether to include E8×E8 effects
        show_conventional (bool, optional): Whether to show conventional ΛCDM model
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Figure object
    """
    # Set default plotting style
    set_default_plotting_style()
    
    # Extract data
    l = cmb_data.get('l', None)
    
    if l is None:
        raise ValueError("CMB data must contain 'l' array")
    
    # Validate spectrum types
    valid_types = ['TT', 'EE', 'TE']
    for spec_type in spectrum_types:
        if spec_type not in valid_types:
            raise ValueError(f"Invalid spectrum type: {spec_type}. Must be one of {valid_types}")
        if spec_type not in cmb_data:
            raise ValueError(f"CMB data does not contain '{spec_type}' spectrum")
    
    # Create figure with the appropriate number of subplots
    fig, axes = plt.subplots(len(spectrum_types), 1, figsize=(12, 6 * len(spectrum_types)))
    
    # Handle single spectrum case
    if len(spectrum_types) == 1:
        axes = [axes]  # Wrap in a list rather than an array to ensure it's not a NumPy array
    
    # Plot each spectrum
    for i, spec_type in enumerate(spectrum_types):
        # Get data
        data = cmb_data[spec_type]
        
        # Compute D_l = l(l+1)C_l/(2π)
        D_l = l * (l + 1) * data / (2 * np.pi)
        
        # Plot with E8×E8 effects
        axes[i].plot(l, D_l, 'b-', label=f'{spec_type} with E8×E8 effects')
        
        # Plot conventional model if requested
        if show_conventional:
            # This would normally come from the data
            # For now, we create a simplified conventional model
            if spec_type == 'TT':
                # Simple model of TT spectrum without E8×E8 effects
                conventional = D_l / (1.0 + 0.05 * np.sin(l / 500 * np.pi))
            elif spec_type == 'EE':
                # Simple model of EE spectrum without E8×E8 effects
                conventional = D_l / (1.0 + 0.03 * np.sin(l / 500 * np.pi))
            elif spec_type == 'TE':
                # Simple model of TE spectrum without E8×E8 effects
                conventional = D_l / (1.0 + 0.02 * np.sin(l / 500 * np.pi))
            
            axes[i].plot(l, conventional, 'r--', label=f'{spec_type} conventional model')
        
        # Set axis properties
        axes[i].set_xlabel('Multipole $\\ell$')
        axes[i].set_ylabel('$\\mathcal{D}_\\ell = \\ell(\\ell+1)C_\\ell/2\\pi$ [$\\mu K^2$]')
        axes[i].set_title(f'{spec_type} Power Spectrum')
        axes[i].set_xscale('log')
        axes[i].legend()
        
        # Add grid
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved CMB power spectrum plot to {save_path}")
    
    logger.debug(f"Plotted CMB power spectrum with {len(spectrum_types)} spectrum types")
    return fig

def plot_cosmic_evolution(
    expansion_data: Dict[str, np.ndarray],
    plot_type: str = 'scale_factor',
    hubble_tension: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the cosmic evolution according to the holographic expansion model.
    
    Args:
        expansion_data (Dict[str, np.ndarray]): Expansion data from HolographicExpansion
        plot_type (str, optional): Type of plot ('scale_factor', 'hubble', etc.)
        hubble_tension (bool, optional): Whether to show Hubble tension effects
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Figure object
    """
    # Set default plotting style
    set_default_plotting_style()
    
    # Extract data
    t = expansion_data.get('t', None)
    a = expansion_data.get('a', None)
    h = expansion_data.get('h', None)
    
    if t is None or a is None:
        raise ValueError("Expansion data must contain 't' and 'a' arrays")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if plot_type == 'scale_factor':
        # Plot scale factor evolution
        ax.plot(t, a, 'b-', label='Scale factor (a)')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Scale factor')
        ax.set_title('Universe Scale Factor Evolution')
        
        # Use log scale for better visualization
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    elif plot_type == 'hubble':
        # Check if Hubble parameter is available
        if h is None:
            raise ValueError("Expansion data must contain 'h' array for Hubble plot")
        
        # Convert to km/s/Mpc for better readability
        h_kms_mpc = h * 3.086e22 / 1000
        
        # Plot Hubble parameter evolution
        ax.plot(t, h_kms_mpc, 'r-', label='Hubble parameter')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Hubble parameter (km/s/Mpc)')
        ax.set_title('Hubble Parameter Evolution')
        
        # Use log scale for better visualization
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Add Hubble tension reference if requested
        if hubble_tension:
            # Add horizontal lines for Planck and local H0 measurements
            h0_planck = 67.4  # km/s/Mpc
            h0_local = 73.2   # km/s/Mpc
            
            ax.axhline(h0_planck, color='g', linestyle='--', 
                       label='Planck CMB H₀ = 67.4 km/s/Mpc')
            ax.axhline(h0_local, color='m', linestyle='--', 
                       label='Local H₀ = 73.2 km/s/Mpc')
    
    elif plot_type == 'deceleration':
        # Plot deceleration parameter q = -ä*a/ȧ²
        # We compute this from the scale factor
        
        # Compute derivatives
        loga = np.log(a)
        dloga_dt = np.gradient(loga, t)
        d2loga_dt2 = np.gradient(dloga_dt, t)
        
        # Compute deceleration parameter
        q = -1 - d2loga_dt2 / dloga_dt**2
        
        # Plot deceleration parameter
        ax.plot(t, q, 'g-', label='Deceleration parameter (q)')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Deceleration parameter')
        ax.set_title('Universe Deceleration Parameter Evolution')
        
        # Use log scale for time
        ax.set_xscale('log')
        
        # Add reference line at q=0 (transition from deceleration to acceleration)
        ax.axhline(0, color='k', linestyle='--', label='q = 0')
    
    else:
        raise ValueError(f"Invalid plot type: {plot_type}")
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved cosmic evolution plot to {save_path}")
    
    logger.debug(f"Plotted cosmic evolution with plot type '{plot_type}'")
    return fig

def plot_decoherence_rates(
    system_sizes: np.ndarray,
    rates: np.ndarray,
    theoretical_curve: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot decoherence rates scaling with system size.
    
    Args:
        system_sizes (np.ndarray): System sizes (L)
        rates (np.ndarray): Measured decoherence rates
        theoretical_curve (bool, optional): Whether to show theoretical L^-2 scaling
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Figure object
    """
    # Set default plotting style
    set_default_plotting_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot measured rates
    ax.scatter(system_sizes, rates, c='b', marker='o', s=50, 
              label='Numerical results')
    
    # Plot theoretical curve if requested
    if theoretical_curve:
        # Fit the L^-2 curve to the data
        coeff = np.mean(rates * system_sizes**2)
        theoretical = coeff / system_sizes**2
        
        ax.plot(system_sizes, theoretical, 'r-', 
                label=r'Theoretical scaling ($L^{-2}$)')
    
    # Set axis properties
    ax.set_xlabel('System size (L)')
    ax.set_ylabel('Decoherence rate')
    ax.set_title('Decoherence Rate Scaling with System Size')
    
    # Use log-log scale for better visualization
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved decoherence rates plot to {save_path}")
    
    logger.debug(f"Plotted decoherence rates for {len(system_sizes)} system sizes")
    return fig

def plot_early_universe(
    simulation_results: Dict[str, Dict[str, np.ndarray]],
    plot_type: str = 'energy_densities',
    critical_transitions: Optional[List[Dict[str, float]]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot early universe evolution from simulation results.
    
    Args:
        simulation_results (Dict[str, Dict[str, np.ndarray]]): Results from simulate_early_universe
        plot_type (str, optional): Type of plot
        critical_transitions (List[Dict[str, float]], optional): List of critical transitions
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Figure object
    """
    # Set default plotting style
    set_default_plotting_style()
    
    # Extract inflation and reheating data
    inflation = simulation_results.get('inflation', {})
    reheating = simulation_results.get('reheating', {})
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if plot_type == 'energy_densities':
        # Extract reheating data
        t_reh = reheating.get('t', None)
        rho_phi = reheating.get('rho_phi', None)
        rho_r = reheating.get('rho_r', None)
        
        if t_reh is None or rho_phi is None or rho_r is None:
            raise ValueError("Reheating data must contain 't', 'rho_phi', and 'rho_r' arrays")
        
        # Plot energy densities
        ax.plot(t_reh, rho_phi, 'b-', label='Inflaton energy density')
        ax.plot(t_reh, rho_r, 'r-', label='Radiation energy density')
        ax.plot(t_reh, rho_phi + rho_r, 'k--', label='Total energy density')
        
        # Set axis properties
        ax.set_xlabel('Time (Planck units)')
        ax.set_ylabel('Energy density (Planck units)')
        ax.set_title('Energy Densities During Reheating')
        
        # Use log scales
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    elif plot_type == 'temperature':
        # Extract reheating data
        t_reh = reheating.get('t', None)
        T = reheating.get('T', None)
        
        if t_reh is None or T is None:
            raise ValueError("Reheating data must contain 't' and 'T' arrays")
        
        # Plot temperature evolution
        ax.plot(t_reh, T, 'r-', label='Temperature')
        
        # Set axis properties
        ax.set_xlabel('Time (Planck units)')
        ax.set_ylabel('Temperature (GeV)')
        ax.set_title('Temperature Evolution During Reheating')
        
        # Use log scales
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    elif plot_type == 'inflaton':
        # Extract inflation data
        t_infl = inflation.get('t', None)
        phi = inflation.get('phi', None)
        
        if t_infl is None or phi is None:
            raise ValueError("Inflation data must contain 't' and 'phi' arrays")
        
        # Plot inflaton field evolution
        ax.plot(t_infl, phi, 'b-', label='Inflaton field value')
        
        # Set axis properties
        ax.set_xlabel('Time (Planck units)')
        ax.set_ylabel('Inflaton field value (Planck units)')
        ax.set_title('Inflaton Field Evolution During Inflation')
    
    else:
        raise ValueError(f"Invalid plot type: {plot_type}")
    
    # Add critical transitions if provided
    if critical_transitions:
        for transition in critical_transitions:
            name = transition.get('name', 'Transition')
            time = transition.get('time', None)
            
            if time is not None:
                ax.axvline(time, color='g', linestyle='--', label=name)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved early universe plot to {save_path}")
    
    logger.debug(f"Plotted early universe with plot type '{plot_type}'")
    return fig

def plot_root_system(
    root_system: np.ndarray,
    dimension: int = 3,
    projection_matrix: Optional[np.ndarray] = None,
    show_connections: bool = True,
    connection_threshold: float = 1.5,
    marker_size: int = 30,
    show_labels: bool = False,
    highlight_roots: Optional[List[int]] = None,
    title: str = "Root System Visualization",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a general root system with options for projections and connections.
    
    This is a more general version of visualize_e8_projection that works with
    any root system, not just E8.
    
    Args:
        root_system (np.ndarray): Root vectors, shape (n_roots, dimension)
        dimension (int, optional): Dimension to project to (2 or 3)
        projection_matrix (np.ndarray, optional): Custom projection matrix
        show_connections (bool, optional): Whether to draw lines between roots
        connection_threshold (float, optional): Maximum distance to draw connections
        marker_size (int, optional): Size of markers in the plot
        show_labels (bool, optional): Whether to show root indices
        highlight_roots (List[int], optional): Indices of roots to highlight
        title (str, optional): Plot title
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Figure object
    """
    # Check inputs
    if dimension not in [2, 3]:
        raise ValueError("Dimension must be 2 or 3")
    
    # Get original dimension of root system
    orig_dim = root_system.shape[1]
    
    # Set default plotting style
    set_default_plotting_style()
    
    # Create default projection matrix if none provided
    if projection_matrix is None:
        if dimension == 2:
            # Project onto first two dimensions
            projection_matrix = np.zeros((orig_dim, 2))
            projection_matrix[0, 0] = 1.0
            projection_matrix[1, 1] = 1.0
        else:  # dimension == 3
            # Project onto first three dimensions
            projection_matrix = np.zeros((orig_dim, 3))
            projection_matrix[0, 0] = 1.0
            projection_matrix[1, 1] = 1.0
            if orig_dim > 2:
                projection_matrix[2, 2] = 1.0
    
    # Project root vectors
    projected_roots = root_system @ projection_matrix
    
    # Create plot
    fig = plt.figure(figsize=(12, 10))
    
    # Set up colors - default and highlighted
    colors = np.linalg.norm(root_system, axis=1)
    
    # Prepare highlight mask if needed
    highlight_mask = np.zeros(len(root_system), dtype=bool)
    if highlight_roots is not None:
        highlight_mask[highlight_roots] = True
    
    if dimension == 2:
        ax = fig.add_subplot(111)
        
        # Draw connections first (so they appear behind points)
        if show_connections:
            # Calculate pairwise distances in projected space
            for i in range(len(projected_roots)):
                for j in range(i+1, len(projected_roots)):
                    dist = np.linalg.norm(projected_roots[i] - projected_roots[j])
                    if dist < connection_threshold:
                        ax.plot([projected_roots[i, 0], projected_roots[j, 0]],
                               [projected_roots[i, 1], projected_roots[j, 1]],
                               color='gray', alpha=0.3, linewidth=1)
        
        # Plot regular roots
        scatter_regular = ax.scatter(
            projected_roots[~highlight_mask, 0],
            projected_roots[~highlight_mask, 1],
            c=colors[~highlight_mask] if any(~highlight_mask) else [],
            cmap=cm.viridis,
            s=marker_size,
            alpha=0.7
        )
        
        # Plot highlighted roots if any
        if highlight_roots:
            scatter_highlight = ax.scatter(
                projected_roots[highlight_mask, 0],
                projected_roots[highlight_mask, 1],
                c='red',
                s=marker_size*1.5,
                alpha=0.9,
                marker='*'
            )
        
        # Add colorbar
        cbar = plt.colorbar(scatter_regular)
        cbar.set_label('Root norm')
        
        # Set labels
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(title)
        
        # Add labels if requested
        if show_labels:
            for i, (x, y) in enumerate(projected_roots):
                ax.annotate(str(i), (x, y), fontsize=8)
    
    else:  # 3D plot
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw connections first
        if show_connections:
            for i in range(len(projected_roots)):
                for j in range(i+1, len(projected_roots)):
                    dist = np.linalg.norm(projected_roots[i] - projected_roots[j])
                    if dist < connection_threshold:
                        ax.plot([projected_roots[i, 0], projected_roots[j, 0]],
                               [projected_roots[i, 1], projected_roots[j, 1]],
                               [projected_roots[i, 2], projected_roots[j, 2]],
                               color='gray', alpha=0.3, linewidth=1)
        
        # Plot regular roots
        scatter_regular = ax.scatter(
            projected_roots[~highlight_mask, 0],
            projected_roots[~highlight_mask, 1],
            projected_roots[~highlight_mask, 2],
            c=colors[~highlight_mask] if any(~highlight_mask) else [],
            cmap=cm.viridis,
            s=marker_size,
            alpha=0.7
        )
        
        # Plot highlighted roots if any
        if highlight_roots:
            scatter_highlight = ax.scatter(
                projected_roots[highlight_mask, 0],
                projected_roots[highlight_mask, 1],
                projected_roots[highlight_mask, 2],
                c='red',
                s=marker_size*1.5,
                alpha=0.9,
                marker='*'
            )
        
        # Add colorbar
        cbar = plt.colorbar(scatter_regular)
        cbar.set_label('Root norm')
        
        # Set labels
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title(title)
        
        # Add labels if requested
        if show_labels:
            for i, (x, y, z) in enumerate(projected_roots):
                ax.text(x, y, z, str(i), fontsize=8)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved root system visualization to {save_path}")
    
    logger.debug(f"Visualized root system in {dimension}D")
    return fig 