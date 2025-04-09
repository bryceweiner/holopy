"""
Utility Functions for HoloPy.

This module provides various utility functions used throughout the HoloPy library,
including mathematical operations, visualization, and logging utilities.
"""

from holopy.utils.math_utils import (
    tensor_contraction,
    numerical_gradient,
    metrics_equivalent
)

from holopy.utils.tensor_utils import (
    raise_index,
    lower_index,
    symmetrize,
    antisymmetrize,
    compute_christoffel_symbols,
    compute_riemann_tensor,
    kill_indices,
    compute_gradient,
    compute_divergence,
    compute_laplacian
)

from holopy.utils.visualization import (
    set_default_plotting_style,
    visualize_e8_projection,
    plot_wavefunction_evolution,
    plot_cmb_power_spectrum,
    plot_cosmic_evolution,
    plot_decoherence_rates,
    plot_early_universe,
    plot_root_system
)

from holopy.utils.logging import (
    get_logger,
    log_execution_time,
    ProgressTracker,
    configure_logging
)

# Define what gets imported with "from holopy.utils import *"
__all__ = [
    # From math_utils
    'tensor_contraction',
    'numerical_gradient',
    'metrics_equivalent',
    
    # From tensor_utils
    'raise_index',
    'lower_index',
    'symmetrize',
    'antisymmetrize',
    'compute_christoffel_symbols',
    'compute_riemann_tensor',
    'kill_indices',
    'compute_gradient',
    'compute_divergence',
    'compute_laplacian',
    
    # From visualization
    'set_default_plotting_style',
    'visualize_e8_projection',
    'plot_wavefunction_evolution',
    'plot_cmb_power_spectrum',
    'plot_cosmic_evolution',
    'plot_decoherence_rates',
    'plot_early_universe',
    'plot_root_system',
    
    # From logging
    'get_logger',
    'log_execution_time',
    'ProgressTracker',
    'configure_logging'
] 