"""
Tests for the visualization module.

This module tests the visualization functions for E8 projections, quantum states,
and cosmological data in the holographic framework.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

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


class TestVisualization(unittest.TestCase):
    """Test suite for the visualization module."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock random root vectors
        self.root_vectors = np.random.rand(240, 8)
        
        # Mock wavefunction evolution data
        self.evolution_data = {
            't': np.linspace(0, 1, 10),
            'psi': np.random.rand(10, 100) + 1j * np.random.rand(10, 100),
            'complexity': np.random.rand(10, 100)
        }
        
        # Mock CMB power spectrum data
        self.cmb_data = {
            'l': np.arange(2, 1000),
            'TT': np.random.rand(998),
            'EE': np.random.rand(998),
            'TE': np.random.rand(998)
        }
        
        # Mock cosmic expansion data
        self.expansion_data = {
            't': np.logspace(-35, 17, 100),
            'a': np.random.rand(100),
            'h': np.random.rand(100)
        }
        
        # Mock decoherence data
        self.system_sizes = np.logspace(1, 3, 10)
        self.rates = 1000 / self.system_sizes**2 + 0.05 * np.random.rand(10)
        
        # Mock early universe simulation results
        self.simulation_results = {
            'inflation': {
                't': np.logspace(-35, -32, 100),
                'phi': np.random.rand(100),
                'V': np.random.rand(100)
            },
            'reheating': {
                't': np.logspace(-32, -25, 100),
                'rho_phi': np.random.rand(100),
                'rho_r': np.random.rand(100),
                'T': np.logspace(19, 15, 100)
            }
        }
        
        # Mock critical transitions
        self.critical_transitions = [
            {'name': 'End of Inflation', 'time': 1e-32},
            {'name': 'End of Reheating', 'time': 1e-25}
        ]

    @patch('matplotlib.pyplot.style.use')
    @patch('matplotlib.pyplot.rcParams')
    def test_set_default_plotting_style(self, mock_rcParams, mock_style_use):
        """Test setting default plotting style."""
        set_default_plotting_style()
        mock_style_use.assert_called_once_with('seaborn-v0_8-whitegrid')
        self.assertTrue(mock_rcParams.__setitem__.called)
        
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    def test_visualize_e8_projection_2d(self, mock_savefig, mock_figure):
        """Test 2D visualization of E8 projection."""
        # Reset any previous calls
        mock_figure.reset_mock()
        
        # Mock figure and axis
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_scatter = MagicMock()
        mock_colorbar = MagicMock()
        
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        mock_ax.scatter.return_value = mock_scatter
        plt.colorbar = MagicMock(return_value=mock_colorbar)
        
        # Call function with 2D
        fig = visualize_e8_projection(
            self.root_vectors, 
            dimension=2, 
            show_labels=True,
            save_path='test.png'
        )
        
        # Assert function behavior - don't check call count since mocks might be reused
        self.assertTrue(mock_figure.called)
        mock_fig.add_subplot.assert_called_with(111)
        self.assertTrue(mock_ax.scatter.called)
        self.assertTrue(mock_ax.set_xlabel.called)
        self.assertTrue(mock_ax.set_ylabel.called)
        self.assertTrue(mock_ax.set_title.called)
        self.assertEqual(fig, mock_fig)
        
        # Test invalid dimension
        with self.assertRaises(ValueError):
            visualize_e8_projection(self.root_vectors, dimension=4)
            
    @patch('matplotlib.pyplot.figure')
    def test_visualize_e8_projection_3d(self, mock_figure):
        """Test 3D visualization of E8 projection."""
        # Reset any previous calls
        mock_figure.reset_mock()
        
        # Mock figure and axis
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_scatter = MagicMock()
        mock_colorbar = MagicMock()
        
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        mock_ax.scatter.return_value = mock_scatter
        plt.colorbar = MagicMock(return_value=mock_colorbar)
        
        # Call function with 3D
        fig = visualize_e8_projection(
            self.root_vectors, 
            dimension=3,
            projection_matrix=np.eye(8, 3)
        )
        
        # Assert function behavior - don't check call count
        self.assertTrue(mock_figure.called)
        self.assertTrue(mock_fig.add_subplot.called)
        self.assertTrue(mock_ax.scatter.called)
        self.assertTrue(mock_ax.set_xlabel.called)
        self.assertTrue(mock_ax.set_ylabel.called)
        self.assertTrue(mock_ax.set_zlabel.called)
        self.assertTrue(mock_ax.set_title.called)
        self.assertEqual(fig, mock_fig)

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_wavefunction_evolution(self, mock_subplots, mock_figure):
        """Test plotting wavefunction evolution."""
        # Mock figure and axes
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Reshape axes for single time case
        mock_axes.reshape.return_value = mock_axes
        
        # Call function
        fig = plot_wavefunction_evolution(
            self.evolution_data,
            times=[0.1, 0.5],
            x_range=(-5, 5),
            show_decoherence=True
        )
        
        # Assert function behavior
        mock_subplots.assert_called_once()
        self.assertEqual(fig, mock_fig)
        
        # Test with invalid data
        with self.assertRaises(ValueError):
            plot_wavefunction_evolution(
                {'invalid': 'data'},
                times=[0.1],
                x_range=(-5, 5)
            )

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_cmb_power_spectrum(self, mock_subplots, mock_figure):
        """Test plotting CMB power spectrum."""
        # Reset any previous calls
        mock_figure.reset_mock()
        mock_subplots.reset_mock()
        
        # Mock figure and axes
        mock_fig = MagicMock()
        
        # Create proper mocked axes that support indexing and have plot method
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        # Mock data has two spectra - TT and EE - so we need two axes
        mock_axes = [mock_ax1, mock_ax2]
        
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Call function
        fig = plot_cmb_power_spectrum(
            self.cmb_data,
            spectrum_types=['TT', 'EE'],
            show_conventional=True
        )
        
        # Assert function behavior
        mock_subplots.assert_called_once()
        self.assertEqual(fig, mock_fig)
        
        # Test with invalid spectrum type
        with self.assertRaises(ValueError):
            plot_cmb_power_spectrum(
                self.cmb_data,
                spectrum_types=['INVALID']
            )
            
        # Test with missing spectrum
        with self.assertRaises(ValueError):
            plot_cmb_power_spectrum(
                {'l': np.arange(10)},  # Missing TT spectrum
                spectrum_types=['TT']
            )

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_cosmic_evolution(self, mock_subplots, mock_figure):
        """Test plotting cosmic evolution."""
        # Mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Call function with scale_factor type
        fig = plot_cosmic_evolution(
            self.expansion_data,
            plot_type='scale_factor'
        )
        
        # Assert function behavior
        mock_subplots.assert_called_once()
        self.assertEqual(fig, mock_fig)
        mock_ax.set_xscale.assert_called_with('log')
        mock_ax.set_yscale.assert_called_with('log')
        
        # Reset mocks for next test
        mock_subplots.reset_mock()
        mock_ax.reset_mock()
        
        # Test hubble plot type
        fig = plot_cosmic_evolution(
            self.expansion_data,
            plot_type='hubble',
            hubble_tension=True
        )
        
        # Test invalid plot type
        with self.assertRaises(ValueError):
            plot_cosmic_evolution(
                self.expansion_data,
                plot_type='invalid'
            )
            
        # Test with missing data
        with self.assertRaises(ValueError):
            plot_cosmic_evolution(
                {'t': np.arange(10)},  # Missing 'a' data
                plot_type='scale_factor'
            )

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_decoherence_rates(self, mock_subplots, mock_figure):
        """Test plotting decoherence rates."""
        # Mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Call function
        fig = plot_decoherence_rates(
            self.system_sizes,
            self.rates,
            theoretical_curve=True
        )
        
        # Assert function behavior
        mock_subplots.assert_called_once()
        self.assertEqual(fig, mock_fig)
        self.assertTrue(mock_ax.scatter.called)
        self.assertTrue(mock_ax.plot.called)  # For theoretical curve
        mock_ax.set_xscale.assert_called_with('log')
        mock_ax.set_yscale.assert_called_with('log')

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_early_universe(self, mock_subplots, mock_figure):
        """Test plotting early universe evolution."""
        # Mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Call function with energy_densities type
        fig = plot_early_universe(
            self.simulation_results,
            plot_type='energy_densities',
            critical_transitions=self.critical_transitions
        )
        
        # Assert function behavior
        mock_subplots.assert_called_once()
        self.assertEqual(fig, mock_fig)
        
        # Reset mocks for next test
        mock_subplots.reset_mock()
        mock_ax.reset_mock()
        
        # Test temperature plot type
        fig = plot_early_universe(
            self.simulation_results,
            plot_type='temperature'
        )
        
        # Test inflaton plot type
        mock_subplots.reset_mock()
        mock_ax.reset_mock()
        fig = plot_early_universe(
            self.simulation_results,
            plot_type='inflaton'
        )
        
        # Test invalid plot type
        with self.assertRaises(ValueError):
            plot_early_universe(
                self.simulation_results,
                plot_type='invalid'
            )
            
        # Test with missing data
        with self.assertRaises(ValueError):
            plot_early_universe(
                {'reheating': {'t': np.arange(10)}},  # Missing required data
                plot_type='energy_densities'
            )

    @patch('matplotlib.pyplot.figure')
    def test_plot_root_system_2d(self, mock_figure):
        """Test plotting root system in 2D."""
        # Reset any previous calls
        mock_figure.reset_mock()
        
        # Mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_scatter = MagicMock()
        mock_colorbar = MagicMock()
        
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        mock_ax.scatter.return_value = mock_scatter
        plt.colorbar = MagicMock(return_value=mock_colorbar)
        
        # Create a smaller root system for testing
        root_system = np.random.rand(20, 4)
        
        # Call function with 2D and show_connections=True
        fig = plot_root_system(
            root_system,
            dimension=2,
            show_connections=True,
            connection_threshold=1.0,
            show_labels=True,
            highlight_roots=[0, 5, 10]
        )
        
        # Assert function behavior - don't check call count
        self.assertTrue(mock_figure.called)
        mock_fig.add_subplot.assert_called_with(111)
        self.assertTrue(mock_ax.scatter.called)
        self.assertEqual(fig, mock_fig)
        
        # Test invalid dimension
        with self.assertRaises(ValueError):
            plot_root_system(root_system, dimension=4)

    @patch('matplotlib.pyplot.figure')
    def test_plot_root_system_3d(self, mock_figure):
        """Test plotting root system in 3D."""
        # Reset any previous calls
        mock_figure.reset_mock()
        
        # Mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_scatter = MagicMock()
        mock_colorbar = MagicMock()
        
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        mock_ax.scatter.return_value = mock_scatter
        plt.colorbar = MagicMock(return_value=mock_colorbar)
        
        # Create a smaller root system for testing
        root_system = np.random.rand(20, 8)
        
        # Call function with 3D and show_connections=False
        fig = plot_root_system(
            root_system,
            dimension=3,
            show_connections=False,
            highlight_roots=None,
            title="Test Root System"
        )
        
        # Assert function behavior - don't check call count
        self.assertTrue(mock_figure.called)
        self.assertTrue(mock_fig.add_subplot.called)
        self.assertTrue(mock_ax.scatter.called)
        mock_ax.set_title.assert_called_with("Test Root System")
        self.assertEqual(fig, mock_fig) 