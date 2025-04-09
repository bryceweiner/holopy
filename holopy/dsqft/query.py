"""
Query Module

This module implements query utilities for accessing bulk and boundary information
in the dS/QFT correspondence framework. These utilities provide a unified interface
for extracting physical observables from simulation results.
"""

import numpy as np
import logging
from typing import Dict, Union, Optional, Callable, Tuple, List, Any, Literal
from enum import Enum
from scipy.interpolate import LinearNDInterpolator

from holopy.constants.physical_constants import PhysicalConstants
from holopy.constants.dsqft_constants import DSQFTConstants
from holopy.dsqft.causal_patch import CausalPatch, PatchType

# Setup logging
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Enumeration of query types."""
    FIELD_VALUE = 1    # Field value at a point
    CORRELATION = 2    # Correlation function between points
    ENTROPY = 3        # Entropy density at a point
    ENERGY = 4         # Energy density at a point
    GEOMETRY = 5       # Geometric property (distance, volume, etc.)
    OBSERVABLE = 6     # Physical observable (temperature, curvature, etc.)

class QueryResult:
    """
    Container for query results.
    
    This class stores the result of a query operation, including the value,
    uncertainty, and metadata about the query.
    
    Attributes:
        value (Any): Result value
        uncertainty (float): Uncertainty in the result (if applicable)
        metadata (Dict): Additional metadata about the query
    """
    
    def __init__(self, value: Any, uncertainty: Optional[float] = None,
                metadata: Optional[Dict] = None):
        """
        Initialize a query result.
        
        Args:
            value (Any): Result value
            uncertainty (float, optional): Uncertainty in the result
            metadata (Dict, optional): Additional metadata
        """
        self.value = value
        self.uncertainty = uncertainty
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        """String representation of the query result."""
        if self.uncertainty is not None:
            return f"{self.value} ± {self.uncertainty}"
        else:
            return f"{self.value}"
    
    def __repr__(self) -> str:
        """Detailed representation of the query result."""
        return f"QueryResult(value={self.value}, uncertainty={self.uncertainty}, metadata={self.metadata})"

class DSQFTQuery:
    """
    Query interface for the dS/QFT correspondence.
    
    This class provides a unified interface for querying bulk and boundary
    information in the dS/QFT correspondence framework.
    
    Attributes:
        simulation (object): The simulation to query
        causal_patch (CausalPatch): The causal patch defining the observation region
        d (int): Number of spacetime dimensions (typically 4)
        gamma (float): Information processing rate γ
        hubble_parameter (float): Hubble parameter H
    """
    
    def __init__(self, simulation: Any, causal_patch: Optional[CausalPatch] = None,
                d: int = 4, gamma: Optional[float] = None, 
                hubble_parameter: Optional[float] = None):
        """
        Initialize the query interface.
        
        Args:
            simulation (object): The simulation to query
            causal_patch (CausalPatch, optional): The causal patch
            d (int, optional): Number of spacetime dimensions (default: 4)
            gamma (float, optional): Information processing rate γ (default: from constants)
            hubble_parameter (float, optional): Hubble parameter H (default: from constants)
        """
        self.simulation = simulation
        self.d = d
        
        # Get constants if not provided
        pc = PhysicalConstants()
        self.gamma = gamma if gamma is not None else pc.gamma
        self.hubble_parameter = hubble_parameter if hubble_parameter is not None else pc.hubble_parameter
        
        # Use provided causal patch or get from simulation
        if causal_patch is not None:
            self.causal_patch = causal_patch
        elif hasattr(simulation, 'causal_patch'):
            self.causal_patch = simulation.causal_patch
        else:
            self.causal_patch = CausalPatch(
                d=self.d, gamma=self.gamma, hubble_parameter=self.hubble_parameter
            )
        
        # Initialize interpolators
        self.field_interpolators = {}
        self.energy_interpolator = None
        
        # Setup interpolators for efficient querying
        self._setup_interpolators()
        
        logger.info("DSQFTQuery initialized")
    
    def _setup_interpolators(self) -> None:
        """Setup interpolators for efficient querying."""
        # Check if simulation has results
        if not hasattr(self.simulation, 'results') or not self.simulation.results:
            logger.warning("No simulation results available for interpolation")
            return
        
        # Get results
        results = self.simulation.results
        
        # Create interpolators for each field
        self.field_interpolators = {}
        
        if 'field_evolution' in results and 'time_points' in results and 'spatial_grid' in results:
            time_points = results['time_points']
            spatial_grid = results['spatial_grid']
            
            # Get the latest time index
            latest_time_idx = len(time_points) - 1
            
            for field_name, field_values in results['field_evolution'].items():
                if len(field_values) > 0 and latest_time_idx < len(field_values):
                    # Get the latest field values
                    latest_values = field_values[latest_time_idx]
                    
                    # Create interpolator
                    points = spatial_grid
                    values = latest_values
                    
                    try:
                        # Create interpolator for spatial values at latest time
                        self.field_interpolators[field_name] = LinearNDInterpolator(
                            points, values, fill_value=0.0
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create interpolator for field '{field_name}': {e}")
        
        # Create interpolator for energy density
        self.energy_interpolator = None
        
        if 'energy_density_evolution' in results and 'spatial_grid' in results:
            energy_density = results['energy_density_evolution']
            
            if len(energy_density) > 0:
                # Get the latest energy density
                latest_energy = energy_density[-1]
                
                # Create interpolator
                points = spatial_grid
                values = latest_energy
                
                try:
                    # Create interpolator for energy density at latest time
                    self.energy_interpolator = LinearNDInterpolator(
                        points, values, fill_value=0.0
                    )
                except Exception as e:
                    logger.warning(f"Failed to create interpolator for energy density: {e}")
    
    def query_field_value(self, field_name: str, t: float, x: np.ndarray) -> QueryResult:
        """
        Query the value of a field at a specific spacetime point.
        
        Args:
            field_name (str): Name of the field
            t (float): Time coordinate
            x (np.ndarray): Spatial coordinates
            
        Returns:
            QueryResult: Field value and metadata
        """
        try:
            # Get physical constants
            pc = PhysicalConstants()
            
            # Check if we have simulation results
            if hasattr(self.simulation, 'results'):
                # Try to interpolate from stored results
                if 'field_evolution' in self.simulation.results:
                    field_values = self.simulation.results['field_evolution'].get(field_name)
                    time_points = self.simulation.results.get('time_points')
                    spatial_grid = self.simulation.results.get('spatial_grid')
                    
                    if (field_values is not None and time_points is not None and 
                        spatial_grid is not None and len(field_values) > 0):
                        # Find nearest time point
                        t_idx = np.argmin(np.abs(time_points - t))
                        
                        # Get field values at this time
                        values = field_values[t_idx]
                        
                        # Find nearest spatial point
                        distances = np.array([np.linalg.norm(x - grid_point) 
                                           for grid_point in spatial_grid])
                        x_idx = np.argmin(distances)
                        
                        # Get field value
                        value = values[x_idx]
                        
                        # Apply information processing factor
                        r = np.linalg.norm(x)
                        info_factor = np.exp(-pc.gamma * r / pc.c)
                        value *= info_factor
                        
                        return QueryResult(value, metadata={
                            'query_type': QueryType.FIELD_VALUE,
                            'field_name': field_name,
                            'time': t,
                            'position': x
                        })
            
            # If no results available, use boundary values
            if field_name in self.simulation.boundary_values:
                # Get boundary grid
                boundary_grid = self.simulation.causal_patch.boundary_projection()
                
                # Find nearest boundary point
                distances = np.array([np.linalg.norm(x - grid_point) 
                                   for grid_point in boundary_grid])
                x_idx = np.argmin(distances)
                
                # Get field value from boundary
                value = self.simulation.boundary_values[field_name][x_idx]
                
                # Apply information processing factor
                r = np.linalg.norm(x)
                info_factor = np.exp(-pc.gamma * r / pc.c)
                value *= info_factor
                
                return QueryResult(value, metadata={
                    'query_type': QueryType.FIELD_VALUE,
                    'field_name': field_name,
                    'time': t,
                    'position': x
                })
            
            # If no values available, return zero
            return QueryResult(0.0, metadata={
                'query_type': QueryType.FIELD_VALUE,
                'field_name': field_name,
                'time': t,
                'position': x
            })
            
        except Exception as e:
            logger.error(f"Error querying field value: {e}")
            return QueryResult(0.0, metadata={
                'query_type': QueryType.FIELD_VALUE,
                'field_name': field_name,
                'time': t,
                'position': x,
                'error': str(e)
            })
    
    def query_correlation(self, field_name: str, t1: float, x1: np.ndarray,
                         t2: float, x2: np.ndarray) -> QueryResult:
        """
        Query the correlation function between two spacetime points.
        
        Args:
            field_name (str): Name of the field
            t1 (float): Time coordinate of first point
            x1 (np.ndarray): Spatial coordinates of first point
            t2 (float): Time coordinate of second point
            x2 (np.ndarray): Spatial coordinates of second point
            
        Returns:
            QueryResult: Correlation value and metadata
        """
        # Check if field exists
        if not hasattr(self.simulation, 'field_config') or field_name not in self.simulation.field_config:
            raise KeyError(f"Field '{field_name}' not found in simulation")
        
        try:
            # Compute correlation function
            correlation = self.simulation.compute_correlation_function(
                field_name, t1, x1, t2, x2
            )
            
            return QueryResult(correlation, metadata={
                'query_type': QueryType.CORRELATION,
                'field_name': field_name,
                'coordinates': ((t1, x1), (t2, x2)),
                'method': 'direct_computation'
            })
        except Exception as e:
            logger.error(f"Failed to query correlation: {e}")
            return QueryResult(0.0, uncertainty=float('inf'), metadata={
                'query_type': QueryType.CORRELATION,
                'field_name': field_name,
                'coordinates': ((t1, x1), (t2, x2)),
                'error': str(e)
            })
    
    def query_energy_density(self, t: float, x: np.ndarray) -> QueryResult:
        """
        Query the energy density at a specific spacetime point.
        
        Args:
            t (float): Time coordinate
            x (np.ndarray): Spatial coordinates
            
        Returns:
            QueryResult: Energy density and metadata
        """
        # Check if we have an interpolator for energy density
        if self.energy_interpolator is not None:
            # Use interpolator for fast query
            interpolator = self.energy_interpolator
            
            # Convert coordinates to the same format as the interpolator
            x_flat = np.array(x).flatten()
            
            # Query interpolator
            value = float(interpolator(x_flat))
            
            return QueryResult(value, metadata={
                'query_type': QueryType.ENERGY,
                'coordinates': (t, x),
                'method': 'interpolation'
            })
        
        # Fallback to computing from fields
        try:
            total_energy = 0.0
            
            # Sum energy density contributions from all fields
            for field_name in self.simulation.field_config.keys():
                # Get field value
                field_value = self.simulation.compute_bulk_field(field_name, t, x)
                
                # Simplified energy density calculation
                energy_density = np.abs(field_value)**2
                
                # Add to total
                total_energy += energy_density
            
            return QueryResult(total_energy, metadata={
                'query_type': QueryType.ENERGY,
                'coordinates': (t, x),
                'method': 'direct_computation'
            })
        except Exception as e:
            logger.error(f"Failed to query energy density: {e}")
            return QueryResult(0.0, uncertainty=float('inf'), metadata={
                'query_type': QueryType.ENERGY,
                'coordinates': (t, x),
                'error': str(e)
            })
    
    def query_entropy(self, t: float, x: np.ndarray, radius: float = 1.0) -> QueryResult:
        """
        Query the entropy within a region around a specific spacetime point.
        
        Args:
            t (float): Time coordinate
            x (np.ndarray): Spatial coordinates
            radius (float, optional): Radius of the region (default: 1.0)
            
        Returns:
            QueryResult: Entropy and metadata
        """
        # Check if simulation has results
        if not hasattr(self.simulation, 'results') or not self.simulation.results:
            logger.warning("No simulation results available for entropy query")
            return QueryResult(0.0, uncertainty=float('inf'), metadata={
                'query_type': QueryType.ENTROPY,
                'coordinates': (t, x),
                'radius': radius,
                'error': 'No simulation results available'
            })
        
        try:
            # Get the latest entropy
            if 'entropy_evolution' in self.simulation.results:
                entropy = self.simulation.results['entropy_evolution'][-1]
            else:
                # Compute entropy from energy density
                energy_density = self.query_energy_density(t, x).value
                
                # Simplified entropy estimate
                # S ~ E^(3/4) * V^(1/4) (from holographic entropy scaling)
                volume = (4/3) * np.pi * radius**3
                entropy = (energy_density**(3/4)) * (volume**(1/4))
            
            # Scale by the ratio of the query region to the total patch
            total_volume = (4/3) * np.pi * self.causal_patch.radius**3
            volume_ratio = ((4/3) * np.pi * radius**3) / total_volume
            
            # Scale entropy (assuming uniform distribution as a baseline)
            scaled_entropy = entropy * volume_ratio
            
            # Add uncertainty
            uncertainty = 0.1 * scaled_entropy  # 10% uncertainty (arbitrary)
            
            return QueryResult(scaled_entropy, uncertainty=uncertainty, metadata={
                'query_type': QueryType.ENTROPY,
                'coordinates': (t, x),
                'radius': radius,
                'volume_ratio': volume_ratio
            })
        except Exception as e:
            logger.error(f"Failed to query entropy: {e}")
            return QueryResult(0.0, uncertainty=float('inf'), metadata={
                'query_type': QueryType.ENTROPY,
                'coordinates': (t, x),
                'radius': radius,
                'error': str(e)
            })
    
    def query_geometry(self, query_name: str, *args, **kwargs) -> QueryResult:
        """
        Query geometric properties.
        
        Args:
            query_name (str): Name of the geometric property to query
            *args, **kwargs: Additional arguments for the query
            
        Returns:
            QueryResult: Geometric property and metadata
        """
        # Check query name
        if query_name.lower() == 'proper_distance':
            # Query proper distance between two points
            t1 = kwargs.get('t1', args[0] if len(args) > 0 else 0.0)
            x1 = kwargs.get('x1', args[1] if len(args) > 1 else np.zeros(self.d-1))
            t2 = kwargs.get('t2', args[2] if len(args) > 2 else 0.0)
            x2 = kwargs.get('x2', args[3] if len(args) > 3 else np.zeros(self.d-1))
            
            # Convert times to conformal times if using FLRW coordinates
            if self.causal_patch.patch_type == PatchType.COSMOLOGICAL:
                eta1 = self.causal_patch.proper_to_conformal_time(t1)
                eta2 = self.causal_patch.proper_to_conformal_time(t2)
                
                # In FLRW coordinates, the proper distance depends on the scale factor
                a1 = np.exp(self.hubble_parameter * t1)
                a2 = np.exp(self.hubble_parameter * t2)
                
                # If the points are at different times, we need the comoving distance
                if abs(t1 - t2) > 1e-6:
                    # For simplicity, average the scale factors
                    a_avg = (a1 + a2) / 2.0
                    
                    # Spatial distance
                    dx = np.sqrt(np.sum((x1 - x2)**2))
                    
                    # Proper distance (simplified)
                    distance = a_avg * dx
                else:
                    # Same time, just scale the spatial distance
                    dx = np.sqrt(np.sum((x1 - x2)**2))
                    distance = a1 * dx
            else:
                # For static coordinates, use the standard distance formula
                dx = np.sqrt(np.sum((x1 - x2)**2))
                dt = abs(t1 - t2)
                
                # Proper distance (simplified)
                distance = np.sqrt(dt**2 + dx**2)
            
            return QueryResult(distance, metadata={
                'query_type': QueryType.GEOMETRY,
                'property': 'proper_distance',
                'coordinates': ((t1, x1), (t2, x2))
            })
        
        elif query_name.lower() == 'volume':
            # Query volume of a region
            t = kwargs.get('t', args[0] if len(args) > 0 else 0.0)
            x = kwargs.get('x', args[1] if len(args) > 1 else np.zeros(self.d-1))
            radius = kwargs.get('radius', args[2] if len(args) > 2 else 1.0)
            
            # In FLRW coordinates, the volume depends on the scale factor
            if self.causal_patch.patch_type == PatchType.COSMOLOGICAL:
                a = np.exp(self.hubble_parameter * t)
                volume = (4/3) * np.pi * (a * radius)**3
            else:
                # Standard volume formula
                volume = (4/3) * np.pi * radius**3
            
            return QueryResult(volume, metadata={
                'query_type': QueryType.GEOMETRY,
                'property': 'volume',
                'coordinates': (t, x),
                'radius': radius
            })
        
        elif query_name.lower() == 'horizon_radius':
            # Query horizon radius at a time
            t = kwargs.get('t', args[0] if len(args) > 0 else 0.0)
            
            # Horizon radius is 1/H in proper coordinates
            radius = 1.0 / self.hubble_parameter
            
            return QueryResult(radius, metadata={
                'query_type': QueryType.GEOMETRY,
                'property': 'horizon_radius',
                'time': t
            })
        
        else:
            raise ValueError(f"Unknown geometric property: {query_name}")
    
    def query_observable(self, observable_name: str, *args, **kwargs) -> QueryResult:
        """
        Query an observable quantity from the simulation.
        
        Args:
            observable_name (str): Name of the observable to query
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            QueryResult: Observable value and metadata
        """
        # Get physical constants
        pc = PhysicalConstants()
        
        if observable_name.lower() == 'energy_spectrum':
            try:
                # Get fields
                fields = kwargs.get('fields')
                if fields is None:
                    raise ValueError("Fields required for energy spectrum observable")
                
                # Calculate energy spectrum from field values
                spectrum = []
                for field_name in fields:
                    if field_name in self.simulation.field_config:
                        # Get field properties
                        field_info = self.simulation.field_config[field_name]
                        mass = field_info.get('mass', 0.0)
                        
                        # Calculate energy levels based on field properties
                        # For now, just return the rest energy
                        if mass > 0:
                            energy = mass * pc.c**2
                            spectrum.append(energy)
                
                return QueryResult(np.array(spectrum), metadata={
                    'query_type': QueryType.OBSERVABLE,
                    'observable': 'energy_spectrum',
                    'fields': fields
                })
            except Exception as e:
                logger.error(f"Failed to compute energy spectrum: {e}")
                return QueryResult(np.array([]), uncertainty=float('inf'), metadata={
                    'query_type': QueryType.OBSERVABLE,
                    'observable': 'energy_spectrum',
                    'error': str(e)
                })
                
        elif observable_name.lower() == 'binding_energy':
            # Get fields
            fields = kwargs.get('fields')
            if fields is None:
                raise ValueError("Fields required for binding energy observable")
            
            try:
                # Calculate binding energy from field values
                # This requires proper implementation based on field interactions
                binding_energy = 0.0
                
                # Get field properties
                field_properties = {}
                for field_name in fields:
                    if field_name in self.simulation.field_config:
                        field_info = self.simulation.field_config[field_name]
                        mass = field_info.get('mass', 0.0)
                        charge = field_info.get('charge', 0.0)
                        field_properties[field_name] = {
                            'mass': mass,
                            'charge': charge
                        }
                
                # Calculate Coulomb energy
                if len(field_properties) == 2:
                    # Get charges
                    q1 = field_properties[fields[0]]['charge']
                    q2 = field_properties[fields[1]]['charge']
                    
                    # Get positions from boundary values
                    pos1 = self.simulation.boundary_values[fields[0]]
                    pos2 = self.simulation.boundary_values[fields[1]]
                    
                    # Ensure positions are 3D vectors
                    if not isinstance(pos1, np.ndarray) or pos1.shape != (3,):
                        raise ValueError(f"Invalid position shape for {fields[0]}: {pos1.shape}")
                    if not isinstance(pos2, np.ndarray) or pos2.shape != (3,):
                        raise ValueError(f"Invalid position shape for {fields[1]}: {pos2.shape}")
                    
                    # Calculate separation
                    r = np.linalg.norm(pos1 - pos2)
                    if r > 0:
                        # Coulomb energy in eV (negative for attractive force)
                        # The factor of 1/2 is needed because binding energy is half the Coulomb energy
                        # For attractive forces (opposite charges), q1*q2 is negative, so we don't need an explicit minus sign
                        binding_energy = (q1 * q2) / (8 * np.pi * pc.epsilon_0 * r * pc.elementary_charge)
                    else:
                        # Use Bohr radius as default separation
                        r = 4 * np.pi * pc.epsilon_0 * pc.hbar**2 / (pc.elementary_charge**2 * pc.electron_mass)
                        binding_energy = (q1 * q2) / (8 * np.pi * pc.epsilon_0 * r * pc.elementary_charge)
                
                return QueryResult(binding_energy, metadata={
                    'query_type': QueryType.OBSERVABLE,
                    'observable': 'binding_energy',
                    'fields': fields
                })
            except Exception as e:
                logger.error(f"Failed to compute binding energy: {e}")
                return QueryResult(0.0, uncertainty=float('inf'), metadata={
                    'query_type': QueryType.OBSERVABLE,
                    'observable': 'binding_energy',
                    'error': str(e)
                })
                
        elif observable_name.lower() == 'coherence_scale':
            try:
                # Calculate coherence scale from field properties
                coherence_scale = float('inf')
                
                # Find smallest length scale from field masses
                for field_name, field_info in self.simulation.field_config.items():
                    mass = field_info.get('mass', 0.0)
                    if mass > 0:
                        # Use Compton wavelength as coherence scale
                        scale = pc.hbar / (mass * pc.c)
                        coherence_scale = min(coherence_scale, scale)
                
                # If no massive fields, use simulation radius
                if coherence_scale == float('inf'):
                    coherence_scale = self.simulation.radius
                
                return QueryResult(coherence_scale, metadata={
                    'query_type': QueryType.OBSERVABLE,
                    'observable': 'coherence_scale'
                })
            except Exception as e:
                logger.error(f"Failed to compute coherence scale: {e}")
                return QueryResult(0.0, uncertainty=float('inf'), metadata={
                    'query_type': QueryType.OBSERVABLE,
                    'observable': 'coherence_scale',
                    'error': str(e)
                })
                
        elif observable_name.lower() == 'decoherence_time':
            try:
                # Calculate decoherence time from field properties
                decoherence_time = float('inf')
                
                # Find smallest time scale from field masses
                for field_name, field_info in self.simulation.field_config.items():
                    mass = field_info.get('mass', 0.0)
                    if mass > 0:
                        # Use inverse of rest energy as time scale
                        scale = pc.hbar / (mass * pc.c**2)
                        decoherence_time = min(decoherence_time, scale)
                
                # If no massive fields, use light crossing time
                if decoherence_time == float('inf'):
                    decoherence_time = self.simulation.radius / pc.c
                
                return QueryResult(decoherence_time, metadata={
                    'query_type': QueryType.OBSERVABLE,
                    'observable': 'decoherence_time'
                })
            except Exception as e:
                logger.error(f"Failed to compute decoherence time: {e}")
                return QueryResult(0.0, uncertainty=float('inf'), metadata={
                    'query_type': QueryType.OBSERVABLE,
                    'observable': 'decoherence_time',
                    'error': str(e)
                })
                
        elif observable_name.lower() == 'information_flow':
            try:
                # Calculate information flow rate
                # This is related to the entropy production rate
                if hasattr(self.simulation, 'results'):
                    if 'entropy_evolution' in self.simulation.results:
                        entropy = self.simulation.results['entropy_evolution']
                        if len(entropy) > 1:
                            # Calculate rate of entropy change
                            dt = (self.simulation.results['time_points'][-1] - 
                                 self.simulation.results['time_points'][0])
                            info_flow = (entropy[-1] - entropy[0]) / dt
                        else:
                            info_flow = 0.0
                    else:
                        # Use gamma parameter as default flow rate
                        info_flow = pc.gamma
                else:
                    info_flow = pc.gamma
                
                return QueryResult(info_flow, metadata={
                    'query_type': QueryType.OBSERVABLE,
                    'observable': 'information_flow'
                })
            except Exception as e:
                logger.error(f"Failed to compute information flow: {e}")
                return QueryResult(0.0, uncertainty=float('inf'), metadata={
                    'query_type': QueryType.OBSERVABLE,
                    'observable': 'information_flow',
                    'error': str(e)
                })
        else:
            raise ValueError(f"Unknown observable: {observable_name}")
    
    def bulk_to_boundary(self, t: float, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Convert bulk coordinates to boundary coordinates.
        
        Args:
            t (float): Bulk time coordinate
            x (np.ndarray): Bulk spatial coordinates
            
        Returns:
            Tuple[float, np.ndarray]: Boundary coordinates (time, space)
        """
        # The mapping depends on the patch type
        if self.causal_patch.patch_type == PatchType.COSMOLOGICAL:
            # In FLRW coordinates, the boundary is at conformal time η = 0
            # and spatial coordinates are rescaled
            eta = self.causal_patch.proper_to_conformal_time(t)
            
            # Compute scale factor
            a = np.exp(self.hubble_parameter * t)
            
            # Rescale spatial coordinates
            # In the FLRW case, boundary spatial coordinates are the same as
            # comoving bulk spatial coordinates
            boundary_x = x
            
            # Boundary time is related to conformal time
            # For simplicity, we use the same time coordinate
            boundary_t = t
            
            return (boundary_t, boundary_x)
        
        else:
            # For static coordinates, we use a simpler mapping
            # Boundary time is the same as bulk time
            boundary_t = t
            
            # Boundary spatial coordinates are projected to the boundary
            # For simplicity, we use the same spatial coordinates
            boundary_x = x
            
            return (boundary_t, boundary_x)
    
    def boundary_to_bulk(self, t: float, x: np.ndarray, z: float = 0.0) -> Tuple[float, np.ndarray]:
        """
        Convert boundary coordinates to bulk coordinates.
        
        Args:
            t (float): Boundary time coordinate
            x (np.ndarray): Boundary spatial coordinates
            z (float, optional): Bulk radial coordinate (default: 0.0)
            
        Returns:
            Tuple[float, np.ndarray]: Bulk coordinates (time, space)
        """
        # The mapping depends on the patch type
        if self.causal_patch.patch_type == PatchType.COSMOLOGICAL:
            # In FLRW coordinates, the bulk time is related to the scale factor
            # For simplicity, we use the same time coordinate
            bulk_t = t
            
            # Compute scale factor
            a = np.exp(self.hubble_parameter * t)
            
            # Bulk spatial coordinates are the same as boundary coordinates
            # but scaled by the scale factor
            bulk_x = x / a
            
            return (bulk_t, bulk_x)
        
        else:
            # For static coordinates, we use a simpler mapping
            # Bulk time is the same as boundary time
            bulk_t = t
            
            # Bulk spatial coordinates include the radial coordinate z
            # For simplicity, we just add z to the first coordinate
            bulk_x = x.copy()
            if len(bulk_x) > 0:
                bulk_x[0] += z
            
            return (bulk_t, bulk_x)
    
    def query(self, query_type: Union[str, QueryType], *args, **kwargs) -> QueryResult:
        """
        Unified query interface.
        
        Args:
            query_type (str or QueryType): Type of query
            *args, **kwargs: Additional arguments for the query
            
        Returns:
            QueryResult: Query result
        """
        # Convert string to enum if needed
        if isinstance(query_type, str):
            try:
                query_type = getattr(QueryType, query_type.upper())
            except AttributeError:
                raise ValueError(f"Unknown query type: {query_type}")
        
        # Dispatch query based on type
        if query_type == QueryType.FIELD_VALUE:
            field_name = kwargs.get('field_name', args[0] if len(args) > 0 else None)
            t = kwargs.get('t', args[1] if len(args) > 1 else 0.0)
            x = kwargs.get('x', args[2] if len(args) > 2 else np.zeros(self.d-1))
            
            if field_name is None:
                raise ValueError("Field name must be provided for field value query")
            
            return self.query_field_value(field_name, t, x)
        
        elif query_type == QueryType.CORRELATION:
            field_name = kwargs.get('field_name', args[0] if len(args) > 0 else None)
            t1 = kwargs.get('t1', args[1] if len(args) > 1 else 0.0)
            x1 = kwargs.get('x1', args[2] if len(args) > 2 else np.zeros(self.d-1))
            t2 = kwargs.get('t2', args[3] if len(args) > 3 else 0.0)
            x2 = kwargs.get('x2', args[4] if len(args) > 4 else np.zeros(self.d-1))
            
            if field_name is None:
                raise ValueError("Field name must be provided for correlation query")
            
            return self.query_correlation(field_name, t1, x1, t2, x2)
        
        elif query_type == QueryType.ENERGY:
            t = kwargs.get('t', args[0] if len(args) > 0 else 0.0)
            x = kwargs.get('x', args[1] if len(args) > 1 else np.zeros(self.d-1))
            
            return self.query_energy_density(t, x)
        
        elif query_type == QueryType.ENTROPY:
            t = kwargs.get('t', args[0] if len(args) > 0 else 0.0)
            x = kwargs.get('x', args[1] if len(args) > 1 else np.zeros(self.d-1))
            radius = kwargs.get('radius', args[2] if len(args) > 2 else 1.0)
            
            return self.query_entropy(t, x, radius)
        
        elif query_type == QueryType.GEOMETRY:
            query_name = kwargs.get('query_name', args[0] if len(args) > 0 else None)
            
            if query_name is None:
                raise ValueError("Query name must be provided for geometry query")
            
            return self.query_geometry(query_name, *args[1:], **kwargs)
        
        elif query_type == QueryType.OBSERVABLE:
            observable_name = kwargs.get('observable_name', args[0] if len(args) > 0 else None)
            
            if observable_name is None:
                raise ValueError("Observable name must be provided for observable query")
            
            return self.query_observable(observable_name, *args[1:], **kwargs)
        
        else:
            raise ValueError(f"Unknown query type: {query_type}") 