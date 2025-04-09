"""
Causal Patch Module

This module implements causal patch definitions and utilities for the dS/QFT
correspondence. A causal patch defines an observation region in de Sitter space,
with various geometries supported (cosmological, static, flat slicing).
"""

import numpy as np
import logging
from typing import Dict, Union, Optional, Callable, Tuple, List, Any, Literal
from enum import Enum

from holopy.constants.physical_constants import PhysicalConstants
from holopy.constants.dsqft_constants import DSQFTConstants

# Setup logging
logger = logging.getLogger(__name__)

class PatchType(Enum):
    """Enumeration of causal patch types."""
    COSMOLOGICAL = 1  # FLRW coordinates with cosmic expansion
    STATIC = 2        # Static de Sitter coordinates with cosmological horizon
    FLAT = 3          # Flat slicing coordinates
    CUSTOM = 4        # Custom user-defined patch


class CausalPatch:
    """
    Causal patch definition for observation regions in de Sitter space.
    
    This class defines observation regions in de Sitter space for the dS/QFT
    correspondence. It supports various patch geometries and coordinate systems,
    enabling the study of holographic physics in different contexts.
    
    Attributes:
        patch_type (PatchType): Type of causal patch
        d (int): Number of spacetime dimensions (typically 4)
        gamma (float): Information processing rate γ
        hubble_parameter (float): Hubble parameter H
        radius (float): Physical radius of the patch in meters
        reference_time (float): Reference time for the patch definition
        horizon_distance (float): Distance to the cosmological horizon
    """
    
    def __init__(self, radius: Optional[float] = None, 
                reference_frame: str = 'static',
                observer_time: float = 0.0,
                d: int = 4, gamma: Optional[float] = None, 
                hubble_parameter: Optional[float] = None):
        """
        Initialize a causal patch.
        
        Args:
            radius (float, optional): Physical radius of the patch in meters
            reference_frame (str): Type of reference frame ('static', 'cosmological', 'flat')
            observer_time (float): Observer's time coordinate
            d (int, optional): Number of spacetime dimensions (default: 4)
            gamma (float, optional): Information processing rate γ (default: from constants)
            hubble_parameter (float, optional): Hubble parameter H (default: from constants)
        """
        self.d = d
        
        # Get constants if not provided
        pc = PhysicalConstants()
        self.gamma = gamma if gamma is not None else pc.gamma
        self.hubble_parameter = hubble_parameter if hubble_parameter is not None else pc.hubble_parameter
        
        # Set patch type based on reference frame
        if reference_frame.lower() == 'static':
            self.patch_type = PatchType.STATIC
        elif reference_frame.lower() == 'cosmological' or reference_frame.lower() == 'flrw':
            self.patch_type = PatchType.COSMOLOGICAL
        elif reference_frame.lower() == 'flat':
            self.patch_type = PatchType.FLAT
        else:
            self.patch_type = PatchType.CUSTOM
            logger.warning(f"Unknown reference frame '{reference_frame}', using custom patch type")
        
        # Set reference time
        self.reference_time = observer_time
        
        # Calculate horizon distance
        self.horizon_distance = 1.0 / self.hubble_parameter
        
        # Set or calculate patch radius
        if radius is not None:
            self.radius = radius
        else:
            # Default to horizon distance
            self.radius = self.horizon_distance
        
        # Calculate conformal time for cosmological coordinates
        if self.patch_type == PatchType.COSMOLOGICAL:
            # η = -1/(a(t)H) = -e^(-Ht)/H
            self.conformal_time = -np.exp(-self.hubble_parameter * self.reference_time) / self.hubble_parameter
        else:
            self.conformal_time = None
        
        logger.debug(f"CausalPatch initialized with type={self.patch_type}, radius={self.radius}")
    
    def is_inside_patch(self, t: float, x: np.ndarray) -> bool:
        """
        Check if a spacetime point is inside the causal patch.
        
        Args:
            t (float): Time coordinate
            x (np.ndarray): Spatial coordinates
            
        Returns:
            bool: True if the point is inside the patch, False otherwise
        """
        # Check dimensions
        if len(x) != self.d - 1:
            raise ValueError(f"Spatial coordinates must have dimension {self.d-1}")
        
        # Calculate distance from the patch center
        r = np.sqrt(np.sum(x**2))
        
        # Check if inside patch based on patch type
        if self.patch_type == PatchType.STATIC:
            # In static coordinates, the patch is a region within the horizon
            # and within the specified radius
            return r <= min(self.radius, self.horizon_distance)
        
        elif self.patch_type == PatchType.COSMOLOGICAL:
            # In cosmological coordinates, we need to account for cosmic expansion
            # a(t) = e^(Ht)
            scale_factor = np.exp(self.hubble_parameter * t)
            physical_distance = scale_factor * r
            
            # The physical patch radius also expands with time
            expanded_radius = self.radius * np.exp(self.hubble_parameter * (t - self.reference_time))
            
            return physical_distance <= expanded_radius
        
        elif self.patch_type == PatchType.FLAT:
            # In flat slicing, the patch is defined by proper distance
            return r <= self.radius
        
        else:  # CUSTOM
            # For custom patches, we use a simple proper distance check
            return r <= self.radius
    
    def is_causally_connected(self, t1: float, x1: np.ndarray, 
                             t2: float, x2: np.ndarray) -> bool:
        """
        Check if two spacetime points are causally connected within the patch.
        
        Args:
            t1 (float): Time coordinate of first point
            x1 (np.ndarray): Spatial coordinates of first point
            t2 (float): Time coordinate of second point
            x2 (np.ndarray): Spatial coordinates of second point
            
        Returns:
            bool: True if the points are causally connected, False otherwise
        """
        # Check if both points are inside the patch
        if not (self.is_inside_patch(t1, x1) and self.is_inside_patch(t2, x2)):
            return False
        
        # Calculate spatial distance
        dx = x2 - x1
        r = np.sqrt(np.sum(dx**2))
        
        # Calculate the light travel time between points
        dt = abs(t2 - t1)
        
        if self.patch_type == PatchType.STATIC:
            # In static coordinates, causal connection if |Δt| ≥ |Δx|/c
            return dt >= r / PhysicalConstants().c
        
        elif self.patch_type == PatchType.COSMOLOGICAL:
            # In FLRW coordinates, we need to integrate the null geodesic
            # This is a simplification for small distances
            a1 = np.exp(self.hubble_parameter * t1)
            a2 = np.exp(self.hubble_parameter * t2)
            average_a = 0.5 * (a1 + a2)
            
            # Proper distance between points at average scale factor
            proper_distance = average_a * r
            
            # Causal connection if |Δt| ≥ proper_distance/c
            return dt >= proper_distance / PhysicalConstants().c
        
        else:  # FLAT or CUSTOM
            # Simple causality check
            return dt >= r / PhysicalConstants().c
    
    def proper_time(self, coordinate_time: float) -> float:
        """
        Convert coordinate time to proper time for an observer at the patch center.
        
        Args:
            coordinate_time (float): Coordinate time
            
        Returns:
            float: Proper time for an observer at the patch center
        """
        if self.patch_type == PatchType.STATIC:
            # In static coordinates, proper time at r=0 equals coordinate time
            return coordinate_time
        
        elif self.patch_type == PatchType.COSMOLOGICAL:
            # In FLRW coordinates, proper time equals coordinate time
            return coordinate_time
        
        else:  # FLAT or CUSTOM
            # For simplicity, use coordinate time
            return coordinate_time
    
    def coordinate_time(self, proper_time: float) -> float:
        """
        Convert proper time to coordinate time for an observer at the patch center.
        
        Args:
            proper_time (float): Proper time for an observer at the patch center
            
        Returns:
            float: Coordinate time
        """
        if self.patch_type == PatchType.STATIC:
            # In static coordinates, coordinate time at r=0 equals proper time
            return proper_time
        
        elif self.patch_type == PatchType.COSMOLOGICAL:
            # In FLRW coordinates, coordinate time equals proper time
            return proper_time
        
        else:  # FLAT or CUSTOM
            # For simplicity, use proper time
            return proper_time
    
    def conformal_to_proper_time(self, eta: float) -> float:
        """
        Convert conformal time to proper time.
        
        Args:
            eta (float): Conformal time
            
        Returns:
            float: Proper time
        """
        if self.patch_type == PatchType.COSMOLOGICAL:
            # In FLRW coordinates, t = -ln(-Hη)/H
            if eta >= 0:
                raise ValueError("Conformal time must be negative in de Sitter space")
            
            return -np.log(-self.hubble_parameter * eta) / self.hubble_parameter
        
        else:
            # For other coordinates, we need a more complex conversion
            # This is a simplified approximation
            if eta >= 0:
                raise ValueError("Conformal time must be negative in de Sitter space")
            
            return -np.log(-self.hubble_parameter * eta) / self.hubble_parameter
    
    def proper_to_conformal_time(self, t: float) -> float:
        """
        Convert proper time to conformal time.
        
        Args:
            t (float): Proper time
            
        Returns:
            float: Conformal time
        """
        if self.patch_type == PatchType.COSMOLOGICAL:
            # In FLRW coordinates, η = -e^(-Ht)/H
            return -np.exp(-self.hubble_parameter * t) / self.hubble_parameter
        
        else:
            # For other coordinates, we need a more complex conversion
            # This is a simplified approximation
            return -np.exp(-self.hubble_parameter * t) / self.hubble_parameter
    
    def static_to_cosmological(self, t_static: float, x_static: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Convert coordinates from static to cosmological frame.
        
        Args:
            t_static (float): Time in static coordinates
            x_static (np.ndarray): Spatial coordinates in static frame
            
        Returns:
            Tuple[float, np.ndarray]: Time and spatial coordinates in cosmological frame
        """
        # This is a non-trivial coordinate transformation
        # For simplicity, we use an approximation valid for small r
        
        r_static = np.sqrt(np.sum(x_static**2))
        
        if r_static >= self.horizon_distance:
            logger.warning(f"Static coordinates outside horizon (r={r_static}), transformation may be inaccurate")
        
        # Approximate transformation for small r
        t_cosmo = t_static + 0.5 * self.hubble_parameter * r_static**2
        
        # Spatial coordinates scale with the scale factor
        scale_factor = np.exp(self.hubble_parameter * t_cosmo)
        x_cosmo = x_static / scale_factor
        
        return t_cosmo, x_cosmo
    
    def cosmological_to_static(self, t_cosmo: float, x_cosmo: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Convert coordinates from cosmological to static frame.
        
        Args:
            t_cosmo (float): Time in cosmological coordinates
            x_cosmo (np.ndarray): Spatial coordinates in cosmological frame
            
        Returns:
            Tuple[float, np.ndarray]: Time and spatial coordinates in static frame
        """
        # This is the inverse of the previous transformation
        # Also an approximation valid for small r
        
        # Scale spatial coordinates
        scale_factor = np.exp(self.hubble_parameter * t_cosmo)
        x_static = x_cosmo * scale_factor
        
        r_static = np.sqrt(np.sum(x_static**2))
        
        # Approximate time transformation
        t_static = t_cosmo - 0.5 * self.hubble_parameter * r_static**2
        
        return t_static, x_static
    
    def boundary_projection(self, resolution: int = 100) -> np.ndarray:
        """
        Project the causal patch to its boundary.
        
        Args:
            resolution (int): Number of points to use for the boundary projection
            
        Returns:
            np.ndarray: Array of boundary points
        """
        # Implementation depends on the patch type
        if self.patch_type == PatchType.STATIC:
            # For static patch, the boundary is a 2-sphere at r = 1/H
            boundary_points = []
            
            # Generate points on a sphere
            # We use the Fibonacci sphere algorithm for uniform distribution
            phi = (1 + np.sqrt(5)) / 2  # Golden ratio
            
            for i in range(resolution):
                y = 1 - 2 * (i / (resolution - 1))  # y goes from 1 to -1
                radius = np.sqrt(1 - y**2)  # radius at y
                
                theta = 2 * np.pi * i / phi  # Golden angle increment
                
                x = radius * np.cos(theta)
                z = radius * np.sin(theta)
                
                # Scale by horizon distance
                point = np.array([x, y, z]) * self.horizon_distance
                
                boundary_points.append(point)
            
            return np.array(boundary_points)
        
        elif self.patch_type == PatchType.COSMOLOGICAL:
            # For FLRW patch, the boundary is at conformal time η = 0
            # It's a 2-sphere of any radius (we use the horizon distance)
            return self.boundary_projection(resolution)  # Same as static
        
        else:  # FLAT or CUSTOM
            # For flat patch, we use a large sphere
            return self.boundary_projection(resolution)  # Same as static
    
    def create_spatial_grid(self, resolution: int = 20) -> np.ndarray:
        """
        Create a spatial grid for calculations within the causal patch.
        
        Args:
            resolution (int): Resolution of the grid (points per dimension)
            
        Returns:
            np.ndarray: Array of spatial grid points
        """
        # Create a 3D grid based on patch type
        grid = []
        
        if self.patch_type == PatchType.STATIC:
            # For static patch, use a spherical grid within the radius
            # Use spherical coordinates for uniform coverage
            for r_idx in range(resolution):
                # Radial coordinate (denser near origin)
                r = self.radius * (r_idx / (resolution-1))**2
                
                # Number of angular points proportional to shell area
                angular_res = max(4, int(np.sqrt(resolution) * r / self.radius))
                
                for theta_idx in range(angular_res):
                    theta = np.pi * theta_idx / (angular_res-1)
                    
                    for phi_idx in range(angular_res):
                        phi = 2 * np.pi * phi_idx / (angular_res-1)
                        
                        # Convert to Cartesian coordinates
                        x = r * np.sin(theta) * np.cos(phi)
                        y = r * np.sin(theta) * np.sin(phi)
                        z = r * np.cos(theta)
                        
                        grid.append(np.array([x, y, z]))
        
        elif self.patch_type == PatchType.COSMOLOGICAL:
            # For FLRW patch, use a cubic grid with physical coordinates
            # Calculate size based on the current size of the patch
            half_size = self.radius
            step = 2 * half_size / (resolution - 1)
            
            for i in range(resolution):
                x = -half_size + i * step
                for j in range(resolution):
                    y = -half_size + j * step
                    for k in range(resolution):
                        z = -half_size + k * step
                        
                        # Only include points within the patch radius
                        if np.sqrt(x**2 + y**2 + z**2) <= self.radius:
                            grid.append(np.array([x, y, z]))
        
        else:  # FLAT or CUSTOM
            # For flat patch, use a simple cubic grid
            half_size = self.radius
            step = 2 * half_size / (resolution - 1)
            
            for i in range(resolution):
                x = -half_size + i * step
                for j in range(resolution):
                    y = -half_size + j * step
                    for k in range(resolution):
                        z = -half_size + k * step
                        
                        # Only include points within the patch radius
                        if np.sqrt(x**2 + y**2 + z**2) <= self.radius:
                            grid.append(np.array([x, y, z]))
        
        return np.array(grid)
    
    def transform_to_patch_coordinates(self, position: np.ndarray) -> np.ndarray:
        """
        Transform a position vector to patch-specific coordinates.
        
        Args:
            position (np.ndarray): Position vector in standard Cartesian coordinates
            
        Returns:
            np.ndarray: Position vector in patch-specific coordinates
        """
        # Check dimensions
        if len(position) != self.d - 1:
            raise ValueError(f"Position must have dimension {self.d-1}")
        
        # Transform based on patch type
        if self.patch_type == PatchType.STATIC:
            # For static patch, no transformation needed
            return position
        
        elif self.patch_type == PatchType.COSMOLOGICAL:
            # For FLRW patch, we convert from physical to comoving coordinates
            # a(t) = e^(Ht)
            scale_factor = np.exp(self.hubble_parameter * self.reference_time)
            return position / scale_factor
        
        else:  # FLAT or CUSTOM
            # For flat patch, no transformation needed
            return position 