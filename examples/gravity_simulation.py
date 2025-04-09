import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from typing import Optional, Dict, List, Tuple, Any

from holopy.constants.physical_constants import PhysicalConstants
from holopy.gravity.einstein_field import ModifiedEinsteinField
from holopy.gravity.emergent_metric import InfoSpacetimeMetric
from holopy.e8.heterotic import E8E8Heterotic
from holopy.info.current import InfoCurrentTensor
from holopy.info.tensor import compute_higher_order_functional
from holopy.gravity.spacetime import (
    metric_from_quantum_state,
    compute_riemann_tensor,
    compute_ricci_tensor,
    compute_ricci_scalar,
    compute_einstein_tensor
)

# Configure logging
logger = logging.getLogger('holopy')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if os.environ.get('HOLOPY_WORKER') != '1':
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(handler)
        os.environ['HOLOPY_WORKER'] = '1'

class GravitySimulation:
    """
    Simulation of gravitational field for a particle.
    
    This class implements a physically accurate simulation of the gravitational
    field around a small particle, using the holographic gravity framework.
    """
    
    def __init__(
        self,
        mass: float,              # Mass in grams
        radius: float,            # Radius in centimeters
        grid_size: int = 50,      # Number of points in spatial grid
        sim_radius: float = 10.0  # Simulation radius in centimeters
    ):
        """
        Initialize the gravity simulation.
        
        Args:
            mass: Mass of the particle in grams
            radius: Radius of the particle in centimeters
            grid_size: Number of points in the spatial grid (default 50)
            sim_radius: Radius of the simulation space in centimeters (default 10.0)
        """
        # Convert to SI units
        self.mass_kg = mass / 1000.0  # Convert g to kg
        self.radius_m = radius / 100.0  # Convert cm to m
        self.sim_radius_m = sim_radius / 100.0  # Convert cm to m
        
        # Store original units for reference
        self.mass_g = mass
        self.radius_cm = radius
        self.sim_radius_cm = sim_radius
        
        # Grid parameters
        self.grid_size = grid_size
        
        # Physical constants
        self.constants = PhysicalConstants()
        self.gamma = self.constants.gamma
        
        # Initialize E8×E8 heterotic structure for accurate physical calculations
        self.e8_structure = E8E8Heterotic()
        
        # Initialize spatial grid
        self.initialize_grid()
        
        # Initialize metric and curvature storage
        self.metrics = None
        self.riemann_tensors = None
        self.ricci_tensors = None
        self.ricci_scalars = None
        
        logger.info(f"Initialized gravity simulation for {mass}g particle with radius {radius}cm")
        logger.info(f"Information processing rate γ: {self.gamma:.6e} s^-1")
        
    def initialize_grid(self):
        """Initialize the spatial grid for the simulation."""
        # Create a spherical grid around the origin
        # We'll use spherical coordinates (r, θ, φ) and convert to Cartesian
        
        # Radial coordinates (logarithmically spaced to focus on near-field)
        r_values = np.logspace(
            np.log10(self.radius_m * 1.01),  # Start just outside the particle
            np.log10(self.sim_radius_m),     # End at simulation boundary
            self.grid_size
        )
        
        # Angular coordinates (uniform spacing)
        theta_values = np.linspace(0, np.pi, self.grid_size // 2)
        phi_values = np.linspace(0, 2*np.pi, self.grid_size)
        
        # Create meshgrid
        r_grid, theta_grid, phi_grid = np.meshgrid(r_values, theta_values, phi_values, indexing='ij')
        
        # Convert to Cartesian coordinates
        x_grid = r_grid * np.sin(theta_grid) * np.cos(phi_grid)
        y_grid = r_grid * np.sin(theta_grid) * np.sin(phi_grid)
        z_grid = r_grid * np.cos(theta_grid)
        
        # Store the grid
        self.grid = {
            'r': r_grid,
            'theta': theta_grid,
            'phi': phi_grid,
            'x': x_grid,
            'y': y_grid,
            'z': z_grid,
            'r_values': r_values,
            'theta_values': theta_values,
            'phi_values': phi_values
        }
        
        # Calculate the spatial coordinates at each grid point
        self.spatial_coords = np.stack([
            x_grid.flatten(),
            y_grid.flatten(),
            z_grid.flatten()
        ], axis=1)
        
        logger.info(f"Created spatial grid with {self.spatial_coords.shape[0]} points")
        
    def create_density_function(self) -> callable:
        """
        Create a density function for the particle.
        
        Returns:
            callable: Function that returns matter density at a point
        """
        # Calculate particle volume
        volume = (4/3) * np.pi * self.radius_m**3
        
        # Calculate particle density
        particle_density = self.mass_kg / volume
        
        # Create density function
        def density_function(point: np.ndarray) -> float:
            """Calculate matter density at a point."""
            # Calculate distance from origin
            r = np.linalg.norm(point)
            
            # Return density based on distance
            if r <= self.radius_m:
                return particle_density
            else:
                # Exponentially decreasing density outside the particle
                # to account for quantum effects
                return particle_density * np.exp(-(r - self.radius_m) / (self.radius_m * 0.1))
        
        return density_function
    
    def compute_information_current(self) -> InfoCurrentTensor:
        """
        Compute the information current tensor for the particle.
        
        Returns:
            InfoCurrentTensor: Information current tensor
        """
        # Create density function
        density_func = self.create_density_function()
        
        # Create information current tensor from density function
        info_current = InfoCurrentTensor.from_density(
            density_function=density_func,
            grid_size=self.grid_size,
            domain=[(-self.sim_radius_m, self.sim_radius_m)] * 4,  # 4D spacetime
            coordinates='cartesian',
            dimension=4,
            gamma=self.gamma
        )
        
        return info_current
    
    def compute_metrics(self):
        """Compute the spacetime metric at each grid point."""
        # Initialize storage for metrics and curvature
        n_points = self.spatial_coords.shape[0]
        self.metrics = np.zeros((n_points, 4, 4))
        self.riemann_tensors = np.zeros((n_points, 4, 4, 4, 4))
        self.ricci_tensors = np.zeros((n_points, 4, 4))
        self.ricci_scalars = np.zeros(n_points)
        
        # Compute information current tensor
        info_current = self.compute_information_current()
        
        # Create spacetime metric calculator
        spacetime_metric = InfoSpacetimeMetric(
            e8_structure=self.e8_structure,
            info_current=info_current
        )
        
        # Compute metrics for all grid points
        logger.info("Computing metrics at all grid points...")
        for i, spatial_point in enumerate(self.spatial_coords):
            # Create 4D spacetime point (t=0, x, y, z)
            coords = np.array([0.0, spatial_point[0], spatial_point[1], spatial_point[2]])
            
            # Compute metric
            metric = spacetime_metric.compute_metric(coords)
            self.metrics[i] = metric
            
            # Compute curvature tensors
            riemann = compute_riemann_tensor(metric, coords)
            self.riemann_tensors[i] = riemann
            
            ricci = compute_ricci_tensor(riemann)
            self.ricci_tensors[i] = ricci
            
            ricci_scalar = compute_ricci_scalar(ricci, metric)
            self.ricci_scalars[i] = ricci_scalar
            
            # Log progress periodically
            if (i + 1) % (n_points // 10) == 0:
                logger.info(f"Computed metrics for {i + 1}/{n_points} points ({(i + 1) / n_points * 100:.1f}%)")
        
        logger.info("Finished computing metrics and curvature tensors")
    
    def compute_modified_field_equations(self, point_index: int) -> np.ndarray:
        """
        Compute the modified Einstein field equations at a specific point.
        
        Args:
            point_index: Index of the point in the grid
            
        Returns:
            np.ndarray: Modified Einstein tensor
        """
        # Get the metric at the specified point
        metric = self.metrics[point_index]
        
        # Create the energy-momentum tensor for this point
        # For a simple mass, we use the perfect fluid energy-momentum tensor
        T_munu = np.zeros((4, 4))
        
        # Get spatial coordinates
        spatial_point = self.spatial_coords[point_index]
        r = np.linalg.norm(spatial_point)
        
        # Calculate matter density at this point
        density_func = self.create_density_function()
        rho = density_func(spatial_point)
        
        # For a perfect fluid at rest:
        # T^00 = ρc² (energy density)
        # T^ii = P (pressure, which is zero for dust)
        T_munu[0, 0] = rho * self.constants.c**2
        
        # Create information current tensor
        info_current = self.compute_information_current()
        
        # Create modified Einstein field equation solver
        einstein_field = ModifiedEinsteinField(
            metric=metric,
            energy_momentum=T_munu,
            info_current=info_current
        )
        
        # Solve the field equations
        einstein_tensor = einstein_field.compute_einstein_tensor()
        k_tensor = einstein_field.compute_k_tensor()
        
        # Compute the modified Einstein tensor with information correction
        # G_μν + Λg_μν = (8πG/c⁴)T_μν + γ · 𝒦_μν
        G_constant = self.constants.G
        c = self.constants.c
        gamma = self.constants.gamma
        
        modified_einstein = einstein_tensor - (8 * np.pi * G_constant / c**4) * T_munu - gamma * k_tensor
        
        return modified_einstein
    
    def calculate_gravitational_potential(self) -> np.ndarray:
        """
        Calculate the Newtonian gravitational potential for comparison.
        
        Returns:
            np.ndarray: Gravitational potential at each grid point
        """
        # Initialize potential array
        n_points = self.spatial_coords.shape[0]
        potential = np.zeros(n_points)
        
        # Constants
        G = self.constants.G
        
        # Calculate potential at each point
        for i, point in enumerate(self.spatial_coords):
            r = np.linalg.norm(point)
            
            # Inside the particle
            if r < self.radius_m:
                # Potential inside a uniform sphere
                potential[i] = -G * self.mass_kg * (3 * self.radius_m**2 - r**2) / (2 * self.radius_m**3)
            else:
                # Potential outside (standard Newtonian)
                potential[i] = -G * self.mass_kg / r
        
        return potential
    
    def plot_gravitational_field(self):
        """Plot the gravitational field around the particle."""
        # Make sure metrics have been computed
        if self.metrics is None:
            logger.info("Computing metrics first...")
            self.compute_metrics()
        
        # Calculate Newtonian potential for comparison
        newtonian_potential = self.calculate_gravitational_potential()
        
        # Calculate holographic corrections to the potential
        # Extracted from the g_00 component of the metric
        holographic_potential = np.zeros_like(newtonian_potential)
        for i, metric in enumerate(self.metrics):
            # The g_00 component relates to the gravitational potential
            # In weak field: g_00 ≈ -(1 + 2Φ/c²)
            holographic_potential[i] = -0.5 * (metric[0, 0] + 1) * self.constants.c**2
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create x-axis values (distance from center)
        r_values = np.array([np.linalg.norm(point) for point in self.spatial_coords])
        
        # Sort by radius for clean plotting
        sort_idx = np.argsort(r_values)
        r_sorted = r_values[sort_idx]
        newtonian_sorted = newtonian_potential[sort_idx]
        holographic_sorted = holographic_potential[sort_idx]
        
        # Remove duplicate r values by averaging the potentials
        r_unique, indices = np.unique(r_sorted, return_inverse=True)
        newtonian_averaged = np.zeros_like(r_unique)
        holographic_averaged = np.zeros_like(r_unique)
        
        for i, r in enumerate(r_unique):
            mask = (indices == i)
            newtonian_averaged[i] = np.mean(newtonian_sorted[mask])
            holographic_averaged[i] = np.mean(holographic_sorted[mask])
        
        # Converted to centimeters for plotting
        r_cm = r_unique * 100
        
        # Plot potentials
        plt.subplot(2, 2, 1)
        plt.plot(r_cm, newtonian_averaged, 'b-', label='Newtonian')
        plt.plot(r_cm, holographic_averaged, 'r--', label='Holographic')
        plt.xscale('log')
        plt.xlabel('Distance from center (cm)')
        plt.ylabel('Gravitational Potential (J/kg)')
        plt.title('Gravitational Potential')
        plt.grid(True)
        plt.legend()
        
        # Plot potential difference
        plt.subplot(2, 2, 2)
        diff_percentage = (holographic_averaged - newtonian_averaged) / np.abs(newtonian_averaged) * 100
        plt.plot(r_cm, diff_percentage, 'g-')
        plt.xscale('log')
        plt.xlabel('Distance from center (cm)')
        plt.ylabel('Difference (%)')
        plt.title('Holographic Correction to Potential')
        plt.grid(True)
        
        # Plot space curvature (Ricci scalar)
        plt.subplot(2, 2, 3)
        # Calculate average Ricci scalar at each radius
        ricci_averaged = np.zeros_like(r_unique)
        for i, r in enumerate(r_unique):
            mask = (indices == i)
            ricci_averaged[i] = np.mean(self.ricci_scalars[sort_idx][mask])
        
        plt.plot(r_cm, ricci_averaged, 'k-')
        plt.xscale('log')
        plt.xlabel('Distance from center (cm)')
        plt.ylabel('Ricci Scalar (m⁻²)')
        plt.title('Spacetime Curvature')
        plt.grid(True)
        
        # Plot gravitational acceleration
        plt.subplot(2, 2, 4)
        # Calculate gravitational acceleration
        # g = -∇Φ ≈ -dΦ/dr for spherically symmetric field
        newtonian_g = np.zeros_like(r_unique[:-1])
        holographic_g = np.zeros_like(r_unique[:-1])
        r_mid = 0.5 * (r_unique[1:] + r_unique[:-1])
        
        for i in range(len(r_unique) - 1):
            dr = r_unique[i+1] - r_unique[i]
            newtonian_g[i] = -(newtonian_averaged[i+1] - newtonian_averaged[i]) / dr
            holographic_g[i] = -(holographic_averaged[i+1] - holographic_averaged[i]) / dr
        
        plt.plot(r_mid * 100, newtonian_g, 'b-', label='Newtonian')
        plt.plot(r_mid * 100, holographic_g, 'r--', label='Holographic')
        plt.xscale('log')
        plt.xlabel('Distance from center (cm)')
        plt.ylabel('Gravitational Acceleration (m/s²)')
        plt.title('Gravitational Acceleration')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('gravity_simulation_results.png', dpi=300)
        plt.show()
        
        logger.info("Saved plot to 'gravity_simulation_results.png'")
    
    def print_physical_results(self):
        """Print physical results of the simulation."""
        # Make sure metrics have been computed
        if self.metrics is None:
            logger.info("Computing metrics first...")
            self.compute_metrics()
        
        # Calculate classical Schwarzschild radius for comparison
        r_s = 2 * self.constants.G * self.mass_kg / self.constants.c**2
        
        # Calculate holographic correction factor based on information processing rate
        gamma_factor = self.constants.gamma / self.constants.hubble_parameter
        info_correction = 1 + gamma_factor * np.log(self.constants.l_p / self.radius_m)
        
        # Estimate corrected Schwarzschild radius
        r_s_holographic = r_s * info_correction
        
        # Print results
        print("\n===== Gravity Simulation Results =====")
        print(f"Particle mass: {self.mass_g} g ({self.mass_kg} kg)")
        print(f"Particle radius: {self.radius_cm} cm ({self.radius_m} m)")
        
        print("\nClassical gravitational parameters:")
        print(f"Newtonian gravitational field at surface: {self.constants.G * self.mass_kg / self.radius_m**2:.8e} m/s²")
        print(f"Classical Schwarzschild radius: {r_s:.8e} m")
        
        print("\nHolographic E8×E8 corrections:")
        print(f"Information processing rate (γ): {self.constants.gamma:.8e} s⁻¹")
        print(f"Holographic correction factor: {info_correction:.8f}")
        print(f"Holographic Schwarzschild radius: {r_s_holographic:.8e} m")
        
        # Print spacetime curvature stats
        ricci_min = np.min(self.ricci_scalars)
        ricci_max = np.max(self.ricci_scalars)
        ricci_mean = np.mean(self.ricci_scalars)
        
        print("\nSpacetime curvature statistics:")
        print(f"Ricci scalar (min): {ricci_min:.6e} m⁻²")
        print(f"Ricci scalar (max): {ricci_max:.6e} m⁻²")
        print(f"Ricci scalar (mean): {ricci_mean:.6e} m⁻²")
        
        # Calculate information properties
        volume = (4/3) * np.pi * self.radius_m**3
        surface_area = 4 * np.pi * self.radius_m**2
        
        # Maximum entropy (holographic bound)
        max_entropy = (surface_area * self.constants.c**3) / (4 * self.constants.G * self.constants.hbar)
        
        # Information processing rate
        info_rate = self.constants.gamma * (surface_area / self.constants.l_p**2)
        
        print("\nInformation theoretic properties:")
        print(f"Maximum entropy (holographic bound): {max_entropy:.4e} bits")
        print(f"Information processing rate: {info_rate:.4e} bits/s")
        print("\n========================================")


def main():
    """Main execution function."""
    # Create gravity simulation for a 1cm, 1g particle
    sim = GravitySimulation(
        mass=1.0,       # 1 gram
        radius=0.5,     # 1 cm diameter (0.5 cm radius)
        grid_size=40,   # Grid resolution
        sim_radius=50.0 # Simulation radius in cm
    )
    
    # Compute metrics and curvature
    sim.compute_metrics()
    
    # Print physical results
    sim.print_physical_results()
    
    # Plot gravitational field
    sim.plot_gravitational_field()


if __name__ == "__main__":
    main() 