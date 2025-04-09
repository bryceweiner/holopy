import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from typing import Optional, Dict, List, Tuple, Any
import time
from datetime import datetime

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

# Configure logging with more detailed formatting
logger = logging.getLogger('holopy')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if os.environ.get('HOLOPY_WORKER') != '1':
        # Create console handler with detailed formatting
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter(
                '[%(asctime)s] %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
        logger.addHandler(console_handler)
        
        # Create file handler for persistent logging
        log_dir = 'gravity_simulation/logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, f'gravity_sim_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                '[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
        logger.addHandler(file_handler)
        
        os.environ['HOLOPY_WORKER'] = '1'

# Add debug level logging for development
if os.environ.get('HOLOPY_DEBUG'):
    logger.setLevel(logging.DEBUG)
    logger.debug("Debug logging enabled")

# Timer decorator for performance logging
def log_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.debug(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"Completed {func.__name__} in {end_time - start_time:.3f} seconds")
        return result
    return wrapper

class GravitySimulation:
    """
    Simulates gravity using holographic principles and information theory.
    """
    @log_execution_time
    def __init__(self, mass: float = 1.0, radius: float = 1.0, info_rate: float = 1e6, grid_size: int = 40):
        """
        Initialize gravity simulation with given parameters.
        """
        logger.info(f"Initializing GravitySimulation with mass={mass}, radius={radius}, info_rate={info_rate}, grid_size={grid_size}")
        self.mass = mass
        self.radius = radius
        self.info_rate = info_rate
        self.grid_size = grid_size
        self.constants = PhysicalConstants()
        
        # Convert to SI units for calculations
        self.mass_kg = mass / 1000.0  # Convert g to kg
        self.radius_m = radius / 100.0  # Convert cm to m
        
        # Info processing rate (gamma)
        self.gamma = self.constants.gamma
        
        # Initialize simulation components
        logger.debug("Initializing E8 heterotic string theory components")
        self.e8 = E8E8Heterotic()
        
        logger.debug("Initializing information spacetime metric")
        self.metric = InfoSpacetimeMetric()
        
        # Create default metric and energy-momentum for Einstein field
        default_metric = np.eye(4)
        default_energy_momentum = np.zeros((4, 4))
        
        logger.debug("Initializing modified Einstein field equations")
        self.einstein_field = ModifiedEinsteinField(
            metric=default_metric,
            energy_momentum=default_energy_momentum
        )
        
        logger.info("Gravity simulation initialized successfully")

    @log_execution_time
    def compute_quantum_gravity_effects(self) -> Dict[str, Any]:
        """
        Compute quantum gravity effects using holographic principles.
        """
        logger.info("Computing quantum gravity effects")
        try:
            # Calculate information content
            logger.debug("Computing information content from mass and radius")
            info_content = (self.mass * self.constants.c**2) / (self.constants.hbar * self.info_rate)
            logger.info(f"System information content: {info_content:.2e} bits")
            
            # Calculate quantum state and metric
            logger.debug("Computing spacetime metric from quantum state")
            quantum_state = self.e8.compute_quantum_state(self.mass, self.radius)
            metric_tensor = metric_from_quantum_state(quantum_state)
            logger.debug(f"Metric tensor shape: {metric_tensor.shape}")
            
            # Compute geometric tensors
            logger.debug("Computing geometric tensors")
            riemann = compute_riemann_tensor(metric_tensor)
            ricci_tensor = compute_ricci_tensor(riemann)
            ricci_scalar = compute_ricci_scalar(ricci_tensor)
            einstein_tensor = compute_einstein_tensor(ricci_tensor, ricci_scalar)
            
            # Calculate information current
            logger.debug("Computing information current tensor")
            info_current = InfoCurrentTensor(self.info_rate)
            current_tensor = info_current.compute(metric_tensor)
            
            # Higher order corrections
            logger.debug("Computing higher order quantum corrections")
            quantum_corrections = compute_higher_order_functional(
                metric_tensor, 
                riemann,
                self.constants.planck_length
            )
            
            results = {
                'info_content': info_content,
                'metric_tensor': metric_tensor,
                'riemann_tensor': riemann,
                'ricci_tensor': ricci_tensor,
                'ricci_scalar': ricci_scalar,
                'einstein_tensor': einstein_tensor,
                'current_tensor': current_tensor,
                'quantum_corrections': quantum_corrections
            }
            
            logger.info("Quantum gravity effects computed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error computing quantum gravity effects: {str(e)}", exc_info=True)
            raise

    @log_execution_time
    def run_simulation(self, time_steps: int = 100) -> Dict[str, np.ndarray]:
        """
        Run the gravity simulation for specified time steps.
        """
        logger.info(f"Starting gravity simulation with {time_steps} time steps")
        try:
            results = {
                'time': np.zeros(time_steps),
                'energy': np.zeros(time_steps),
                'entropy': np.zeros(time_steps),
                'curvature': np.zeros(time_steps)
            }
            
            for t in range(time_steps):
                if t % 10 == 0:  # Log progress every 10 steps
                    logger.info(f"Simulation progress: {t}/{time_steps} steps ({t/time_steps*100:.1f}%)")
                
                # Compute quantum effects for current timestep
                quantum_effects = self.compute_quantum_gravity_effects()
                
                # Store results
                results['time'][t] = t * self.constants.planck_time
                results['energy'][t] = np.mean(np.abs(quantum_effects['einstein_tensor']))
                results['entropy'][t] = self.calculate_entropy(quantum_effects)
                results['curvature'][t] = np.mean(np.abs(quantum_effects['ricci_scalar']))
                
                logger.debug(f"Step {t}: Energy={results['energy'][t]:.2e}, "
                           f"Entropy={results['entropy'][t]:.2e}, "
                           f"Curvature={results['curvature'][t]:.2e}")
            
            logger.info("Simulation completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error during simulation: {str(e)}", exc_info=True)
            raise

    @log_execution_time
    def calculate_entropy(self, quantum_effects: Dict[str, Any]) -> float:
        """
        Calculate the holographic entropy from quantum effects.
        """
        logger.debug("Calculating holographic entropy")
        try:
            # Area in Planck units
            area = 4 * np.pi * self.radius**2 / self.constants.planck_length**2
            
            # Quantum corrections
            corrections = np.mean(np.abs(quantum_effects['quantum_corrections']))
            
            # Holographic entropy (Bekenstein-Hawking with quantum corrections)
            entropy = (area / 4) * (1 + corrections)
            
            logger.debug(f"Calculated entropy: {entropy:.2e}")
            return entropy
            
        except Exception as e:
            logger.error(f"Error calculating entropy: {str(e)}", exc_info=True)
            raise

    def initialize_grid(self):
        """Initialize the spatial grid for the simulation."""
        # Create a spherical grid around the origin
        # We'll use spherical coordinates (r, θ, φ) and convert to Cartesian
        
        # Radial coordinates (logarithmically spaced to focus on near-field)
        r_values = np.logspace(
            np.log10(self.radius * 1.01),  # Start just outside the particle
            np.log10(self.radius * 10.0),     # End at simulation boundary
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
        volume = (4/3) * np.pi * self.radius**3
        
        # Calculate particle density
        particle_density = self.mass / volume
        
        # Create density function
        def density_function(point: np.ndarray) -> float:
            """Calculate matter density at a point."""
            # Calculate distance from origin
            r = np.linalg.norm(point)
            
            # Return density based on distance
            if r <= self.radius:
                return particle_density
            else:
                # Exponentially decreasing density outside the particle
                # to account for quantum effects
                return particle_density * np.exp(-(r - self.radius) / (self.radius * 0.1))
        
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
            domain=[(-self.radius, self.radius)] * 4,  # 4D spacetime
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
            e8_structure=self.e8,
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
            if r < self.radius:
                # Potential inside a uniform sphere
                potential[i] = -G * self.mass * (3 * self.radius**2 - r**2) / (2 * self.radius**3)
            else:
                # Potential outside (standard Newtonian)
                potential[i] = -G * self.mass / r
        
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
        plt.savefig('gravity_simulation/figures/gravity_simulation_results.png', dpi=300)
        plt.show()
        
        logger.info("Saved plot to 'gravity_simulation_results.png'")
    
    def print_physical_results(self):
        """Print physical results of the simulation."""
        # Make sure metrics have been computed
        if self.metrics is None:
            logger.info("Computing metrics first...")
            self.compute_metrics()
        
        # Calculate classical Schwarzschild radius for comparison
        r_s = 2 * self.constants.G * self.mass / self.constants.c**2
        
        # Calculate holographic correction factor based on information processing rate
        gamma_factor = self.constants.gamma / self.constants.hubble_parameter
        info_correction = 1 + gamma_factor * np.log(self.constants.l_p / self.radius)
        
        # Estimate corrected Schwarzschild radius
        r_s_holographic = r_s * info_correction
        
        # Print results
        print("\n===== Gravity Simulation Results =====")
        print(f"Particle mass: {self.mass} g ({self.mass_kg} kg)")
        print(f"Particle radius: {self.radius} cm ({self.radius_m} m)")
        
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
        volume = (4/3) * np.pi * self.radius**3
        surface_area = 4 * np.pi * self.radius**2
        
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
    logger.info("Starting gravity simulation main function")
    sim = GravitySimulation(
        mass=1.0,      # 1 gram
        radius=0.5,    # 0.5 cm radius
        info_rate=1e6  # Information processing rate
    )
    
    # Initialize the grid for visualization
    logger.debug("Initializing simulation grid")
    sim.initialize_grid()
    
    # Compute metrics and curvature
    logger.info("Computing metrics and curvature")
    sim.compute_metrics()
    
    # Print physical results
    logger.info("Printing physical results")
    sim.print_physical_results()
    
    # Run the quantum simulation
    logger.info("Running quantum gravity simulation")
    results = sim.run_simulation(time_steps=50)
    
    # Plot gravitational field
    logger.info("Plotting gravitational field")
    sim.plot_gravitational_field()
    
    logger.info("Simulation completed successfully")
    return results


if __name__ == "__main__":
    main() 