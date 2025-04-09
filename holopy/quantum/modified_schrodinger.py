"""
Modified Schrödinger Equation Module for HoloPy.

This module implements the modified Schrödinger equation that incorporates
holographic decoherence through spatial complexity:

    iℏ ∂|ψ⟩/∂t = Ĥ|ψ⟩ - iγℏ 𝒟[|ψ⟩]

where 𝒟[|ψ⟩] = |∇ψ|² is the decoherence functional based on spatial complexity.
"""

import numpy as np
from typing import Callable, List, Tuple, Dict, Optional, Union, Any
from dataclasses import dataclass
import scipy.sparse as sp
import scipy.constants as scipy_constants
from scipy.integrate import solve_ivp
from holopy.constants.physical_constants import PHYSICAL_CONSTANTS
from holopy.utils.logging import get_logger, log_execution_time, ProgressTracker
import scipy.integrate

# Configure logging
logger = get_logger('quantum.schrodinger')

class WaveFunction:
    """
    Class representing a quantum wavefunction in the holographic framework.
    
    Attributes:
        psi: The wavefunction values on the grid or basis
        grid: The spatial grid points (if using a position basis)
        basis: The basis in which the wavefunction is represented
        dimension: The spatial dimension of the system
    """
    
    def __init__(self, 
                initial_function: Optional[Callable] = None,
                grid: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                values: Optional[np.ndarray] = None,
                domain: Optional[List[Tuple[float, float]]] = None,
                grid_size: int = 100,
                basis: str = 'position',
                do_normalize: bool = True):
        """
        Initialize a wavefunction.
        
        Args:
            initial_function: Function that returns the wavefunction value at a point
            grid: Spatial grid points if provided directly
            values: Wavefunction values if provided directly
            domain: Domain boundaries for each dimension (if generating a grid)
            grid_size: Number of grid points in each dimension
            basis: The basis in which the wavefunction is represented ('position' or 'momentum')
            do_normalize: Whether to normalize the wavefunction after initialization
        """
        self.basis = basis
        
        # Set up the grid
        if grid is not None:
            self.grid = grid
            if isinstance(grid, list):
                # Multi-dimensional grid
                if len(grid) > 0 and hasattr(grid[0], 'shape'):
                    self.dimension = len(grid)
                else:
                    # Empty grid or no shape attribute
                    self.dimension = 1
            else:
                # 1D grid
                self.dimension = 1
        elif domain is not None:
            self.dimension = len(domain)
            self.grid = self._generate_grid(domain, grid_size)
        else:
            raise ValueError("Either grid or domain must be provided")
        
        # Set up the wavefunction values
        if values is not None:
            self.psi = values
        elif initial_function is not None:
            self.psi = self._evaluate_function(initial_function)
        else:
            raise ValueError("Either values or initial_function must be provided")
        
        # Normalize the wavefunction if requested
        if do_normalize:
            self.normalize()
    
    def copy(self):
        """Create a copy of this wavefunction."""
        return WaveFunction(
            grid=self.grid,
            values=np.copy(self.psi),
            basis=self.basis
        )
    
    def _generate_grid(self, domain: List[Tuple[float, float]], grid_size: int) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate a grid based on domain and grid size."""
        if len(domain) == 1:
            # 1D case
            return np.linspace(domain[0][0], domain[0][1], grid_size)
        else:
            # Multi-dimensional case
            grids = []
            for d in range(self.dimension):
                grids.append(np.linspace(domain[d][0], domain[d][1], grid_size))
            
            return np.meshgrid(*grids, indexing='ij')
    
    def _evaluate_function(self, func: Callable) -> np.ndarray:
        """Evaluate the function on the grid."""
        if self.dimension == 1 and not isinstance(self.grid, list):
            # 1D case with simple grid
            psi = np.zeros(self.grid.shape, dtype=complex)
            for i in range(self.grid.size):
                psi[i] = func(self.grid[i])
            return psi
        else:
            # Multi-dimensional case or 1D with complex grid
            shape = self.grid[0].shape if isinstance(self.grid, list) else self.grid.shape
            psi = np.zeros(shape, dtype=complex)
            
            # Evaluate the function at each grid point
            grid_iterator = np.ndindex(shape)
            for idx in grid_iterator:
                if isinstance(self.grid, list):
                    point = np.array([grid[idx] for grid in self.grid])
                else:
                    point = self.grid[idx]
                psi[idx] = func(point)
            
            return psi
    
    def normalize(self) -> None:
        """Normalize the wavefunction."""
        # Calculate the norm
        norm = self.get_norm()
        
        # Normalize
        if norm > 0:
            self.psi /= np.sqrt(norm)
    
    def get_norm(self) -> float:
        """Calculate the norm of the wavefunction."""
        # Calculate |ψ|²
        prob_density = np.abs(self.psi)**2
        
        # Integrate over the grid using Simpson's rule for higher accuracy
        if self.dimension == 1 and not isinstance(self.grid, list):
            # 1D case with simple grid
            dx = (self.grid.max() - self.grid.min()) / (self.grid.size - 1)
            norm = scipy.integrate.simps(prob_density, dx=dx)
        else:
            # Multi-dimensional case or 1D with complex grid
            if isinstance(self.grid, list):
                # Use multi-dimensional integration
                # Prepare arrays for integration
                if self.dimension == 2:
                    # 2D case - use Simpson's rule in each dimension
                    x = self.grid[0]
                    y = self.grid[1]
                    dx = (x.max() - x.min()) / (x.shape[0] - 1)
                    dy = (y.max() - y.min()) / (y.shape[0] - 1)
                    
                    # First integrate along rows (y-direction)
                    row_integrals = np.zeros(prob_density.shape[0])
                    for i in range(prob_density.shape[0]):
                        row_integrals[i] = scipy.integrate.simps(prob_density[i], dx=dy)
                    
                    # Then integrate along columns (x-direction)
                    norm = scipy.integrate.simps(row_integrals, dx=dx)
                elif self.dimension == 3:
                    # 3D case
                    x = self.grid[0]
                    y = self.grid[1]
                    z = self.grid[2]
                    dx = (x.max() - x.min()) / (x.shape[0] - 1)
                    dy = (y.max() - y.min()) / (y.shape[0] - 1)
                    dz = (z.max() - z.min()) / (z.shape[0] - 1)
                    
                    # Integrate along z-direction first
                    xy_integrals = np.zeros((prob_density.shape[0], prob_density.shape[1]))
                    for i in range(prob_density.shape[0]):
                        for j in range(prob_density.shape[1]):
                            xy_integrals[i, j] = scipy.integrate.simps(prob_density[i, j], dx=dz)
                    
                    # Integrate along y-direction
                    x_integrals = np.zeros(prob_density.shape[0])
                    for i in range(prob_density.shape[0]):
                        x_integrals[i] = scipy.integrate.simps(xy_integrals[i], dx=dy)
                    
                    # Finally integrate along x-direction
                    norm = scipy.integrate.simps(x_integrals, dx=dx)
                else:
                    # For higher dimensions, fall back to numpy's trapz as a better alternative
                    # to the rectangular rule
                    # Get the grid spacing in each dimension
                    dx = []
                    for d in range(self.dimension):
                        x = self.grid[d]
                        dx.append((x.max() - x.min()) / (x.shape[0] - 1))
                    
                    # Use an iterative approach to apply trapezoidal rule in each dimension
                    result = prob_density.copy()
                    for d in range(self.dimension):
                        result = np.trapz(result, dx=dx[d], axis=self.dimension-d-1)
                    
                    norm = result
            else:
                # Use default dx for any other case with more accurate trapezoidal rule
                dx = (self.grid.max() - self.grid.min()) / (self.grid.size - 1)
                norm = np.trapz(prob_density, dx=dx)
        
        return norm
    
    def compute_gradient(self) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Compute the gradient of the wavefunction.
        
        Returns:
            The gradient of the wavefunction at each grid point
        """
        if self.dimension == 1 and not isinstance(self.grid, list):
            # For a 1D grid
            dx = (self.grid.max() - self.grid.min()) / (self.grid.size - 1)
            gradient = np.gradient(self.psi, dx)
            return gradient
        
        # For multi-dimensional grids
        gradients = []
        for d in range(self.dimension):
            # Get the spacing in this dimension
            x = self.grid[d]
            dx = (x.max() - x.min()) / (x.shape[0] - 1)
            
            # Compute the gradient along this dimension
            grad_d = np.gradient(self.psi, dx, axis=d)
            gradients.append(grad_d)
        
        return gradients
    
    def compute_laplacian(self) -> np.ndarray:
        """
        Compute the Laplacian of the wavefunction.
        
        Returns:
            The Laplacian of the wavefunction at each grid point
        """
        if self.dimension == 1 and not isinstance(self.grid, list):
            # 1D case
            dx = (self.grid.max() - self.grid.min()) / (self.grid.size - 1)
            return np.gradient(np.gradient(self.psi, dx), dx)
        
        # Initialize the Laplacian
        laplacian = np.zeros_like(self.psi)
        
        # Sum the second derivatives in each dimension
        for d in range(self.dimension):
            # Get the spacing in this dimension
            x = self.grid[d]
            dx = (x.max() - x.min()) / (x.shape[0] - 1)
            
            # Add the second derivative along this dimension
            laplacian += np.gradient(np.gradient(self.psi, dx, axis=d), dx, axis=d)
        
        return laplacian
    
    def to_vector(self) -> np.ndarray:
        """Convert the wavefunction to a flat vector."""
        return self.psi.flatten()
    
    def from_vector(self, vector: np.ndarray) -> None:
        """Reshape a flat vector back to the wavefunction grid."""
        self.psi = vector.reshape(self.psi.shape)
    
    def probability_density(self) -> np.ndarray:
        """
        Compute the probability density |ψ|².
        
        Returns:
            The probability density at each grid point
        """
        return np.abs(self.psi)**2
    
    def expectation_value(self, operator: Callable) -> complex:
        """
        Compute the expectation value of an operator.
        
        Args:
            operator: Function that applies the operator to the wavefunction
            
        Returns:
            The expectation value ⟨ψ|Â|ψ⟩
        """
        # Apply the operator to the wavefunction
        A_psi = operator(self)
        
        # Calculate ⟨ψ|Â|ψ⟩
        if self.dimension == 1 and not isinstance(self.grid, list):
            # 1D case with simple grid
            dx = (self.grid.max() - self.grid.min()) / (self.grid.size - 1)
            expectation = np.sum(np.conjugate(self.psi) * A_psi.psi) * dx
        else:
            # Multi-dimensional case or 1D with complex grid
            if isinstance(self.grid, list):
                # Get the grid spacing in each dimension
                dx = []
                for d in range(self.dimension):
                    x = self.grid[d]
                    dx.append((x.max() - x.min()) / (x.shape[0] - 1))
                
                # Compute the volume element
                dV = np.prod(dx)
                
                # Integrate
                expectation = np.sum(np.conjugate(self.psi) * A_psi.psi) * dV
            else:
                # Use default dx for any other case
                dx = 1.0 / self.psi.size
                expectation = np.sum(np.conjugate(self.psi) * A_psi.psi) * dx
        
        return expectation


class DecoherenceFunctional:
    """
    Class for computing the decoherence functional 𝒟[|ψ⟩] = |∇ψ|².
    """
    
    def __init__(self, wavefunction: WaveFunction):
        """
        Initialize the decoherence functional.
        
        Args:
            wavefunction: The wavefunction to evaluate
        """
        self.wavefunction = wavefunction
    
    @log_execution_time
    def evaluate(self) -> np.ndarray:
        """
        Evaluate the decoherence functional 𝒟[|ψ⟩] = |∇ψ|².
        
        Returns:
            The value of 𝒟[|ψ⟩] at each grid point
        """
        # Compute the gradient
        gradient = self.wavefunction.compute_gradient()
        
        # Compute |∇ψ|²
        if isinstance(gradient, list):
            # Multi-dimensional case
            gradient_squared = np.zeros_like(self.wavefunction.psi, dtype=float)
            for grad_d in gradient:
                gradient_squared += np.abs(grad_d)**2
        else:
            # 1D case
            gradient_squared = np.abs(gradient)**2
        
        return gradient_squared
    
    @log_execution_time
    def total(self) -> float:
        """
        Compute the total spatial complexity ∫|∇ψ|² dx.
        
        Returns:
            The total spatial complexity
        """
        # Evaluate the decoherence functional
        gradient_squared = self.evaluate()
        
        # Integrate over the grid
        if self.wavefunction.dimension == 1 and not isinstance(self.wavefunction.grid, list):
            # 1D case with simple grid
            dx = (self.wavefunction.grid.max() - self.wavefunction.grid.min()) / (self.wavefunction.grid.size - 1)
            total = np.sum(gradient_squared) * dx
        else:
            # Multi-dimensional case or 1D with complex grid
            if isinstance(self.wavefunction.grid, list):
                # Get the grid spacing in each dimension
                dx = []
                for d in range(self.wavefunction.dimension):
                    x = self.wavefunction.grid[d]
                    dx.append((x.max() - x.min()) / (x.shape[0] - 1))
                
                # Compute the volume element
                dV = np.prod(dx)
                
                # Integrate
                total = np.sum(gradient_squared) * dV
            else:
                # Use default dx for any other case
                dx = 1.0 / gradient_squared.size
                total = np.sum(gradient_squared) * dx
        
        return total


@dataclass
class Evolution:
    """Class to store the results of quantum evolution."""
    times: np.ndarray
    states: np.ndarray
    final_state: WaveFunction
    parameters: Dict[str, Any]


class ModifiedSchrodinger:
    """
    Solver for the modified Schrödinger equation.
    
    The modified Schrödinger equation includes the standard Hamiltonian evolution
    and a decoherence term based on spatial complexity:
    
    iℏ ∂|ψ⟩/∂t = Ĥ|ψ⟩ - iγℏ 𝒟[|ψ⟩]
    """
    
    def __init__(self, hamiltonian: Callable[[WaveFunction], WaveFunction]):
        """
        Initialize the solver with a Hamiltonian.
        
        Args:
            hamiltonian: Function that applies the Hamiltonian to a wavefunction
        """
        self.hamiltonian = hamiltonian
        self.constants = PHYSICAL_CONSTANTS
        logger.debug(f"ModifiedSchrodinger solver initialized with hamiltonian: {hamiltonian.__name__ if hasattr(hamiltonian, '__name__') else 'unnamed'}")
    
    def _evolution_function(self, 
                          t: float, 
                          psi_vec: np.ndarray, 
                          wavefunction: WaveFunction,
                          gamma: float,
                          progress_tracker: Optional[ProgressTracker] = None) -> np.ndarray:
        """
        Define the ODE function for the Schrödinger equation.
        
        This implements the right-hand side of:
        
        d|ψ⟩/dt = -i/ℏ Ĥ|ψ⟩ - γ 𝒟[|ψ⟩]
        
        The decoherence functional should actually cause wavefunction damping.
        
        Args:
            t: Current time
            psi_vec: Flattened wavefunction vector (1D)
            wavefunction: Original wavefunction object (for shape and grid information)
            gamma: Decoherence strength parameter
            progress_tracker: Optional progress tracker for long computations
            
        Returns:
            Time derivative of the wavefunction (1D)
        """
        # Ensure psi_vec is flattened completely to 1D
        psi_vec_flat = psi_vec.ravel()
        
        # Create a copy of the wavefunction with current values
        current_wf = wavefunction.copy()
        current_wf.from_vector(psi_vec_flat)
        
        # Standard Schrödinger term: -i/ℏ Ĥ|ψ⟩
        H_psi = self.hamiltonian(current_wf)
        d_psi_dt = -1j / self.constants.hbar * H_psi.to_vector().ravel()  # Ensure 1D
        
        # Add decoherence term: -γ 𝒟[|ψ⟩]
        if gamma > 0:
            # Calculate the decoherence functional
            decoherence = DecoherenceFunctional(current_wf)
            D_psi = decoherence.evaluate()
            
            # Flatten the decoherence values to 1D
            D_psi_vec = D_psi.ravel()
            
            # Ensure shapes match by using broadcasting rules properly
            if D_psi_vec.shape != psi_vec_flat.shape:
                logger.warning(f"Shape mismatch: D_psi_vec {D_psi_vec.shape} != psi_vec {psi_vec_flat.shape}")
                
                # Resize D_psi_vec to match psi_vec_flat's length
                if len(D_psi_vec) != len(psi_vec_flat):
                    # Either truncate or pad with zeros
                    if len(D_psi_vec) > len(psi_vec_flat):
                        D_psi_vec = D_psi_vec[:len(psi_vec_flat)]
                    else:
                        D_psi_vec = np.pad(D_psi_vec, (0, len(psi_vec_flat) - len(D_psi_vec)))
            
            # Corrected decoherence term: we're implementing the modified Schrödinger equation with:
            # d|ψ⟩/dt = -i/ℏ Ĥ|ψ⟩ - γ|∇ψ|²|ψ⟩
            # For non-unitary evolution that actually reduces the norm
            # The negative sign ensures the amplitude decreases over time
            d_psi_dt = d_psi_dt - gamma * D_psi_vec * psi_vec_flat
            
            # Log the maximum decoherence strength to help debug
            max_decoherence = np.max(gamma * D_psi_vec)
            if max_decoherence > 0:
                logger.debug(f"Max decoherence effect: {max_decoherence:.6e}")
        
        # Update progress if tracker is provided
        if progress_tracker is not None:
            progress_tracker.update(1)
        
        # Return result with same shape as input psi_vec
        if psi_vec.shape != psi_vec_flat.shape:
            return d_psi_dt.reshape(psi_vec.shape)
        else:
            return d_psi_dt
    
    @log_execution_time
    def solve(self, 
             initial_state: WaveFunction, 
             t_span: List[float], 
             dt: Optional[float] = None,
             gamma: Optional[float] = None,
             method: str = 'RK45',
             atol: float = 1e-8,
             rtol: float = 1e-6,
             max_step: Optional[float] = None,
             first_step: Optional[float] = None,
             progress_updates: bool = True,
             vectorized: bool = True) -> Evolution:
        """
        Solve the modified Schrödinger equation.
        
        Args:
            initial_state: Initial wavefunction
            t_span: [t_initial, t_final] time span to solve over
            dt: Time step for dense output (if None, use solver default)
            gamma: Decoherence parameter (if None, use physical constants)
            method: ODE solver method
            atol: Absolute tolerance for the solver
            rtol: Relative tolerance for the solver
            max_step: Maximum allowed step size for the solver
            first_step: Initial step size for the solver
            progress_updates: Whether to show progress updates
            vectorized: Whether the ODE function is vectorized
            
        Returns:
            Evolution object containing the results
        """
        # Use the fundamental value of gamma if not specified
        if gamma is None:
            gamma = PHYSICAL_CONSTANTS.get_gamma()
            logger.info(f"Using the fundamental value of gamma: {gamma:.4e} s^-1")
        else:
            logger.info(f"Using specified gamma value: {gamma:.4e} s^-1")
        
        # Initialize progress tracking
        t_initial, t_final = t_span
        estimated_steps = 1000  # A rough estimate based on typical solver behavior
        
        # Create progress tracker if progress updates are requested
        progress_tracker = None
        if progress_updates:
            progress_tracker = ProgressTracker(
                total_steps=estimated_steps,
                description="Quantum evolution",
                log_interval=1.0  # Update every second
            )
            logger.info(f"Starting quantum evolution from t={t_initial} to t={t_final}")
        
        # Initial condition - ensure it's a 1D array
        psi0 = initial_state.to_vector()
        psi0 = psi0.ravel()  # Ensure it's strictly 1D
        
        # Define the ODE system that preserves shape consistency
        def ode_system(t, y):
            # Ensure y is the same shape as initial psi0
            if y.shape != psi0.shape:
                y = y.reshape(psi0.shape)
            return self._evolution_function(t, y, initial_state, gamma, progress_tracker)
        
        # Prepare solver options
        solver_kwargs = {
            'method': method,
            'atol': atol,
            'rtol': rtol,
            'vectorized': vectorized,
            'dense_output': dt is not None
        }
        
        # Add optional parameters only if they are not None
        if max_step is not None:
            solver_kwargs['max_step'] = max_step
        
        if first_step is not None:
            solver_kwargs['first_step'] = first_step
        
        # Solve the ODE system
        logger.info(f"Solving quantum evolution with {method} method (atol={atol}, rtol={rtol})")
        solution = solve_ivp(
            ode_system,
            t_span,
            psi0,
            **solver_kwargs
        )
        
        # Check if the solution was successful
        if not solution.success:
            error_msg = f"ODE solver failed: {solution.message}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Generate dense output if requested
        if dt is not None:
            logger.debug(f"Generating dense output with dt={dt}")
            times = np.arange(t_initial, t_final, dt)
            states = solution.sol(times).T
        else:
            times = solution.t
            states = solution.y.T
        
        # Create the final state
        logger.debug("Creating final wavefunction state")
        final_state = WaveFunction(
            grid=initial_state.grid,
            values=states[-1].reshape(initial_state.psi.shape),
            basis=initial_state.basis,
            do_normalize=False  # Don't normalize to preserve the decoherence effect
        )
        
        # Complete progress tracking
        if progress_tracker:
            progress_tracker.complete()
        
        # Record parameters used
        parameters = {
            'gamma': gamma,
            'method': method,
            'atol': atol,
            'rtol': rtol,
            'max_step': max_step,
            'first_step': first_step,
            'success': solution.success,
            'message': solution.message,
            'nfev': solution.nfev,
            'njev': solution.njev,
            'nlu': solution.nlu if hasattr(solution, 'nlu') else None
        }
        
        # Log final statistics
        logger.info(f"Evolution complete: {solution.nfev} function evaluations, " 
                   f"{len(times)} time points, final norm: {final_state.get_norm():.6f}")
        
        return Evolution(times, states, final_state, parameters)


@log_execution_time
def free_particle_hamiltonian(wavefunction: WaveFunction) -> WaveFunction:
    """
    Hamiltonian for a free particle: Ĥ = -ℏ²/(2m) ∇²
    
    Args:
        wavefunction: The wavefunction to apply the Hamiltonian to
        
    Returns:
        The result of applying the Hamiltonian
    """
    # Get constants
    pc = PHYSICAL_CONSTANTS
    hbar = pc.hbar
    mass = scipy_constants.electron_mass  # Use scipy's electron mass
    
    # Compute the Laplacian
    laplacian = wavefunction.compute_laplacian()
    
    # Compute Ĥψ = -ℏ²/(2m) ∇²ψ
    result = WaveFunction(
        grid=wavefunction.grid,
        values=-hbar**2 / (2 * mass) * laplacian,
        basis=wavefunction.basis
    )
    
    return result


@log_execution_time
def harmonic_oscillator_hamiltonian(wavefunction: WaveFunction, 
                                   omega: float) -> WaveFunction:
    """
    Hamiltonian for a harmonic oscillator: Ĥ = -ℏ²/(2m) ∇² + (1/2)mω²x²
    
    Args:
        wavefunction: The wavefunction to apply the Hamiltonian to
        omega: Angular frequency of the oscillator
        
    Returns:
        The result of applying the Hamiltonian
    """
    # Get constants
    pc = PHYSICAL_CONSTANTS
    hbar = pc.hbar
    mass = scipy_constants.electron_mass  # Use scipy's electron mass
    
    # Compute the Laplacian
    laplacian = wavefunction.compute_laplacian()
    
    # Compute the potential term V(x) = (1/2)mω²x²
    if wavefunction.dimension > 1 or isinstance(wavefunction.grid, list):
        # Multi-dimensional harmonic oscillator
        r_squared = np.zeros_like(wavefunction.grid[0])
        for d in range(wavefunction.dimension):
            r_squared += wavefunction.grid[d]**2
        
        potential = 0.5 * mass * omega**2 * r_squared
    else:
        # 1D harmonic oscillator
        potential = 0.5 * mass * omega**2 * wavefunction.grid**2
    
    # Compute Ĥψ = -ℏ²/(2m) ∇²ψ + V(x)ψ
    H_psi = -hbar**2 / (2 * mass) * laplacian + potential * wavefunction.psi
    
    result = WaveFunction(
        grid=wavefunction.grid,
        values=H_psi,
        basis=wavefunction.basis
    )
    
    return result 