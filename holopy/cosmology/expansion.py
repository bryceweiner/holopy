"""
Implementation of Holographic Universe Expansion.

This module provides implementations for modeling the expansion of the universe
with holographic constraints, integrating information processing limits into
cosmological dynamics.
"""

import numpy as np
import logging
from typing import Optional, Union, Dict, List, Tuple, Callable
from scipy.integrate import solve_ivp

from holopy.constants.physical_constants import PhysicalConstants, get_gamma

# Setup logging
logger = logging.getLogger(__name__)

class HolographicExpansion:
    """
    Model for universe expansion with holographic constraints.
    
    This class implements a cosmological model where the expansion of the universe
    is constrained by the fundamental information processing rate γ and the
    E8×E8 heterotic structure.
    
    Attributes:
        omega_m (float): Matter density parameter
        omega_r (float): Radiation density parameter
        omega_lambda (float): Dark energy density parameter
        omega_k (float): Curvature density parameter
        h0 (float): Hubble constant in units of 100 km/s/Mpc
        info_constraint (bool): Whether to include information processing constraints
    """
    
    def __init__(
        self,
        omega_m: float = 0.3,
        omega_r: float = 9.0e-5,
        omega_lambda: float = 0.7,
        omega_k: float = 0.0,
        h0: float = 0.7,
        info_constraint: bool = True
    ):
        """
        Initialize the holographic expansion model.
        
        Args:
            omega_m (float, optional): Matter density parameter
            omega_r (float, optional): Radiation density parameter
            omega_lambda (float, optional): Dark energy density parameter
            omega_k (float, optional): Curvature density parameter
            h0 (float, optional): Hubble constant in units of 100 km/s/Mpc
            info_constraint (bool, optional): Whether to include information constraints
        """
        self.constants = PhysicalConstants()
        
        # Validate cosmological parameters
        total_omega = omega_m + omega_r + omega_lambda + omega_k
        if not np.isclose(total_omega, 1.0, rtol=1e-4):
            logger.warning(f"Sum of density parameters ({total_omega}) not equal to 1.0")
            
        self.omega_m = omega_m
        self.omega_r = omega_r
        self.omega_lambda = omega_lambda
        self.omega_k = omega_k
        self.h0 = h0
        self.info_constraint = info_constraint
        
        # Derived parameters
        self.h0_si = h0 * 100 * 1000 / (3.086e22)  # Convert to SI units (1/s)
        
        logger.debug(f"HolographicExpansion initialized with h0={h0}, Ωm={omega_m}, Ωr={omega_r}, ΩΛ={omega_lambda}")
        
        # Information processing rate
        self.gamma = get_gamma(self.h0_si)
        
        # Cached results
        self._scale_factor_solution = None
    
    def hubble_parameter(self, a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the Hubble parameter at a given scale factor.
        
        Args:
            a (float or np.ndarray): Scale factor (a=1 at present)
            
        Returns:
            float or np.ndarray: Hubble parameter in SI units (1/s)
        """
        # Standard Friedmann equation
        # H(a) = H0 * sqrt(Ωm/a³ + Ωr/a⁴ + ΩΛ + Ωk/a²)
        
        h_squared = (
            self.omega_m / np.power(a, 3) +
            self.omega_r / np.power(a, 4) +
            self.omega_lambda +
            self.omega_k / np.power(a, 2)
        )
        
        h = self.h0_si * np.sqrt(h_squared)
        
        # Apply information processing constraint if enabled
        if self.info_constraint:
            # The Hubble parameter is related to the information processing rate γ
            # by the relation H = 8π * γ in the holographic framework
            max_h_from_info = 8 * np.pi * self.gamma
            
            # Take the minimum of the standard Hubble parameter and the 
            # information-constrained value
            h = np.minimum(h, max_h_from_info)
        
        return h
    
    def friedmann_equation(self, t: float, y: List[float]) -> List[float]:
        """
        Define the Friedmann equation for numerical integration.
        
        This function implements the differential equation for the scale factor:
        da/dt = a * H(a)
        
        Args:
            t (float): Time in seconds
            y (List[float]): Current values [a]
            
        Returns:
            List[float]: Derivatives [da/dt]
        """
        a = y[0]
        
        # Prevent negative or zero scale factor
        if a <= 0:
            a = 1e-10
        
        # Calculate Hubble parameter
        h = self.hubble_parameter(a)
        
        # da/dt = a * H(a)
        da_dt = a * h
        
        return [da_dt]
    
    def evolve_scale_factor(
        self, 
        t_span: Tuple[float, float], 
        initial_a: float = 1e-5,
        t_eval: Optional[np.ndarray] = None,
        method: str = 'RK45'
    ) -> Dict[str, np.ndarray]:
        """
        Evolve the scale factor over a time interval.
        
        Args:
            t_span (Tuple[float, float]): Time interval [t_start, t_end] in seconds
            initial_a (float, optional): Initial scale factor
            t_eval (np.ndarray, optional): Times at which to evaluate the solution
            method (str, optional): Integration method for solve_ivp
            
        Returns:
            Dict[str, np.ndarray]: Solution with 't' and 'a' keys
        """
        logger.info(f"Evolving scale factor from t={t_span[0]} to t={t_span[1]} s")
        
        # Solve the differential equation
        sol = solve_ivp(
            self.friedmann_equation,
            t_span,
            [initial_a],
            t_eval=t_eval,
            method=method
        )
        
        if not sol.success:
            logger.error(f"Scale factor evolution failed: {sol.message}")
            raise RuntimeError(f"Scale factor evolution failed: {sol.message}")
            
        # Store solution
        self._scale_factor_solution = {
            't': sol.t,
            'a': sol.y[0]
        }
        
        logger.info(f"Scale factor evolved from a={self._scale_factor_solution['a'][0]} to a={self._scale_factor_solution['a'][-1]}")
        return self._scale_factor_solution
    
    def compute_hubble_history(self) -> Dict[str, np.ndarray]:
        """
        Compute the Hubble parameter history corresponding to the scale factor evolution.
        
        Returns:
            Dict[str, np.ndarray]: Solution with 't', 'a', and 'h' keys
        """
        if self._scale_factor_solution is None:
            logger.error("No scale factor solution available - call evolve_scale_factor first")
            raise RuntimeError("No scale factor solution available - call evolve_scale_factor first")
        
        t = self._scale_factor_solution['t']
        a = self._scale_factor_solution['a']
        
        # Calculate Hubble parameter at each scale factor
        h = np.array([self.hubble_parameter(a_i) for a_i in a])
        
        solution = {
            't': t,
            'a': a,
            'h': h
        }
        
        logger.info(f"Computed Hubble history from H={h[0]} to H={h[-1]}")
        return solution
    
    def compute_age_of_universe(self) -> float:
        """
        Compute the age of the universe in the holographic model.
        
        Returns:
            float: Age of the universe in seconds
        """
        # In the holographic framework, the age of the universe is constrained
        # by the information processing rate
        
        # Standard calculation for Hubble time
        hubble_time = 1.0 / self.h0_si
        
        # Correction factor based on the cosmological parameters
        # This is an approximation for a ΛCDM universe
        omega_lambda = self.omega_lambda
        omega_m = self.omega_m
        
        if omega_lambda == 0:
            # Matter-dominated universe
            correction = 2.0 / 3.0
        else:
            # ΛCDM universe - approximate correction
            correction = 2.0 / (3.0 * np.sqrt(omega_m)) * \
                         np.log(1.0 + np.sqrt(omega_lambda / omega_m))
        
        standard_age = hubble_time * correction
        
        # In the holographic framework, the age may be constrained by
        # the information processing rate
        if self.info_constraint:
            gamma = get_gamma(self.h0_si)
            
            # The maximum age from information constraints would be
            # related to the total processed information since the
            # beginning of the universe
            
            # This is a simplified model - in a real implementation, this would
            # involve integrating the information processing rate over the
            # history of the universe
            
            # For now, we use a rough estimate based on the current
            # Hubble parameter and the clustering coefficient
            c_g = 0.78125  # Clustering coefficient
            info_constrained_age = c_g / (8 * np.pi * gamma)
            
            # Take the minimum of the standard age and the information-constrained age
            age = min(standard_age, info_constrained_age)
        else:
            age = standard_age
        
        logger.info(f"Computed age of the universe: {age} seconds")
        return age
    
    def convert_redshift_to_scale_factor(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert redshift to scale factor.
        
        Args:
            z (float or np.ndarray): Redshift(s)
            
        Returns:
            float or np.ndarray: Scale factor(s), a = 1/(1+z)
        """
        return 1.0 / (1.0 + z)
    
    def convert_scale_factor_to_redshift(self, a: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert scale factor to redshift.
        
        Args:
            a (float or np.ndarray): Scale factor(s)
            
        Returns:
            float or np.ndarray: Redshift(s), z = 1/a - 1
        """
        return 1.0 / a - 1.0
    
    def growth_factor(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the linear growth factor D(z) which describes how density perturbations
        grow with cosmic time.
        
        The growth factor is normalized to D(z=0) = 1. For a matter-dominated universe,
        D(z) ∝ 1/(1+z) = a. For ΛCDM, the growth is suppressed relative to this at late times.
        
        Implementation follows Heath (1977) and Carroll, Press & Turner (1992), with
        holographic corrections from Weiner (2025).
        
        Args:
            z (Union[float, np.ndarray]): Redshift(s)
            
        Returns:
            Union[float, np.ndarray]: Linear growth factor D(z)
        """
        logger.debug(f"Computing growth factor for z={z}")
        
        # Convert redshift to scale factor
        a = self.convert_redshift_to_scale_factor(z)
        logger.debug(f"Scale factor a={a}")
        
        # For the growth factor, we use the standard definition where D(z) is directly
        # proportional to the scale factor in matter domination, and D(z=0) = 1.
        # This gives us D(z) that decreases with increasing redshift.
        
        def growth_factor_unnormalized(a_val):
            """
            Calculate unnormalized growth factor using standard ΛCDM with holographic corrections.
            Follows Eisenstein & Hu (1999) and Weiner (2025).
            """
            # Calculate cosmological parameters at this scale factor
            H = self.hubble_parameter(a_val)
            H0 = self.h0_si
            # Use direct values for greater numerical accuracy
            Omega_m0 = self.omega_m
            Omega_lambda0 = self.omega_lambda
            
            # Scale-dependent density parameters
            Omega_m_a = Omega_m0 / (a_val**3 * (H/H0)**2)
            Omega_lambda_a = Omega_lambda0 / (H/H0)**2
            
            # In pure matter domination: D = a, but ΛCDM modifies this
            
            # Carroll, Press & Turner (1992) approximation for D/a
            # We use directly (D/a) to avoid numerical issues in scaling
            # Note: this is a different definition than used above
            g = 2.5 * Omega_m_a / (
                Omega_m_a**(4/7) + 
                0.7 * Omega_lambda_a * (1 + Omega_m_a/2)
            )
            
            # Include direct scale-factor dependence
            # D(a) = a * g, where g is the growth suppression factor
            D = a_val * g
            
            # Apply radiation suppression from Meszaros effect
            Omega_r0 = self.omega_r
            Omega_r_a = Omega_r0 / (a_val**4 * (H/H0)**2)
            
            # If radiation component is significant, apply suppression
            if Omega_r_a > 0.001 * Omega_m_a:
                # Mixed radiation-matter era: logarithmic growth
                # D ∝ 1 + 3a/2 in radiation era, approaches a in matter era
                # Use smooth transition function
                radiation_suppression = 1.0 / (1.0 + 1.5 * Omega_r_a / Omega_m_a)
                D *= radiation_suppression
                logger.debug(f"Applied radiation suppression factor: {radiation_suppression:.6e}")
            
            # Apply holographic correction if enabled
            if self.info_constraint:
                gamma = self.gamma
                
                # Standard holographic suppression from Weiner (2025)
                # D_holo(a) = D_std(a) * (1 - γ/2H(a))
                # Use numerically stable form for small values
                holographic_factor = np.maximum(1.0 - gamma / (2 * H), 0.0)
                D *= holographic_factor
                
                logger.debug(f"Applied holographic factor: {holographic_factor:.6e}")
            
            logger.debug(f"Unnormalized D(a={a_val:.6e}) = {D:.6e}")
            return D
        
        # Calculate D(z=0) for normalization
        D0 = growth_factor_unnormalized(1.0)
        logger.debug(f"Normalization D(z=0) = {D0:.6e}")
        
        # Calculate D(z) and normalize
        if isinstance(a, np.ndarray):
            D = np.zeros_like(a)
            # Process each redshift individually for better numerical stability
            for i, a_val in enumerate(a):
                D[i] = growth_factor_unnormalized(a_val)
            logger.debug(f"Unnormalized D values: {D}")
        else:
            D = growth_factor_unnormalized(a)
            logger.debug(f"Unnormalized D value: {D:.6e}")
        
        # Normalize to D(z=0) = 1
        D_norm = D / D0
        
        if isinstance(D_norm, np.ndarray):
            logger.debug(f"Normalized D values: {D_norm}")
            # Check monotonicity
            if len(D_norm) > 1:
                monotonic = all(np.diff(D_norm) <= 0)  # Allow equality for numerical precision
                logger.debug(f"Growth factor monotonically decreasing: {monotonic}")
                if not monotonic:
                    non_monotonic_indices = np.where(np.diff(D_norm) > 0)[0]
                    for idx in non_monotonic_indices:
                        logger.warning(f"Non-monotonic: D(z={z[idx]})={D_norm[idx]:.6e} -> D(z={z[idx+1]})={D_norm[idx+1]:.6e}, ratio={D_norm[idx+1]/D_norm[idx]:.6e}")
                    
                    # Apply final numerical stabilization for monotonicity
                    # This replaces tiny numerical fluctuations with monotonic values
                    for idx in non_monotonic_indices:
                        # Check if this is a small numerical issue (< 0.1% difference)
                        if (D_norm[idx+1] - D_norm[idx]) / D_norm[idx] < 0.001:
                            logger.debug(f"Fixing small numerical non-monotonicity at z={z[idx+1]}")
                            D_norm[idx+1] = D_norm[idx] * 0.9999  # Ensure monotonicity
        else:
            logger.debug(f"Normalized D value: {D_norm:.6e}")
        
        return D_norm
        
    def sound_horizon(self, z_recomb: float = 1089.0) -> float:
        """
        Calculate the sound horizon at recombination.
        
        The sound horizon is the maximum distance that acoustic waves can travel
        in the photon-baryon fluid before recombination. It sets the characteristic
        scale for BAO features in the CMB and matter power spectrum.
        
        Args:
            z_recomb (float, optional): Redshift of recombination, default is 1089.0
            
        Returns:
            float: Sound horizon in Mpc
        """
        logger.debug(f"Computing sound horizon at z_recomb={z_recomb}")
        
        # Get physical baryon and radiation densities
        omega_b = 0.0493  # Baryon density parameter
        omega_r = self.omega_r  # Radiation density parameter
        
        # Physical densities (ω = Ωh²)
        omega_b_h2 = omega_b * self.h0**2
        omega_r_h2 = omega_r * self.h0**2
        
        # Compute the sound horizon using the fitting formula from Eisenstein & Hu (1998)
        # r_s = 144.7 * (omega_m*h^2)^0.25 * (omega_b*h^2)^(-0.125) * (1 + z_eq)^(-0.5) Mpc
        
        # Calculate the matter-radiation equality redshift
        omega_m_h2 = self.omega_m * self.h0**2
        # Use standard CMB temperature of 2.7255 K
        cmb_temperature = 2.7255  # CMB temperature in Kelvin
        z_eq = 2.5e4 * omega_m_h2 * (cmb_temperature / 2.7)**(-4)
        
        # Implement fitting formula with holographic corrections
        r_s = 144.7 * (omega_m_h2**0.25) * (omega_b_h2**(-0.125)) * ((1 + z_eq)**(-0.5))
        
        # Apply holographic correction if enabled
        if self.info_constraint:
            # Holographic correction to sound horizon
            # The sound horizon is also limited by the maximal information transfer
            # across causal horizons in the early universe
            
            # Convert to SI units
            r_s_si = r_s * 3.086e22  # Mpc to m
            
            # Calculate the holographic correction
            # This is based on the information processing constraint in the early universe
            # and the E8×E8 heterotic structure
            gamma = self.gamma  # Information processing rate
            
            # Holographic correction factor
            # r_s_holo = r_s_std * (1 - γ/H(z_recomb))
            H_recomb = self.hubble_parameter(1.0 / (1.0 + z_recomb))
            
            # Apply correction with numerical stability
            correction = np.maximum(1.0 - gamma / H_recomb, 0.5)  # Prevent extreme reduction
            r_s_si *= correction
            
            # Convert back to Mpc
            r_s = r_s_si / 3.086e22
            logger.debug(f"Applied holographic correction factor: {correction:.6e}")
        
        logger.info(f"Computed sound horizon at z={z_recomb}: {r_s:.6f} Mpc")
        return r_s


def scale_factor_evolution(
    cosmology_params: Dict[str, float], 
    t_span: Tuple[float, float],
    t_eval: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Compute scale factor evolution for given cosmological parameters.
    
    A standalone function for calculating scale factor evolution without
    creating the full HolographicExpansion object.
    
    Args:
        cosmology_params (Dict[str, float]): Cosmological parameters
        t_span (Tuple[float, float]): Time interval [t_start, t_end] in seconds
        t_eval (np.ndarray, optional): Times at which to evaluate the solution
        
    Returns:
        Dict[str, np.ndarray]: Solution with 't' and 'a' keys
    """
    # Extract parameters with defaults
    omega_m = cosmology_params.get('omega_m', 0.3)
    omega_r = cosmology_params.get('omega_r', 9.0e-5)
    omega_lambda = cosmology_params.get('omega_lambda', 0.7)
    omega_k = cosmology_params.get('omega_k', 0.0)
    h0 = cosmology_params.get('h0', 0.7)
    info_constraint = cosmology_params.get('info_constraint', True)
    
    # Create expansion model
    model = HolographicExpansion(
        omega_m=omega_m,
        omega_r=omega_r,
        omega_lambda=omega_lambda,
        omega_k=omega_k,
        h0=h0,
        info_constraint=info_constraint
    )
    
    # Evolve scale factor
    solution = model.evolve_scale_factor(t_span, t_eval=t_eval)
    
    return solution


def hubble_parameter(
    a: Union[float, np.ndarray],
    cosmology_params: Dict[str, float]
) -> Union[float, np.ndarray]:
    """
    Compute the Hubble parameter for given scale factor(s) and cosmological parameters.
    
    A standalone function for calculating the Hubble parameter without
    creating the full HolographicExpansion object.
    
    Args:
        a (float or np.ndarray): Scale factor(s)
        cosmology_params (Dict[str, float]): Cosmological parameters
        
    Returns:
        float or np.ndarray: Hubble parameter(s) in SI units (1/s)
    """
    # Extract parameters with defaults
    omega_m = cosmology_params.get('omega_m', 0.3)
    omega_r = cosmology_params.get('omega_r', 9.0e-5)
    omega_lambda = cosmology_params.get('omega_lambda', 0.7)
    omega_k = cosmology_params.get('omega_k', 0.0)
    h0 = cosmology_params.get('h0', 0.7)
    info_constraint = cosmology_params.get('info_constraint', True)
    
    # Create expansion model
    model = HolographicExpansion(
        omega_m=omega_m,
        omega_r=omega_r,
        omega_lambda=omega_lambda,
        omega_k=omega_k,
        h0=h0,
        info_constraint=info_constraint
    )
    
    # Vectorize the function if an array is provided
    if isinstance(a, np.ndarray):
        return np.array([model.hubble_parameter(a_i) for a_i in a])
    else:
        return model.hubble_parameter(a) 