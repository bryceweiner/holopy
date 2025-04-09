"""
Logging utilities for HoloPy.

This module provides logging configuration and utilities for the HoloPy library.
"""

import logging
import sys
import os
import time
from functools import wraps
from typing import Callable, Any, Optional, Union, Dict

# Create a custom logger
logger = logging.getLogger('holopy')

# Default logging level
DEFAULT_LEVEL = logging.INFO

# Flag to track if logging has been configured
_logging_configured = False

def configure_logging(level: Union[int, str] = DEFAULT_LEVEL, 
                     file_path: Optional[str] = None,
                     console: bool = True) -> None:
    """
    Configure logging for HoloPy. This function should only be called once.
    
    Parameters
    ----------
    level : int or str, optional
        Logging level (e.g., logging.DEBUG, 'INFO', etc.)
    file_path : str, optional
        Path to log file. If provided, logs will be written to this file.
    console : bool, optional
        Whether to log to console.
    """
    global _logging_configured
    
    # Prevent multiple configurations
    if _logging_configured:
        logger.warning("Logging has already been configured. Skipping reconfiguration.")
        return
    
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Remove all handlers
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if requested
    if file_path:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    logger.setLevel(level)
    _logging_configured = True
    logger.info(f"HoloPy logging configured with level: {logging.getLevelName(level)}")
    logger.info("HoloPy v0.1.0 - Python Library for Holographic Cosmology and Holographic gravity Simulations")
    logger.info("For more information, see https://holopy.readthedocs.io")

def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Parameters
    ----------
    name : str, optional
        Logger name. If None, returns the root HoloPy logger.
    
    Returns
    -------
    logging.Logger
        Logger instance.
    """
    if name:
        return logging.getLogger(f'holopy.{name}')
    return logger

def log_execution_time(func: Callable = None, level: int = logging.INFO) -> Callable:
    """
    Decorator to log the execution time of a function.
    
    Parameters
    ----------
    func : callable, optional
        Function to decorate.
    level : int, optional
        Logging level for the timing message.
    
    Returns
    -------
    callable
        Decorated function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Format time based on duration using ASCII instead of Unicode
            if execution_time < 0.001:
                time_str = f"{execution_time * 1000000:.2f} us"  # microseconds
            elif execution_time < 1:
                time_str = f"{execution_time * 1000:.2f} ms"  # milliseconds
            else:
                time_str = f"{execution_time:.2f} s"  # seconds
                
            logger.log(level, f"Function '{func.__name__}' executed in {time_str}")
            return result
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)

# Progress tracking
class ProgressTracker:
    """Tracks progress of long-running computations."""
    
    def __init__(self, total_steps: int, description: str = "Progress", log_interval: float = 0.5):
        """
        Initialize a progress tracker.
        
        Parameters
        ----------
        total_steps : int
            Total number of steps in the computation.
        description : str, optional
            Description of the tracked process.
        log_interval : float, optional
            Minimum interval between log messages in seconds.
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.last_log_time = 0
        self.log_interval = log_interval
    
    def update(self, steps: int = 1, force_log: bool = False) -> None:
        """
        Update progress by the specified number of steps.
        
        Parameters
        ----------
        steps : int, optional
            Number of steps to advance.
        force_log : bool, optional
            Force logging even if the log interval hasn't elapsed.
        """
        self.current_step += steps
        current_time = time.time()
        
        # Only log if log_interval seconds have passed or it's forced
        if force_log or (current_time - self.last_log_time >= self.log_interval):
            percentage = min(100.0, (self.current_step / self.total_steps) * 100)
            elapsed = current_time - self.start_time
            
            # Estimate remaining time if we have progress
            if self.current_step > 0:
                remaining = elapsed * (self.total_steps - self.current_step) / self.current_step
                logger.info(f"{self.description}: {percentage:.1f}% ({self.current_step}/{self.total_steps}) - "
                           f"Elapsed: {self._format_time(elapsed)} - "
                           f"Remaining: {self._format_time(remaining)}")
            else:
                logger.info(f"{self.description}: {percentage:.1f}% ({self.current_step}/{self.total_steps}) - "
                           f"Elapsed: {self._format_time(elapsed)}")
            
            self.last_log_time = current_time
    
    def complete(self) -> None:
        """Mark the tracked process as complete."""
        self.current_step = self.total_steps
        elapsed = time.time() - self.start_time
        logger.info(f"{self.description} completed - Total time: {self._format_time(elapsed)}")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into a human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h" 