"""
Pytest configuration file for HoloPy tests.
"""

import pytest
import logging
from holopy.utils.logging import configure_logging
import os

def pytest_configure(config):
    """Configure logging for tests."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging to both console and file
    configure_logging(
        level='DEBUG',  # Set to DEBUG to see all log messages
        file_path='logs/test.log',
        console=True
    )
    
    # Ensure pytest doesn't capture the logs
    config.option.capture_log = False
    config.option.log_cli = True
    config.option.log_cli_level = 'DEBUG'
    config.option.log_cli_format = '%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)'
    config.option.log_cli_date_format = '%Y-%m-%d %H:%M:%S' 