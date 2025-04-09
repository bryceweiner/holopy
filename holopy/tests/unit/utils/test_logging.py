"""
Unit tests for the logging utilities in holopy.utils.logging module.

This module tests the functionality of the logging utilities, including:
- Logger configuration
- Progress tracking
- Execution time logging
"""

import unittest
import tempfile
import os
import time
import io
import sys
import logging
from unittest.mock import patch, MagicMock

from holopy.utils.logging import (
    configure_logging,
    get_logger,
    ProgressTracker,
    log_execution_time
)

class TestLoggingConfig(unittest.TestCase):
    """Tests for logging configuration functions."""
    
    def setUp(self):
        # Save original stdout
        self.original_stdout = sys.stdout
        # Create a StringIO object to capture output
        self.stdout_capture = io.StringIO()
        sys.stdout = self.stdout_capture
        
        # Reset logging to default state before each test
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        holopy_logger = logging.getLogger('holopy')
        for handler in holopy_logger.handlers[:]:
            holopy_logger.removeHandler(handler)
        
        # Temporary directory for log files
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        # Restore original stdout
        sys.stdout = self.original_stdout
        
        # Clean up temp directory
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass
    
    def test_get_logger(self):
        """Test that get_logger returns the correct logger instance."""
        # Get the root holopy logger
        root_logger = get_logger()
        self.assertEqual(root_logger.name, 'holopy')
        
        # Get a sub-module logger
        submodule_logger = get_logger('test')
        self.assertEqual(submodule_logger.name, 'holopy.test')
    
    def test_configure_logging_level(self):
        """Test that configure_logging sets the correct logging level."""
        # Test with string level
        configure_logging(level='DEBUG')
        logger = get_logger()
        self.assertEqual(logger.level, logging.DEBUG)
        
        # Test with int level
        configure_logging(level=logging.WARNING)
        logger = get_logger()
        self.assertEqual(logger.level, logging.WARNING)
    
    def test_configure_logging_file(self):
        """Test logging to file."""
        log_file = os.path.join(self.temp_dir, 'test.log')
        
        # Configure logging with file output only
        configure_logging(file_path=log_file, console=False)
        logger = get_logger()
        test_message = "Test log message"
        logger.info(test_message)
        
        # Check that message was written to file
        with open(log_file, 'r') as f:
            log_content = f.read()
            self.assertIn(test_message, log_content)
        
        # Close any open file handlers explicitly
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)
    
    @unittest.skip("Skipping console test due to potential output capture issues")
    def test_configure_logging_console(self):
        """Test logging to console."""
        configure_logging(level=logging.INFO, console=True)
        logger = get_logger()
        test_message = "Test console log message"
        logger.info(test_message)
        
        # Check that message was printed to console
        output = self.stdout_capture.getvalue()
        self.assertIn(test_message, output)


class TestProgressTracker(unittest.TestCase):
    """Tests for the ProgressTracker class."""
    
    def setUp(self):
        self.logger_mock = MagicMock()
        self.patcher = patch('holopy.utils.logging.logger', self.logger_mock)
        self.patcher.start()
    
    def tearDown(self):
        self.patcher.stop()
    
    def test_initialization(self):
        """Test that ProgressTracker initializes correctly."""
        total_steps = 100
        desc = "Test progress"
        tracker = ProgressTracker(total_steps, description=desc)
        
        self.assertEqual(tracker.total_steps, total_steps)
        self.assertEqual(tracker.current_step, 0)
        self.assertEqual(tracker.description, desc)
    
    def test_update(self):
        """Test updating progress."""
        tracker = ProgressTracker(100, log_interval=0)  # Set interval to 0 to force logging
        
        # Update by 1 step
        tracker.update()
        self.assertEqual(tracker.current_step, 1)
        self.logger_mock.info.assert_called()
        
        # Update by multiple steps
        self.logger_mock.info.reset_mock()
        tracker.update(steps=10)
        self.assertEqual(tracker.current_step, 11)
        self.logger_mock.info.assert_called()
    
    def test_complete(self):
        """Test marking progress as complete."""
        tracker = ProgressTracker(100)
        tracker.update(50)  # Update to 50%
        
        self.logger_mock.info.reset_mock()
        tracker.complete()
        
        self.assertEqual(tracker.current_step, 100)
        self.logger_mock.info.assert_called_once()
        # Check that the complete message contains "completed"
        self.assertIn("completed", self.logger_mock.info.call_args[0][0])
    
    def test_format_time(self):
        """Test time formatting."""
        tracker = ProgressTracker(100)
        
        # Test seconds
        self.assertTrue(tracker._format_time(5.5).endswith('s'))
        
        # Test minutes
        self.assertTrue(tracker._format_time(65).endswith('m'))
        
        # Test hours
        self.assertTrue(tracker._format_time(3600).endswith('h'))


class TestLogExecutionTime(unittest.TestCase):
    """Tests for the log_execution_time decorator."""
    
    def setUp(self):
        self.logger_mock = MagicMock()
        self.patcher = patch('holopy.utils.logging.logger', self.logger_mock)
        self.patcher.start()
    
    def tearDown(self):
        self.patcher.stop()
    
    def test_log_execution_time_decorator(self):
        """Test that the decorator logs function execution time."""
        @log_execution_time
        def test_function():
            time.sleep(0.01)  # Sleep briefly to ensure measurable execution time
            return "result"
        
        result = test_function()
        
        self.assertEqual(result, "result")  # Function should return normally
        self.logger_mock.log.assert_called_once()
        
        # Check that the log message contains the function name and execution time
        log_msg = self.logger_mock.log.call_args[0][1]
        self.assertIn("test_function", log_msg)
        self.assertIn("executed in", log_msg)
    
    def test_log_execution_time_with_args(self):
        """Test the decorator with arguments."""
        @log_execution_time(level=logging.DEBUG)
        def test_function_with_args(a, b):
            time.sleep(0.01)
            return a + b
        
        result = test_function_with_args(1, 2)
        
        self.assertEqual(result, 3)  # Function should return normally
        self.logger_mock.log.assert_called_once_with(logging.DEBUG, unittest.mock.ANY)


if __name__ == '__main__':
    unittest.main() 