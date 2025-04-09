"""
Test runner for all dS/QFT module tests.

This file provides a convenient way to run all unit, integration, and regression
tests for the dS/QFT module.
"""

import unittest
import sys
import os
import logging
import time
import numpy as np

# Add the parent directory to the path so we can import the test modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging - avoid duplicate logging by checking if handlers already exist
logger = logging.getLogger(__name__)
root_logger = logging.getLogger()

# Only configure logging if it hasn't been configured
if not root_logger.handlers:
    # Create a more detailed formatter for better source identification
    formatter = logging.Formatter(
        '%(asctime)s - [%(name)30s] - %(levelname)5s - %(message)s'
    )
    
    # Configure the root logger
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    
    # Make test runner logs stand out
    logger.info("Logging configured with distinct module identification")

# Import test modules
from holopy.tests.unit.dsqft.test_dsqft_propagator import TestBulkBoundaryPropagator
from holopy.tests.unit.dsqft.test_dsqft_dictionary import TestFieldOperatorDictionary
from holopy.tests.unit.dsqft.test_dsqft_physical_accuracy import TestDSQFTPhysicalAccuracy
from holopy.tests.integration.test_dsqft_simulation import TestDSQFTSimulation
from holopy.tests.regression.test_dsqft_regression import TestDSQFTRegression

class TimingTextTestResult(unittest.TextTestResult):
    """Custom test result class that tracks timing for each test."""
    
    def __init__(self, *args, **kwargs):
        self.max_test_time = kwargs.pop('max_test_time', None)  # Maximum time for a single test in seconds
        super(TimingTextTestResult, self).__init__(*args, **kwargs)
        self.test_timings = {}
        self.current_test = None
    
    def startTest(self, test):
        """Start timing the test."""
        self.current_test = test
        self._test_start_time = time.time()
        logger.info(f"Starting test: {test}")
        super(TimingTextTestResult, self).startTest(test)
    
    def stopTest(self, test):
        """Stop timing the test and record the time."""
        elapsed = time.time() - self._test_start_time
        self.test_timings[test.id()] = elapsed
        logger.info(f"Completed test: {test} in {elapsed:.2f} seconds")
        self.current_test = None
        super(TimingTextTestResult, self).stopTest(test)
        
    def check_test_timeout(self):
        """Check if the current test has exceeded the maximum time limit."""
        if self.current_test and self.max_test_time:
            elapsed = time.time() - self._test_start_time
            if elapsed > self.max_test_time:
                logger.warning(f"Test {self.current_test} exceeded time limit of {self.max_test_time}s (running for {elapsed:.2f}s)")
                return True
        return False
    
    def printTimings(self):
        """Print the time taken for each test."""
        if not self.test_timings:
            return
            
        logger.info("\n===== TEST EXECUTION TIMES =====")
        for test_id, elapsed in sorted(self.test_timings.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"{test_id}: {elapsed:.2f} seconds")
        
        total_time = sum(self.test_timings.values())
        logger.info(f"Total execution time: {total_time:.2f} seconds")

class TimingTextTestRunner(unittest.TextTestRunner):
    """Custom test runner that uses TimingTextTestResult."""
    
    def __init__(self, *args, **kwargs):
        self.max_test_time = kwargs.pop('max_test_time', None)  # Maximum time for a single test in seconds
        kwargs['resultclass'] = TimingTextTestResult
        super(TimingTextTestRunner, self).__init__(*args, **kwargs)
    
    def _makeResult(self):
        return self.resultclass(self.stream, self.descriptions, self.verbosity, max_test_time=self.max_test_time)
    
    def run(self, test):
        """Run the tests and print timing information."""
        logger.info("=== STARTING dS/QFT TEST SUITE ===")
        if self.max_test_time:
            logger.info(f"Maximum test time set to {self.max_test_time} seconds per test")
            
        start_time = time.time()
        result = super(TimingTextTestRunner, self).run(test)
        elapsed = time.time() - start_time
        logger.info(f"=== TEST SUITE COMPLETED IN {elapsed:.2f} SECONDS ===")
        result.printTimings()
        return result

def run_tests(max_test_time=60):
    """
    Run all dS/QFT tests.
    
    Args:
        max_test_time (int, optional): Maximum time in seconds for a single test. Default is 60s.
                                        Set to None for no limit.
    """
    logger.info(f"Preparing dS/QFT test suite with max_test_time={max_test_time}s")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add unit tests
    logger.info("Adding unit tests to test suite")
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestBulkBoundaryPropagator))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestFieldOperatorDictionary))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestDSQFTPhysicalAccuracy))
    
    # Add integration tests
    logger.info("Adding integration tests to test suite")
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestDSQFTSimulation))
    
    # Add regression tests
    logger.info("Adding regression tests to test suite")
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestDSQFTRegression))
    
    # Run tests
    logger.info("Running all tests")
    runner = TimingTextTestRunner(verbosity=2, max_test_time=max_test_time)
    return runner.run(test_suite)

if __name__ == '__main__':
    # Set maximum test time to 10 seconds per test to prevent extremely long test runs
    # Reduced from 30 to 10 to identify issues more quickly
    MAX_TEST_TIME = 10  # seconds
    
    try:
        logger.info(f"Starting dS/QFT test runner with {MAX_TEST_TIME}s timeout per test")
        result = run_tests(max_test_time=MAX_TEST_TIME)
        logger.info("dS/QFT test runner completed")
        sys.exit(0 if result.wasSuccessful() else 1)
    except Exception as e:
        logger.error(f"Error running tests: {e}", exc_info=True)
        sys.exit(1) 