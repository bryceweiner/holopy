"""
Simple script to run the tests directly without pytest configuration.
"""

import sys
import os
import unittest
import importlib.util

def import_module_from_path(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def run_test(test_file_path):
    """Run tests from a specific file."""
    print(f"Running tests from: {test_file_path}")
    
    # Import the test module
    module_name = os.path.basename(test_file_path).replace('.py', '')
    test_module = import_module_from_path(module_name, test_file_path)
    
    # Create a test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_module)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # List of test files to run
    test_files = [
        os.path.join('holopy', 'tests', 'unit', 'cosmology', 'test_expansion.py'),
        os.path.join('holopy', 'tests', 'unit', 'cosmology', 'test_correlation.py'),
    ]
    
    all_passed = True
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n{'='*80}\nRunning {test_file}\n{'='*80}")
            success = run_test(test_file)
            if not success:
                all_passed = False
                print(f"Tests failed in {test_file}!")
        else:
            print(f"Test file not found: {test_file}")
            all_passed = False
    
    if all_passed:
        print("\n✅ All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 