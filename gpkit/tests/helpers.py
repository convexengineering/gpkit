"""Internal convenience classes and functions for gpkit unit testing"""
import unittest


def run_tests(tests):
    """Default way to run tests, to be called in __main__ methods"""
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))
    unittest.TextTestRunner(verbosity=2).run(suite)
