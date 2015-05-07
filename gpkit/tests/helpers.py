"""Internal convenience classes and functions for gpkit unit testing"""
import unittest


def run_tests(tests, xmloutput=None, verbosity=2):
    """Default way to run tests, to be called in __main__ methods.

    Arguments
    ---------
    tests: iterable of unittest.TestCase
    xmloutput: string or None
        if not None, generate xml output for continuous integration,
        with name given by the input string
    verbosity: int
        verbosity level for unittest.TextTestRunner
    """
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))
    if xmloutput:
        import xmlrunner
        xmlrunner.XMLTestRunner(output=xmloutput).run(suite)
    else:
        unittest.TextTestRunner(verbosity=verbosity).run(suite)
