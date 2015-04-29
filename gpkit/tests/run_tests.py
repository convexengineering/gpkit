"""Scripts for running unit tests"""
import unittest
TESTS = []

from gpkit.tests import t_sub
TESTS += t_sub.TESTS

from gpkit.tests import t_vars
TESTS += t_vars.TESTS

from gpkit.tests import t_nomials
TESTS += t_nomials.TESTS

from gpkit.tests import t_constraints
TESTS += t_constraints.TESTS

from gpkit.tests import t_posy_array
TESTS += t_posy_array.TESTS

from gpkit.tests import t_geometric_program
TESTS += t_geometric_program.TESTS

from gpkit.tests import t_gp_solution_array
TESTS += t_gp_solution_array.TESTS


import gpkit


def run_tests(tests):
    """Default way to run tests, to be called in __main__ methods"""
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))
    unittest.TextTestRunner(verbosity=2).run(suite)


def run(xmloutput=False):
    """Run all gpkit unit tests.

    Arguments
    ---------
    xmloutput: bool
        If true, generate xml output files (used for continuous integration)
    """
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in TESTS:
        suite.addTests(loader.loadTestsFromTestCase(t))

    if xmloutput:
        import xmlrunner
        xmlrunner.XMLTestRunner(output='test_reports').run(suite)
    else:
        unittest.TextTestRunner(verbosity=2).run(suite)

    print("\n##################################"
          "####################################")
    print("Running with units disabled:")
    gpkit.disable_units()

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for t in TESTS:
        suite.addTests(loader.loadTestsFromTestCase(t))

    if xmloutput:
        xmlrunner.XMLTestRunner(output='test_reports_nounits').run(suite)
    else:
        unittest.TextTestRunner(verbosity=1).run(suite)

if __name__ == '__main__':
    run()
