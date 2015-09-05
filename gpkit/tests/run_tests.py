"""Script for running all gpkit unit tests"""
from gpkit.tests.helpers import run_tests
TESTS = []

from gpkit.tests import t_tools
TESTS += t_tools.TESTS

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

from gpkit.tests import t_model
TESTS += t_model.TESTS

from gpkit.tests import t_solution_array
TESTS += t_solution_array.TESTS

from gpkit.tests import t_small
TESTS += t_small.TESTS

from gpkit.tests import t_examples
TESTS += t_examples.TESTS


import gpkit


def run(xmloutput=False):
    """Run all gpkit unit tests.

    Arguments
    ---------
    xmloutput: bool
        If true, generate xml output files for continuous integration
    """
    if xmloutput:
        run_tests(TESTS, xmloutput='test_reports')
    else:
        run_tests(TESTS)
    print("\n##################################"
          "####################################")
    print("Running with units disabled:")
    gpkit.disable_units()
    if xmloutput:
        run_tests(TESTS, xmloutput='test_reports_nounits')
    else:
        run_tests(TESTS, verbosity=1)

if __name__ == '__main__':
    run()
