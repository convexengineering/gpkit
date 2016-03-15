"""Script for running all gpkit unit tests"""
import gpkit
from gpkit.tests.helpers import run_tests


def import_tests():
    """Get a list of all GPkit unit test TestCases"""
    tests = []

    from gpkit.tests import t_tools
    tests += t_tools.TESTS

    from gpkit.tests import t_sub
    tests += t_sub.TESTS

    from gpkit.tests import t_vars
    tests += t_vars.TESTS

    from gpkit.tests import t_nomials
    tests += t_nomials.TESTS

    from gpkit.tests import t_constraints
    tests += t_constraints.TESTS

    from gpkit.tests import t_nomial_array
    tests += t_nomial_array.TESTS

    from gpkit.tests import t_model
    tests += t_model.TESTS

    from gpkit.tests import t_solution_array
    tests += t_solution_array.TESTS

    from gpkit.tests import t_small
    tests += t_small.TESTS

    from gpkit.tests import t_examples
    tests += t_examples.TESTS

    from gpkit.tests import t_keydict
    tests += t_keydict.TESTS

    return tests


def run(xmloutput=False):
    """Run all gpkit unit tests.

    Arguments
    ---------
    xmloutput: bool
        If true, generate xml output files for continuous integration
    """
    tests = import_tests()
    if xmloutput:
        run_tests(tests, xmloutput='test_reports')
    else:
        run_tests(tests)
    print("\n##################################"
          "####################################")
    print("Running with units disabled:")
    gpkit.disable_units()
    if xmloutput:
        run_tests(tests, xmloutput='test_reports_nounits')
    else:
        run_tests(tests, verbosity=1)

if __name__ == '__main__':
    run()
