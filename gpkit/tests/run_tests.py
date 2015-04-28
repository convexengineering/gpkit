import unittest
_TESTS = []

from .t_sub import _TESTS as t_sub
_TESTS += t_sub

from .t_vars import _TESTS as t_vars
_TESTS += t_vars

from .t_nomials import _TESTS as t_nomials
_TESTS += t_nomials

from .t_constraints import _TESTS as t_constraints
_TESTS += t_constraints

from .t_posy_array import _TESTS as t_posy_array
_TESTS += t_posy_array

from .t_geometric_program import _TESTS as t_geometric_program
_TESTS += t_geometric_program

from .t_gp_solution_array import _TESTS as t_gp_solution_array
_TESTS += t_gp_solution_array


import gpkit


def run(xmloutput=False):
    _SUITE = unittest.TestSuite()
    _LOADER = unittest.TestLoader()
    for t in _TESTS:
        _SUITE.addTests(_LOADER.loadTestsFromTestCase(t))

    if xmloutput:
        import xmlrunner
        xmlrunner.XMLTestRunner(output='test_reports').run(_SUITE)
    else:
        unittest.TextTestRunner(verbosity=2).run(_SUITE)

    print("\n######################################################################")
    print("Running with units disabled:")
    gpkit.disableUnits()

    _SUITE = unittest.TestSuite()
    _LOADER = unittest.TestLoader()
    for t in _TESTS:
        _SUITE.addTests(_LOADER.loadTestsFromTestCase(t))

    if xmloutput:
        xmlrunner.XMLTestRunner(output='test_reports_nounits').run(_SUITE)
    else:
        unittest.TextTestRunner(verbosity=1).run(_SUITE)

if __name__ == '__main__':
    run()
