import unittest
tests = []

from .t_sub import tests as t_sub
tests += t_sub

from .t_vars import tests as t_vars
tests += t_vars

from .t_nomials import tests as t_nomials
tests += t_nomials

from .t_constraints import tests as t_constraints
tests += t_constraints

from .t_posy_array import tests as t_posy_array
tests += t_posy_array

from .t_geometric_program import tests as t_geometric_program
tests += t_geometric_program

from .t_gp_solution_array import tests as t_gp_solution_array
tests += t_gp_solution_array


import gpkit

def run():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
    print("\n######################################################################")
    print("Running with units disabled:")
    gpkit.disableUnits()
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))
    unittest.TextTestRunner(verbosity=1).run(suite)

if __name__ == '__main__':
    run()
