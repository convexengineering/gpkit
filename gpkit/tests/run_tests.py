import unittest
tests = []

import t_sub
tests += t_sub.tests

import t_vars
tests += t_vars.tests

import t_nomials
tests += t_nomials.tests

import t_constraints
tests += t_constraints.tests

import t_posy_array
tests += t_posy_array.tests

import t_geometric_program
tests += t_geometric_program.tests

import t_gp_solution_array
tests += t_gp_solution_array.tests


import gpkit


def run(xmloutput=False):

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    if xmloutput:
        import xmlrunner
        xmlrunner.XMLTestRunner(output='test_reports').run(suite)
    else:
        unittest.TextTestRunner(verbosity=2).run(suite)

    print "\n#################################################################"
    print "Running with units disabled:"
    gpkit.disableUnits()

    if xmloutput:
        xmlrunner.XMLTestRunner(output='test_reports_nounits').run(suite)
    else:
        unittest.TextTestRunner(verbosity=1).run(suite)

if __name__ == '__main__':
    run()
