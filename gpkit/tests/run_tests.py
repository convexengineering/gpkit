import unittest
tests = []

import t_utils
tests += t_utils.tests

import t_nomials
tests += t_nomials.tests

import t_array
tests += t_array.tests

import t_geometric_program
tests += t_geometric_program.tests

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
