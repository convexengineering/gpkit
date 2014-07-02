import math
import unittest
from gpkit import GP, monify
from copy import deepcopy

solvers = ['cvxopt', 'mosek', 'mosek_cli']


class t_GP(unittest.TestCase):
    name = "t_"

    def test_trivial_gp(self):
        x, y = monify('x y')
        prob = GP(cost=(x + 2*y),
                  constraints=[x*y >= 1],
                  solver=self.solver)
        sol = prob.solve()
        self.assertAlmostEqual(sol['x'], math.sqrt(2.))
        self.assertAlmostEqual(sol['y'], 1/math.sqrt(2.))

testcases = [t_GP]

tests = []
for testcase in testcases:
    for solver in solvers:
        test = type(testcase.__name__+"_"+solver,
                    (testcase,), {})
        setattr(test, 'solver', solver)
        tests.append(test)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
