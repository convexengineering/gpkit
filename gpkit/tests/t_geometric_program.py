import math
import unittest
from gpkit import GP, monify

NDIGITS = {'cvxopt': 5, 'mosek': 7, 'mosek_cli': 7}
SOLVERS = NDIGITS.keys()


class t_GP(unittest.TestCase):
    name = "t_"

    def test_trivial_gp(self):
        x, y = monify('x y')
        prob = GP(cost=(x + 2*y),
                  constraints=[x*y >= 1],
                  solver=self.solver)
        sol = prob.solve()
        ndig = NDIGITS[self.solver]
        self.assertAlmostEqual(sol['x'], math.sqrt(2.), ndig)
        self.assertAlmostEqual(sol['y'], 1/math.sqrt(2.), ndig)
        self.assertAlmostEqual(sol['x'] + 2*sol['y'], 2*math.sqrt(2), ndig)

testcases = [t_GP]

tests = []
for testcase in testcases:
    for solver in SOLVERS:
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
