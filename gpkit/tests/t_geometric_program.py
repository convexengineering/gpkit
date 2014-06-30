import math
import unittest
from gpkit import GP, monify


class t_GP(unittest.TestCase):

    def test_simplest_gp_ever(self):
        x, y = monify('x y')
        for solver in ['cvxopt']:
            prob = GP(cost=(x + 2*y), constraints=[x*y >= 1], solver=solver)
            sol = prob.solve()
            self.assertAlmostEqual(sol['x'], math.sqrt(2.))
            self.assertAlmostEqual(sol['y'], 1/math.sqrt(2.))

tests = [t_GP]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
