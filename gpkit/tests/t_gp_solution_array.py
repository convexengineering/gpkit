import unittest
import numpy as np
from gpkit import Variable, VectorVariable, GP
from gpkit.geometric_program import GPSolutionArray


class t_GPSolutionArray(unittest.TestCase):

    def test_call(self):
        A = Variable('A', '-', 'Test Variable')
        prob = GP(A, [A >= 1])
        sol = prob.solve(printing=False)
        self.assertTrue(isinstance(sol(A), float))
        self.assertAlmostEqual(sol(A), 1.0, 10)

    def test_call_vector(self):
        n = 5
        x = VectorVariable(n, 'x')
        prob = GP(sum(x), [x >= 2.5])
        sol = prob.solve(printing=False)
        solx = sol(x)
        self.assertEqual(type(solx), np.ndarray)
        self.assertEqual(solx.shape, (n,))
        self.assertTrue((abs(solx - 2.5*np.ones(n)) < 1e-7).all())


tests = [t_GPSolutionArray]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
