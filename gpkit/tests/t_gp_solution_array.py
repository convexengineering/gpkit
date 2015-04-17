import unittest
import time
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

    def test_call_time(self):
        N = 20
        x = VectorVariable(N, 'x', 'm')
        y = VectorVariable(N, 'y', 'm')
        z1 = VectorVariable(N, 'z1', 'm')
        z2 = VectorVariable(N, 'z2', 'm')
        z3 = VectorVariable(N, 'z3', 'm')
        z4 = VectorVariable(N, 'z4', 'm')
        L = Variable('L', 5, 'm')
        prob = GP(sum(x),
                  [x >= y, y >= z1, z1 >= z2, z2 >= z3, z3 >= z4, z4 >= L])
        sol = prob.solve(printing=False)
        t1 = time.time()
        foo = sol(z1)
        self.assertTrue(time.time() - t1 <= 0.05)

    def test_senssubinto(self):
        Nsweep = 20
        H_max = Variable("H_max", 10, "m", "Length")
        A_min = Variable("A_min", 10, "m^2", "Area")
        P_max = Variable("P", np.linspace(13, 24, Nsweep), "m", "Perimeter")
        H = Variable("H", "m", "Length")
        W = Variable("W", "m", "Width")
        gp = GP(12/(W*H**3),
                [H <= H_max,
                 H*W >= A_min,
                 P_max >= 2*H + 2*W])
        sol = gp.solve()
        Psens = sol.senssubinto(P_max)
        self.assertEqual(len(Psens), Nsweep)
        self.assertEqual(type(Psens), np.ndarray)
        self.assertAlmostEqual(Psens[-1], -4.)

tests = [t_GPSolutionArray]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
