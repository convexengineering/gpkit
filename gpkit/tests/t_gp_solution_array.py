"""Tests for GPSolutionArray class"""
import unittest
import time
import numpy as np
from gpkit import Variable, VectorVariable, GP, PosyArray
from gpkit.geometric_program import GPSolutionArray


class TestGPSolutionArray(unittest.TestCase):

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
        _ = sol(z1)
        self.assertTrue(time.time() - t1 <= 0.05)

    def test_subinto_senssubinto(self):
        Nsweep = 20
        Pvals = np.linspace(13, 24, Nsweep)
        H_max = Variable("H_max", 10, "m", "Length")
        A_min = Variable("A_min", 10, "m^2", "Area")
        P_max = Variable("P", Pvals, "m", "Perimeter")
        H = Variable("H", "m", "Length")
        W = Variable("W", "m", "Width")
        gp = GP(12/(W*H**3),
                [H <= H_max,
                 H*W >= A_min,
                 P_max >= 2*H + 2*W])
        sol = gp.solve(printing=False)
        Psens = sol.senssubinto(P_max)
        self.assertEqual(len(Psens), Nsweep)
        self.assertEqual(type(Psens), np.ndarray)
        self.assertAlmostEqual(Psens[-1], -4., 6)
        Psol = sol.subinto(P_max)
        self.assertEqual(len(Psol), Nsweep)
        self.assertEqual(type(Psol), PosyArray)
        self.assertAlmostEqual(0, np.max(np.abs(Pvals - Psol.c)))
        self.assertAlmostEqual(0, np.max(np.abs(Psol.c - sol(P_max))))

    def test_table(self):
        x = Variable('x')
        gp = GP(x, [x >= 12])
        sol = gp.solve(solver='mosek', printing=False)
        tab = sol.table()
        self.assertTrue(isinstance(tab, str))

TESTS = [TestGPSolutionArray]

if __name__ == '__main__':
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
