"""Tests for GPSolutionArray class"""
import unittest
import time
import numpy as np
from gpkit import Variable, VectorVariable, Model, PosyArray
from gpkit.solution_array import SolutionArray
from gpkit.solution_array import results_table
from gpkit.varkey import VarKey


class TestSolutionArray(unittest.TestCase):

    def test_call(self):
        A = Variable('A', '-', 'Test Variable')
        prob = Model(A, [A >= 1])
        sol = prob.solve(verbosity=0)
        self.assertTrue(isinstance(sol(A), float))
        self.assertAlmostEqual(sol(A), 1.0, 10)

    def test_call_vector(self):
        n = 5
        x = VectorVariable(n, 'x')
        prob = Model(sum(x), [x >= 2.5])
        sol = prob.solve(verbosity=0)
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
        prob = Model(sum(x),
                  [x >= y, y >= z1, z1 >= z2, z2 >= z3, z3 >= z4, z4 >= L])
        sol = prob.solve(verbosity=0)
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
        m = Model(12/(W*H**3),
                  [H <= H_max,
                   H*W >= A_min,
                   P_max >= 2*H + 2*W])
        sol = m.solve(verbosity=0)
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
        gp = Model(x, [x >= 12])
        sol = gp.solve(verbosity=0)
        tab = sol.table()
        self.assertTrue(isinstance(tab, str))


class TestResultsTable(unittest.TestCase):
    """TestCase for results_table()"""

    def test_nan_printing(self):
        """Test that solution prints when it contains nans"""
        x = VarKey(name='x')
        data = {x: np.array([np.nan, 1., 1., 1., 1.])}
        title = "Free variables"
        printstr = results_table(data, title)
        self.assertTrue(" - " in printstr)  # nan is printed as " - "
        self.assertTrue(title in printstr)

TESTS = [TestSolutionArray, TestResultsTable]

if __name__ == '__main__':
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
