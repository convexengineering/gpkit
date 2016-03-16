"""Tests for SolutionArray class"""
import unittest
import time
import numpy as np
from gpkit import Variable, VectorVariable, Model
import gpkit
from gpkit.solution_array import results_table
from gpkit.varkey import VarKey


class TestSolutionArray(unittest.TestCase):
    """Unit tests for the SolutionArray class"""

    def test_call(self):
        A = Variable('A', '-', 'Test Variable')
        prob = Model(A, [A >= 1])
        sol = prob.solve(verbosity=0)
        self.assertAlmostEqual(sol(A), 1.0, 10)

    def test_call_units(self):
        # test from issue541
        x = Variable("x", 10, "ft")
        y = Variable("y", "m")
        m = Model(y, [y >= x])
        sol = m.solve(verbosity=0)
        self.assertAlmostEqual(sol("y")/sol("x"), 1.0)
        self.assertAlmostEqual(sol(x)/sol(y), 1.0)

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
        self.assertLess(time.time() - t1, 0.05)

    def test_subinto_sens(self):
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
        Psol = sol.subinto(P_max)
        self.assertEqual(len(Psol), Nsweep)
        self.assertAlmostEqual(0*gpkit.units.m,
                               np.max(np.abs(Pvals*gpkit.units.m - Psol)))
        self.assertAlmostEqual(0*gpkit.units.m,
                               np.max(np.abs(Psol - sol(P_max))))

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
        printstr = "\n".join(results_table(data, title))
        self.assertTrue(" - " in printstr)  # nan is printed as " - "
        self.assertTrue(title in printstr)

TESTS = [TestSolutionArray, TestResultsTable]

if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
