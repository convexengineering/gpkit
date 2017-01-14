"""Tests for SolutionArray class"""
import unittest
import numpy as np
from gpkit import Variable, VectorVariable, Model, SignomialsEnabled
import gpkit
from gpkit.solution_array import results_table
from gpkit.varkey import VarKey
from gpkit.small_classes import Strings


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
        self.assertAlmostEqual(sol("y")/sol("x"), 1.0, 6)
        self.assertAlmostEqual(sol(x)/sol(y), 1.0, 6)

    def test_call_vector(self):
        n = 5
        x = VectorVariable(n, 'x')
        prob = Model(sum(x), [x >= 2.5])
        sol = prob.solve(verbosity=0)
        solx = sol(x)
        self.assertEqual(type(solx), np.ndarray)
        self.assertEqual(solx.shape, (n,))
        self.assertTrue((abs(solx - 2.5*np.ones(n)) < 1e-7).all())

    def test_subinto(self):
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
        self.assertAlmostEqual(0*gpkit.ureg.m,
                               np.max(np.abs(Pvals*gpkit.ureg.m - Psol)))
        self.assertAlmostEqual(0*gpkit.ureg.m,
                               np.max(np.abs(Psol - sol(P_max))))

    def test_table(self):
        x = Variable('x')
        gp = Model(x, [x >= 12])
        sol = gp.solve(verbosity=0)
        tab = sol.table()
        self.assertTrue(isinstance(tab, Strings))

    def test_units_sub(self):
        # issue 809
        T = Variable("T", "N", "thrust")
        Tmin = Variable("T_{min}", "N", "minimum thrust")
        m = Model(T, [T >= Tmin])
        tminsub = 1000 * gpkit.ureg.lbf
        m.substitutions.update({Tmin: tminsub})
        sol = m.solve(verbosity=0)
        self.assertEqual(sol(Tmin), tminsub)
        self.assertFalse(
            "1000N" in
            sol.table().replace(" ", "").replace("[", "").replace("]", ""))

    def test_key_options(self):
        # issue 993
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            m = Model(y, [y + 6*x >= 13 + x**2])
        msol = m.localsolve(verbosity=0)
        spsol = m.sp().localsolve(verbosity=0)  # pylint: disable=no-member
        gpsol = m.program.gps[-1].solve(verbosity=0)
        self.assertEqual(msol(x), msol("x"))
        self.assertEqual(spsol(x), spsol("x"))
        self.assertEqual(gpsol(x), gpsol("x"))
        self.assertEqual(msol(x), spsol(x))
        self.assertEqual(msol(x), gpsol(x))


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

    def test_result_access(self):
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            sig = (y + 6*x >= 13 + x**2)
        m = Model(y, [sig])
        sol = m.localsolve(verbosity=0)
        self.assertTrue(all([isinstance(gp.result.table(), Strings)
                             for gp in m.program.gps]))
        self.assertAlmostEqual(sol["cost"]/4.0, 1.0, 5)
        self.assertAlmostEqual(sol("x")/3.0, 1.0, 3)

TESTS = [TestSolutionArray, TestResultsTable]

if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
