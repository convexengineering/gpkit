"""Unit testing of tests in docs/source/examples"""
import unittest
import os
import pickle
import numpy as np

from gpkit import settings
from gpkit.tests.helpers import generate_example_tests
from gpkit.small_scripts import mag
from gpkit.small_classes import Quantity
from gpkit.constraints.loose import Loose
from gpkit import Model
from gpkit.exceptions import (UnknownInfeasible,
                              PrimalInfeasible, DualInfeasible, UnboundedGP)


def assert_logtol(first, second, logtol=1e-6):
    "Asserts that the logs of two arrays have a given abstol"
    np.testing.assert_allclose(np.log(mag(first)), np.log(mag(second)),
                               atol=logtol, rtol=0)


# pylint: disable=too-many-public-methods
class TestExamples(unittest.TestCase):
    """
    To test a new example, add a function called `test_$EXAMPLENAME`, where
    $EXAMPLENAME is the name of your example in docs/source/examples without
    the file extension.

    This function should accept two arguments (e.g. 'self' and 'example').
    The imported example script will be passed to the second: anything that
    was a global variable (e.g, "sol") in the original script is available
    as an attribute (e.g., "example.sol")

    If you don't want to perform any checks on the example besides making
    sure it runs, just put "pass" as the function's body, e.g.:

          def test_dummy_example(self, example):
              pass

    But it's good practice to ensure the example's solution as well, e.g.:

          def test_dummy_example(self, example):
              self.assertAlmostEqual(example.sol["cost"], 3.121)
    """

    # TODO: allow enabling plotting examples, make plots in correct folder...
    # def test_plot_sweep1d(self, _):
    #     import matplotlib.pyplot as plt
    #     plt.close("all")

    def test_autosweep(self, example):
        from gpkit import ureg
        bst1, tol1 = example.bst1, example.tol1
        bst2, tol2 = example.bst2, example.tol2

        l_ = np.linspace(1, 10, 100)
        for bst in [bst1, example.bst1_loaded]:
            sol1 = bst.sample_at(l_)
            assert_logtol(sol1("l"), l_)
            assert_logtol(sol1("A"), l_**2 + 1, tol1)
            assert_logtol(sol1["cost"], (l_**2 + 1)**2, tol1)
            self.assertEqual(Quantity(1.0, sol1("A").units),
                             Quantity(1.0, ureg.m)**2)

        ndig = -int(np.log10(tol2))
        self.assertAlmostEqual(bst2.cost_at("cost", 3), 1.0, ndig)
        # before corner
        A_bc = np.linspace(1, 3, 50)
        sol_bc = bst2.sample_at(A_bc)
        assert_logtol(sol_bc("A"), (A_bc/3)**0.5, tol2)
        assert_logtol(sol_bc["cost"], A_bc/3, tol2)
        # after corner
        A_ac = np.linspace(3, 10, 50)
        sol_ac = bst2.sample_at(A_ac)
        assert_logtol(sol_ac("A"), (A_ac/3)**2, tol2)
        assert_logtol(sol_ac["cost"], (A_ac/3)**4, tol2)

    def test_model_var_access(self, example):
        model = example.PS
        _ = model["E"]
        with self.assertRaises(ValueError):
            _ = model["m"]  # multiple variables called m

    def test_performance_modeling(self, example):
        m = Model(example.M.cost, Loose(example.M), example.M.substitutions)

        sol = m.solve(verbosity=0)
        sol.table()
        sol.save("solution.pkl")
        sol.table()
        sol_loaded = pickle.load(open("solution.pkl", "rb"))
        sol_loaded.table()

        sweepsol = m.sweep({example.AC.fuse.W: (50, 100, 150)}, verbosity=0)
        sweepsol.table()
        sweepsol.save("sweepsolution.pkl")
        sweepsol.table()
        sol_loaded = pickle.load(open("sweepsolution.pkl", "rb"))
        sol_loaded.table()

    def test_sp_to_gp_sweep(self, example):
        pass

    def test_boundschecking(self, example):  # pragma: no cover
        if "mosek_cli" in settings["default_solver"]:
            with self.assertRaises(UnknownInfeasible):
                example.gp.solve(verbosity=0)
        else:
            example.gp.solve(verbosity=0)  # mosek_conif and cvxopt solve it

    def test_vectorize(self, example):
        pass

    def test_primal_infeasible_ex1(self, example):
        primal_or_unknown = PrimalInfeasible
        if "cvxopt" in settings["default_solver"]:  # pragma: no cover
            primal_or_unknown = UnknownInfeasible
        with self.assertRaises(primal_or_unknown):
            example.m.solve(verbosity=0)

    def test_primal_infeasible_ex2(self, example):
        primal_or_unknown = PrimalInfeasible
        if "cvxopt" in settings["default_solver"]:  # pragma: no cover
            primal_or_unknown = UnknownInfeasible
        with self.assertRaises(primal_or_unknown):
            example.m.solve(verbosity=0)

    def test_docstringparsing(self, example):
        pass

    def test_debug(self, example):
        dual_or_primal = DualInfeasible
        if "mosek_conif" == settings["default_solver"]:  # pragma: no cover
            dual_or_primal = PrimalInfeasible
        with self.assertRaises(UnboundedGP):
            example.m.gp()
        with self.assertRaises(dual_or_primal):
            gp = example.m.gp(checkbounds=False)
            gp.solve(verbosity=0)

        primal_or_unknown = PrimalInfeasible
        if "cvxopt" == settings["default_solver"]:  # pragma: no cover
            primal_or_unknown = UnknownInfeasible
        with self.assertRaises(primal_or_unknown):
            example.m2.solve(verbosity=0)

        with self.assertRaises(UnboundedGP):
            example.m3.gp()
        with self.assertRaises(DualInfeasible):
            gp3 = example.m3.gp(checkbounds=False)
            gp3.solve(verbosity=0)

    def test_simple_sp(self, example):
        pass

    def test_simple_box(self, example):
        pass

    def test_x_greaterthan_1(self, example):
        pass

    def test_beam(self, example):
        self.assertFalse(np.isnan(example.sol("w")).any())

    def test_water_tank(self, example):
        pass

    def test_sin_approx_example(self, example):
        pass

    def test_external_sp(self, example):
        pass

    def test_simpleflight(self, example):
        self.assertTrue(example.sol.almost_equal(example.sol_loaded))
        for sol in [example.sol, example.sol_loaded]:
            freevarcheck = {
                "A": 8.46,
                "C_D": 0.0206,
                "C_f": 0.0036,
                "C_L": 0.499,
                "Re": 3.68e+06,
                "S": 16.4,
                "W": 7.34e+03,
                "V": 38.2,
                "W_w": 2.40e+03
            }
            # sensitivity values from p. 34 of W. Hoburg's thesis
            senscheck = {
                r"(\frac{S}{S_{wet}})": 0.4300,
                "e": -0.4785,
                "V_{min}": -0.3691,
                "k": 0.4300,
                r"\mu": 0.0860,
                "(CDA0)": 0.0915,
                "C_{L,max}": -0.1845,
                r"\tau": -0.2903,
                "N_{ult}": 0.2903,
                "W_0": 1.0107,
                r"\rho": -0.2275
            }
            for key in freevarcheck:
                sol_rat = mag(sol["variables"][key])/freevarcheck[key]
                self.assertTrue(abs(1-sol_rat) < 1e-2)
            for key in senscheck:
                sol_rat = sol["sensitivities"]["variables"][key]/senscheck[key]
                self.assertTrue(abs(1-sol_rat) < 1e-2)

    def test_relaxation(self, example):
        pass

    def test_unbounded(self, example):
        pass


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
EXAMPLE_DIR = os.path.abspath(FILE_DIR + '../../../docs/source/examples')
SOLVERS = settings["installed_solvers"]
if os.path.isdir(EXAMPLE_DIR):
    TESTS = generate_example_tests(EXAMPLE_DIR, [TestExamples], SOLVERS)
else:  # pragma: no cover
    TESTS = []

if __name__ == "__main__":  # pragma: no cover
    # pylint:disable=wrong-import-position
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
