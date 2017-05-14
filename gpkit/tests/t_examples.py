"""Unit testing of tests in docs/source/examples"""
import unittest
import os
import numpy as np

from gpkit import settings
from gpkit.tests.helpers import generate_example_tests
from gpkit.small_scripts import mag
from gpkit.small_classes import Quantity


def assert_logtol(first, second, logtol=1e-6):
    "Asserts that the logs of two arrays have a given abstol"
    np.testing.assert_allclose(np.log(mag(first)), np.log(mag(second)),
                               atol=logtol, rtol=0)


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
        from gpkit import units, ureg
        bst1, tol1 = example.bst1, example.tol1
        bst2, tol2 = example.bst2, example.tol2

        l_ = np.linspace(1, 10, 100)
        sol1 = bst1.sample_at(l_)
        assert_logtol(sol1("l"), l_)
        assert_logtol(sol1("A"), l_**2 + 1, tol1)
        assert_logtol(sol1["cost"], (l_**2 + 1)**2, tol1)
        if units:
            self.assertEqual(Quantity(1.0, sol1["cost"].units),
                             Quantity(1.0, ureg.m)**4)
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
        with self.assertRaises(KeyError):
            _ = model.topvar("E") # try to get a topvar in a submodel

    def test_performance_modeling(self, example):
        pass

    def test_vectorize(self, example):
        pass

    def test_primal_infeasible_ex1(self, example):
        with self.assertRaises(RuntimeWarning) as cm:
            example.m.solve(verbosity=0)
        err = cm.exception
        if "mosek" in err.message:
            self.assertIn("PRIM_INFEAS_CER", err.message)
        elif "cvxopt" in err.message:
            self.assertIn("unknown", err.message)

    def test_primal_infeasible_ex2(self, example):
        with self.assertRaises(RuntimeWarning):
            example.m.solve(verbosity=0)

    def test_debug(self, example):
        pass

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

    def test_external_sp2(self, example):
        pass

    def test_simpleflight(self, example):
        sol = example.sol
        freevarcheck = dict(A=8.46,
                            C_D=0.0206,
                            C_f=0.0036,
                            C_L=0.499,
                            Re=3.68e+06,
                            S=16.4,
                            W=7.34e+03,
                            V=38.2,
                            W_w=2.40e+03)
        # sensitivity values from p. 34 of W. Hoburg's thesis
        consenscheck = {r"(\frac{S}{S_{wet}})": 0.4300,
                        "e": -0.4785,
                        "V_{min}": -0.3691,
                        "k": 0.4300,
                        r"\mu": 0.0860,
                        "(CDA0)": 0.0915,
                        "C_{L,max}": -0.1845,
                        r"\tau": -0.2903,
                        "N_{ult}": 0.2903,
                        "W_0": 1.0107,
                        r"\rho": -0.2275}
        for key in freevarcheck:
            sol_rat = mag(sol["variables"][key])/freevarcheck[key]
            self.assertTrue(abs(1-sol_rat) < 1e-2)
        for key in consenscheck:
            sol_rat = sol["sensitivities"]["constants"][key]/consenscheck[key]
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
else:
    TESTS = []

if __name__ == "__main__":
    # pylint:disable=wrong-import-position
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
