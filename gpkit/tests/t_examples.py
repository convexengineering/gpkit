"""Unit testing of tests in docs/source/examples"""
import unittest
import os
import numpy as np

from gpkit import settings
from gpkit.tests.helpers import generate_example_tests
from gpkit.small_scripts import mag


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
                        r"\pi": -0.4785,
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
