"""Unit testing of tests in docs/source/examples"""
import unittest
import os

from gpkit import settings
from gpkit.tests.helpers import generate_example_tests
from gpkit.small_scripts import mag
import gpkit


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

    def test_simple_sp(self, example):
        pass

    def test_simple_box(self, example):
        pass

    def test_x_greaterthan_1(self, example):
        pass

    def test_beam(self, example):
        pass

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

    def test_breakdown_example(self, example):
        if gpkit.units:
            self.assertAlmostEqual(mag(example.sol["cost"]), 8.448, 3)
            self.assertAlmostEqual(mag(example.sol("w2")), 6.448, 3)
        else:
            self.assertAlmostEqual(example.sol["cost"], 5, 4)
            self.assertAlmostEqual(example.sol("w2"), 3, 4)

##    def test_BoundedConstraintSet_ex(self, example):
##        self.assertAlmostEqual(example.sol["cost"], 0)
        
    def test_LCS_ex(self, example):
        self.assertAlmostEqual(example.sol["cost"], .5)
        
    def test_subinplace(self, example):
        self.assertAlmostEqual(example.sol["cost"], .5)

    def test_dual_infeasible_ex(self, example):
        with self.assertRaises(RuntimeWarning):
            example.m.solve()

    def test_dual_infeasible_ex2(self, example):
        with self.assertRaises((RuntimeWarning, ValueError)):
            example.m.solve()

    def test_primal_infeasible_ex1(self, example):
        with self.assertRaises(RuntimeWarning):
            example.m.solve()

    def test_primal_infeasible_ex2(self, example):
        with self.assertRaises(RuntimeWarning):
            example.m.solve()

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
EXAMPLE_DIR = os.path.abspath(FILE_DIR + '../../../docs/source/examples')
SOLVERS = settings["installed_solvers"]
TESTS = generate_example_tests(EXAMPLE_DIR, [TestExamples], SOLVERS)

if __name__ == "__main__":
    # pylint:disable=wrong-import-position
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
