import unittest
import numpy as np
import sys
import os

from gpkit import settings

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
EXAMPLE_DIR = os.path.abspath(FILE_DIR+'../../../docs/source/examples')
WRITTEN = {}


class capture_stdout(object):
    def __init__(self, filename):
        self.filename = EXAMPLE_DIR+os.sep+"%s_output.txt" % filename

    def __enter__(self):
        if self.filename not in WRITTEN:
            self.original_stdout = sys.stdout
            sys.stdout = open(self.filename, "w")

    def __exit__(self, *args):
        if self.filename not in WRITTEN:
            sys.stdout.close()
            sys.stdout = self.original_stdout
            WRITTEN[self.filename] = True


class TestExamples(unittest.TestCase):

    def test_simple_sp(self):
        with capture_stdout("simple_sp"):
            import simple_sp

    def test_simple_box(self):
        with capture_stdout("simple_box"):
            import simple_box

    def test_x_greaterthan_1(self):
        with capture_stdout("x_greaterthan_1"):
            import x_greaterthan_1

    def test_beam(self):
        with capture_stdout("beam"):
            import beam

    def test_water_tank(self):
        with capture_stdout("water_tank"):
            import water_tank

    def test_simpleflight(self):
        with capture_stdout("simpleflight"):
            import simpleflight
        m = simpleflight.m
        # sweep with various solvers
        m.solve(solver=self.solver, verbosity=0)
        # check that the single solution is accurate
        m.substitutions = {}
        sol = m.solve(solver=self.solver, verbosity=0)
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
            sol_rat = sol["variables"][key]/freevarcheck[key]
            self.assertTrue(abs(1-sol_rat) < 1e-2)
        for key in consenscheck:
            sol_rat = sol["sensitivities"]["variables"][key]/consenscheck[key]
            self.assertTrue(abs(1-sol_rat) < 1e-2)


TESTS = []
if os.path.isdir(EXAMPLE_DIR):
    sys.path.insert(0, EXAMPLE_DIR)
    for solver in settings["installed_solvers"]:
        if solver:
            test = type(TestExamples.__name__+"_"+solver, (TestExamples,), {})
            setattr(test, "solver", solver)
            TESTS.append(test)

if __name__ == "__main__":
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
