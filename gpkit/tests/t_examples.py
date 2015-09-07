import unittest
import numpy as np
import sys
import os
import importlib

from gpkit import settings

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
EXAMPLE_DIR = os.path.abspath(FILE_DIR+'../../../docs/source/examples')
SOLVERS = settings["installed_solvers"]
IMPORTED_EXAMPLES = {}


class TestExamples(unittest.TestCase):

    # To test a new example, add a function called `test_$EXAMPLENAME`, where
    # $EXAMPLENAME is the name of your example in docs/source/examples without
    # the file extension.
    #
    # This function should accept two arguments (e.g. 'self' and 'example').
    # The imported example script will be passed to the second: anything that
    # was a global variable (e.g, "sol") in the original script is available
    # as an attribute (e.g., "example.sol")
    #
    # If you don't want to perform any checks on the example besides making
    # sure it runs, just put "pass" as the function's body, e.g.:
    #
    #       def test_dummy_example(self, example):
    #           pass
    #
    # But it's good practice to ensure the example's solution as well, e.g.:
    #
    #       def test_dummy_example(self, example):
    #           self.assertAlmostEqual(example.sol["cost"], 3.121)
    #

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
            sol_rat = sol["variables"][key]/freevarcheck[key]
            self.assertTrue(abs(1-sol_rat) < 1e-2)
        for key in consenscheck:
            sol_rat = sol["sensitivities"]["variables"][key]/consenscheck[key]
            self.assertTrue(abs(1-sol_rat) < 1e-2)


class NewDefaultSolver(object):
    def __init__(self, solver):
        self.solver = solver

    def __enter__(self):
        import gpkit
        gpkit.settings["installed_solvers"] = [self.solver]

    def __exit__(self, *args):
        import gpkit
        gpkit.settings["installed_solvers"] = SOLVERS


class NullFile(object):
    def write(self, string):
        pass

    def close(self):
        pass


class StdoutCaptured(object):
    def __init__(self, logfilename=None):
        self.logfilename = logfilename

    def __enter__(self):
        self.original_stdout = sys.stdout
        if self.logfilename:
            filepath = EXAMPLE_DIR+os.sep+"%s_output.txt" % self.logfilename
            logfile = open(filepath, "w")
        else:
            logfile = NullFile()
        sys.stdout = logfile

    def __exit__(self, *args):
        sys.stdout.close()
        sys.stdout = self.original_stdout


def new_test(name, solver):
    def test(self):
        with NewDefaultSolver(solver):
            logfilename = name if name not in IMPORTED_EXAMPLES else None
            with StdoutCaptured(logfilename):
                if name not in IMPORTED_EXAMPLES:
                    IMPORTED_EXAMPLES[name] = importlib.import_module(name)
                else:
                    reload(IMPORTED_EXAMPLES[name])
            getattr(self, name)(IMPORTED_EXAMPLES[name])
    return test


TESTS = []
if os.path.isdir(EXAMPLE_DIR):
    sys.path.insert(0, EXAMPLE_DIR)
    for fn in dir(TestExamples):
        if fn[:5] == "test_":
            name = fn[5:]
            old_test = getattr(TestExamples, fn)
            setattr(TestExamples, name, old_test)  # move to a non-test fn
            delattr(TestExamples, fn)  # delete the old old_test
            for solver in SOLVERS:
                new_name = "test_%s_%s" % (name, solver)
                setattr(TestExamples, new_name, new_test(name, solver))
    TESTS.append(TestExamples)

if __name__ == "__main__":
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
