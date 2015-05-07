import unittest
import numpy as np
from gpkit import (GP, Variable, closest_feasible_point, make_feasibility_gp)


class TestFeasibilityHelpers(unittest.TestCase):
    """TestCase for the feasibility scripts"""

    def test_feasibility_gp_(self):
        x = Variable('x')
        gp = GP(x, [x >= 1, x <= 0.5])
        self.assertRaises(RuntimeWarning, gp.solve, printing=False)
        fgp = make_feasibility_gp(gp, flavour=self.flavour)
        sol1 = fgp.solve(printing=False)
        sol2 = closest_feasible_point(gp, flavour=self.flavour, printing=False)
        self.assertAlmostEqual(sol1["cost"], sol2["cost"])

TEST_CASES = [TestFeasibilityHelpers]

TESTS = []
for testcase in TEST_CASES:
    for flavour in ["product", "max"]:
        test = type(testcase.__name__+"_"+flavour,
                    (testcase,), {})
        setattr(test, "flavour", flavour)
        TESTS.append(test)

if __name__ == '__main__':
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
