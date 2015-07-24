"""Tests for tools module"""
import unittest
from gpkit import (GP, Variable, VectorVariable,
                   closest_feasible_point, make_feasibility_gp)


class TestFeasibilityHelpers(unittest.TestCase):
    """TestCase for the feasibility scripts"""

    def test_feasibility_gp_(self):
        x = Variable('x')
        gp = GP(x, [x**2 >= 1, x <= 0.5])
        self.assertRaises(RuntimeWarning, gp.solve, printing=False)
        fgp = make_feasibility_gp(gp, flavour=self.flavour)
        sol1 = fgp.solve(printing=False)
        sol2 = closest_feasible_point(gp, flavour=self.flavour, printing=False)
        self.assertAlmostEqual(sol1["cost"], sol2["cost"])


class TestMathModels(unittest.TestCase):
    """TestCase for math models"""

    def test_te_exp_minus1(self):
        """Test Taylor expansion of e^x - 1"""
        from gpkit.tools import te_exp_minus1
        x = Variable('x')
        self.assertEqual(te_exp_minus1(x, 1), x)
        self.assertEqual(te_exp_minus1(x, 3), x + x**2/2. + x**3/6.)
        self.assertRaises(ValueError, te_exp_minus1, x, 0)
        # make sure x was not modified
        self.assertEqual(x, Variable('x'))
        # try for VectorVariable too
        y = VectorVariable(3, 'y')
        self.assertEqual(te_exp_minus1(y, 1), y)
        self.assertEqual(te_exp_minus1(y, 3), y + y**2/2. + y**3/6.)
        self.assertRaises(ValueError, te_exp_minus1, y, 0)
        # make sure y was not modified
        self.assertEqual(y, VectorVariable(3, 'y'))

TEST_CASES = [TestFeasibilityHelpers]

TESTS = []
for testcase in TEST_CASES:
    for flavour in ["product", "max"]:
        test = type(testcase.__name__+"_"+flavour,
                    (testcase,), {})
        setattr(test, "flavour", flavour)
        TESTS.append(test)
TESTS.extend([TestMathModels])


if __name__ == '__main__':
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
