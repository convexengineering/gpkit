"""Tests for tools module"""
import unittest
import numpy as np
from gpkit import Variable, VectorVariable, Model
from gpkit.tools import (zero_lower_unbounded, composite_objective,
                         te_exp_minus1)


class TestMathModels(unittest.TestCase):
    """TestCase for math models"""

    def test_composite_objective(self):
        L = Variable("L")
        W = Variable("W")
        eqns = [L >= 1, W >= 1,
                L*W == 10]
        obj = composite_objective(L+W, W**-1 * L**-3, sub={L: 1, W: 1})
        m = Model(obj, eqns)
        sol = m.solve(verbosity=0)
        a = sol["sensitivities"]["variables"]["w_{CO}"].flatten()
        b = np.array([0, 0.98809322, 0.99461408, 0.99688676, 0.99804287,
                      0.99874303, 0.99921254, 0.99954926, 0.99980255, 1])
        self.assertTrue((abs(a-b)/(a+b+1e-7) < 1e-7).all())

    def test_zero_lower_unbounded(self):
        x = Variable('x', value=4)
        y = Variable('y', value=0)
        z = Variable('z')
        t1 = Variable('t1')
        t2 = Variable('t2')

        prob = Model(z, [z >= x + t1,
                         t1 >= t2,
                         t2 >= y])
        zero_lower_unbounded(prob)
        sol = prob.solve(verbosity=0)

    def test_te_exp_minus1(self):
        """Test Taylor expansion of e^x - 1"""
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

TESTS = [TestMathModels]


if __name__ == '__main__':
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
