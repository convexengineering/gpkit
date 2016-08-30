"""Tests for tools module"""
import unittest
import numpy as np
from gpkit import Variable, VectorVariable, Model
from gpkit.tools import (composite_objective, te_exp_minus1)
from gpkit.tools.fmincon import generate_mfiles


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
        a = sol["cost"]
        b = np.array([1.58856898, 2.6410391, 3.69348122, 4.74591386])
        self.assertTrue((abs(a-b)/(a+b+1e-7) < 1e-7).all())

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

    def test_fmincon_generator(self):
        """Test fmincon comparison tool"""
        x = Variable('x')
        y = Variable('y')
        m = Model(x, [x**3.2 >= 17*y + y**-0.2,
                      x >= 2,
                      y == 4])
        obj, c, ceq, DC, DCeq = generate_mfiles(m, writefiles=False)
        self.assertEqual(obj, 'x(2)')
        self.assertEqual(c, ['-x(2)**3.2 + 17*x(1) + x(1)**-0.2', '-x(2) + 2'])
        self.assertEqual(ceq, ['-x(1) + 4'])
        self.assertEqual(DC, ['-0.2*x(1).^-1.2 + 17,...\n          ' +
                              '-3.2*x(2).^2.2', '0,...\n          -1'])
        self.assertEqual(DCeq, ['-1,...\n            0'])

TESTS = [TestMathModels]


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
