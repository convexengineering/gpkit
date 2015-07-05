"""Tests for model library (gpkit/models/)"""
import unittest
from gpkit import Variable, VectorVariable


class TestMath(unittest.TestCase):
    """TestCase for math models"""

    def test_te_exp_minus1(self):
        """Test Taylor expansion of e^x - 1"""
        from gpkit.models.math import te_exp_minus1
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


TESTS = [TestMath]


if __name__ == '__main__':
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
