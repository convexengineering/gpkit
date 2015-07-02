"""Tests for model library (gpkit/models/)"""
import unittest
from gpkit import Variable


class TestMath(unittest.TestCase):
    """TestCase for math models"""

    def test_te_exp_minus1(self):
        """Test Taylor expansion of e^x - 1"""
        from gpkit.models.math import te_exp_minus1
        x = Variable('x')
        self.assertEqual(te_exp_minus1(x, 1), x)
        self.assertEqual(te_exp_minus1(x, 3), x + x**2/2. + x**3/6.)
        self.assertRaises(ValueError, te_exp_minus1, x, 0)


TESTS = [TestMath]


if __name__ == '__main__':
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
