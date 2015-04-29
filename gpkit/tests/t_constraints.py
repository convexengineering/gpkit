"""Unit tests for classes Constraint and MonoEQConstraint"""
import unittest
from gpkit import Variable
from gpkit.nomials import Constraint


class TestConstraint(unittest.TestCase):
    """Tests for Constraint class"""

    def test_additive_scalar(self):
        x = Variable('x')
        c1 = 1 >= 10*x
        c2 = 1 >= 5*x + 0.5
        self.assertEqual(type(c1), Constraint)
        self.assertEqual(type(c2), Constraint)
        self.assertEqual(c1.cs, c2.cs)
        self.assertEqual(c1.exps, c2.exps)

    def test_additive_scalar_gt1(self):
        x = Variable('x')

        def constr():
            return (1 >= 5*x + 1.1)
        self.assertRaises(ValueError, constr)


class TestMonoEQConstraint(unittest.TestCase):
    """Test monomial equality constraint class"""

    def test_placeholder(self):
        pass


TESTS = [TestConstraint, TestMonoEQConstraint]

if __name__ == '__main__':
    from gpkit.tests.run_tests import run_tests
    run_tests(TESTS)
