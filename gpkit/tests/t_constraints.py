"""Unit tests for classes Constraint and MonoEQConstraint"""
import unittest
from gpkit import Variable
from gpkit.nomials import Constraint, MonoEQConstraint, Posynomial


class TestConstraint(unittest.TestCase):
    """Tests for Constraint class"""

    def test_additive_scalar(self):
        """Make sure additive scalars simplify properly"""
        x = Variable('x')
        c1 = 1 >= 10*x
        c2 = 1 >= 5*x + 0.5
        self.assertEqual(type(c1), Constraint)
        self.assertEqual(type(c2), Constraint)
        self.assertEqual(c1.cs, c2.cs)
        self.assertEqual(c1.exps, c2.exps)

    def test_additive_scalar_gt1(self):
        """1 can't be greater than (1 + something positive)"""
        x = Variable('x')

        def constr():
            """method that should raise a ValueError"""
            return (1 >= 5*x + 1.1)
        self.assertRaises(ValueError, constr)

    # TODO need test for __init__ and perhaps left/right behavior


class TestMonoEQConstraint(unittest.TestCase):
    """Test monomial equality constraint class"""

    def test_init(self):
        """Make sure initialization functions as expected"""
        x = Variable('x')
        y = Variable('y')
        mec = (x == y**2)
        # test inheritance tree
        self.assertTrue(isinstance(mec, MonoEQConstraint))
        self.assertTrue(isinstance(mec, Constraint))
        # seems like mec should also be Monomial (not just Posynomial),
        # but that fails. TODO change next line to Monomial.
        self.assertTrue(isinstance(mec, Posynomial))
        mono = y**2/x
        self.assertTrue(mec == mono or mec == 1/mono)
        # standard initialization
        mec2 = MonoEQConstraint(x, y**2)
        self.assertTrue(mec2 == mono or mec2 == 1/mono)
        self.assertTrue(mec2 == mec or mec2 == 1/mec)
        # try to initialize a Posynomial Equality constraint
        self.assertRaises(TypeError, MonoEQConstraint, x*y, x + y)


TESTS = [TestConstraint, TestMonoEQConstraint]

if __name__ == '__main__':
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
