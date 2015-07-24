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
        """Test initialization via both operator overloading and __init__"""
        x = Variable('x')
        y = Variable('y')
        mono = y**2/x
        # operator overloading
        mec = (x == y**2)
        # __init__
        mec2 = MonoEQConstraint(x, y**2)
        self.assertTrue(mec2 == mono or mec2 == 1/mono)
        self.assertTrue(mec2 == mec or mec2 == 1/mec)
        self.assertTrue(mec == mono or mec == 1/mono)
        self.assertTrue(mec2 == mono or mec2 == 1/mono)

    def test_inheritance(self):
        """Make sure MonoEQConstraint inherits from the right things"""
        F = Variable('F')
        m = Variable('m')
        a = Variable('a')
        mec = (F == m*a)
        self.assertTrue(isinstance(mec, MonoEQConstraint))
        self.assertTrue(isinstance(mec, Constraint))
        # seems like mec should also be Monomial (not just Posynomial),
        # but that fails. TODO change next line to Monomial.
        self.assertTrue(isinstance(mec, Posynomial))

    def test_non_monomial(self):
        """Try to initialize a MonoEQConstraint with non-monomial args"""
        x = Variable('x')
        y = Variable('y')
        # try to initialize a Posynomial Equality constraint
        self.assertRaises(TypeError, MonoEQConstraint, x*y, x + y)


TESTS = [TestConstraint, TestMonoEQConstraint]

if __name__ == '__main__':
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
