"""Unit tests for Constraint, MonoEQConstraint and SignomialConstraint"""
import unittest
from gpkit import Variable, SignomialsEnabled
from gpkit.nomials import Posynomial
from gpkit.nomials import Constraint, MonoEQConstraint, SignomialConstraint


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

    def test_init(self):
        """Test Constraint __init__"""
        x = Variable('x')
        y = Variable('y')
        # default assumes >= operator
        c = Constraint(x, y**2)
        self.assertEqual(c, y**2/x)
        self.assertEqual(c.left, x)
        self.assertEqual(c.right, y**2)
        self.assertTrue(">=" in str(c))
        # now force <= operator
        c = Constraint(x, y**2, oper_ge=False)
        self.assertEqual(c, x/y**2)
        self.assertEqual(c.left, x)
        self.assertEqual(c.right, y**2)
        self.assertTrue("<=" in str(c))

    def test_oper_overload(self):
        """Test Constraint initialization by operator overloading"""
        x = Variable('x')
        y = Variable('y')
        c = (y >= 1 + x**2)
        self.assertEqual(c, 1/y + x**2/y)
        self.assertEqual(c.left, y)
        self.assertEqual(c.right, 1 + x**2)
        self.assertTrue(">=" in str(c))
        # same constraint, switched operator direction
        c2 = (1 + x**2 <= y)  # same as c
        self.assertEqual(c2, c)


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


class TestSignomialConstraint(unittest.TestCase):
    """Test Signomial constraints"""
    def test_init(self):
        "Test initialization and types"
        D = Variable('D', units="N")
        x1, x2, x3 = (Variable("x_%s" % i, units="N") for i in range(3))
        with SignomialsEnabled():
            sc = (D >= x1 + x2 - x3)
        self.assertTrue(isinstance(sc, SignomialConstraint))
        self.assertFalse(isinstance(sc, Posynomial))


TESTS = [TestConstraint, TestMonoEQConstraint, TestSignomialConstraint]

if __name__ == '__main__':
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
