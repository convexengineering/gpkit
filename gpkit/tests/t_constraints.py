"""Unit tests for Constraint, MonomialEquality and SignomialInequality"""
import unittest
from gpkit import Variable, SignomialsEnabled, Posynomial, VectorVariable
from gpkit.nomials import SignomialInequality, PosynomialInequality
from gpkit.nomials import MonomialEquality
from gpkit import LinkedConstraintSet, Model
from gpkit.constraints.tight import TightConstraintSet
from gpkit.tests.helpers import run_tests


class TestConstraint(unittest.TestCase):
    """Tests for Constraint class"""

    def test_link_conflict(self):
        "Check that substitution conflicts are flagged during linking."
        x_fx1 = Variable("x", 1, models=["fixed1"])
        x_fx1b = Variable("x", 1, models=["fixed1b"])
        x_free = Variable("x", models=["free"])
        x_fx2 = Variable("x", 2, models=["fixed2"])
        lc = LinkedConstraintSet([x_fx1 >= 1, x_fx1b >= 1])
        self.assertEqual(lc.substitutions["x"], 1)
        lc = LinkedConstraintSet([x_fx1 >= 1, x_free >= 1])
        self.assertEqual(lc.substitutions["x"], 1)
        self.assertRaises(ValueError,
                          LinkedConstraintSet, [x_fx1 >= 1, x_fx2 >= 1])
        vecx_free = VectorVariable(3, "x", models=["free"])
        vecx_fixed = VectorVariable(3, "x", [1, 2, 3], models=["fixed"])
        lc = LinkedConstraintSet([vecx_free >= 1, vecx_fixed >= 1])
        self.assertEqual(lc.substitutions["x"].tolist(), [1, 2, 3])

    def test_additive_scalar(self):
        """Make sure additive scalars simplify properly"""
        x = Variable('x')
        c1 = 1 >= 10*x
        c2 = 1 >= 5*x + 0.5
        self.assertEqual(type(c1), PosynomialInequality)
        self.assertEqual(type(c2), PosynomialInequality)
        c1posy, = c1.as_posyslt1()
        c2posy, = c2.as_posyslt1()
        self.assertEqual(c1posy.cs, c2posy.cs)
        self.assertEqual(c1posy.exps, c2posy.exps)

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
        c = PosynomialInequality(x, ">=", y**2)
        self.assertEqual(c.as_posyslt1(), [y**2/x])
        self.assertEqual(c.left, x)
        self.assertEqual(c.right, y**2)
        self.assertTrue(">=" in str(c))
        c = PosynomialInequality(x, "<=", y**2)
        self.assertEqual(c.as_posyslt1(), [x/y**2])
        self.assertEqual(c.left, x)
        self.assertEqual(c.right, y**2)
        self.assertTrue("<=" in str(c))

    def test_oper_overload(self):
        """Test Constraint initialization by operator overloading"""
        x = Variable('x')
        y = Variable('y')
        c = (y >= 1 + x**2)
        self.assertEqual(c.as_posyslt1(), [1/y + x**2/y])
        self.assertEqual(c.left, y)
        self.assertEqual(c.right, 1 + x**2)
        self.assertTrue(">=" in str(c))
        # same constraint, switched operator direction
        c2 = (1 + x**2 <= y)  # same as c
        self.assertEqual(c2.as_posyslt1(), c.as_posyslt1())


class TestMonomialEquality(unittest.TestCase):
    """Test monomial equality constraint class"""

    def test_init(self):
        """Test initialization via both operator overloading and __init__"""
        x = Variable('x')
        y = Variable('y')
        mono = y**2/x
        # operator overloading
        mec = (x == y**2)
        # __init__
        mec2 = MonomialEquality(x, "=", y**2)
        self.assertTrue(mono in mec.as_posyslt1())
        self.assertTrue(mono in mec2.as_posyslt1())

    def test_inheritance(self):
        """Make sure MonomialEquality inherits from the right things"""
        F = Variable('F')
        m = Variable('m')
        a = Variable('a')
        mec = (F == m*a)
        self.assertTrue(isinstance(mec, MonomialEquality))

    def test_non_monomial(self):
        """Try to initialize a MonomialEquality with non-monomial args"""
        x = Variable('x')
        y = Variable('y')

        def constr():
            """method that should raise a TypeError"""
            MonomialEquality(x*y, "=", x+y)
        self.assertRaises(TypeError, constr)

    def test_str(self):
        "Test that MonomialEquality.__str__ returns a string"
        x = Variable('x')
        y = Variable('y')
        mec = (x == y)
        self.assertEqual(type(str(mec)), str)


class TestSignomialInequality(unittest.TestCase):
    """Test Signomial constraints"""
    def test_init(self):
        "Test initialization and types"
        D = Variable('D', units="N")
        x1, x2, x3 = (Variable("x_%s" % i, units="N") for i in range(3))
        with SignomialsEnabled():
            sc = (D >= x1 + x2 - x3)
        self.assertTrue(isinstance(sc, SignomialInequality))
        self.assertFalse(isinstance(sc, Posynomial))

    def test_posyslt1(self):
        x = Variable("x")
        y = Variable("y")
        with SignomialsEnabled():
            sc = (x + y >= x*y)
        # make sure that the error type doesn't change on our users
        with self.assertRaises(TypeError):
            _ = sc.as_posyslt1()


class TestTightConstraintSet(unittest.TestCase):
    """Test tight constraint set"""

    def test_posyconstr_in_gp(self):
        """Tests tight constraint set with solve()"""
        x = Variable('x')
        x_min = Variable('x_{min}', 2)
        m = Model(x, [TightConstraintSet([x >= 1], raiseerror=True),
                      x >= x_min])
        with self.assertRaises(ValueError):
            m.solve(verbosity=0)
        m.substitutions[x_min] = 0.5
        self.assertAlmostEqual(m.solve(verbosity=0)["cost"], 1)

    def test_posyconstr_in_sp(self):
        x = Variable('x')
        y = Variable('y')
        with SignomialsEnabled():
            sig_constraint = (x + y >= 0.1)
        m = Model(x, [TightConstraintSet([x >= y], raiseerror=True),
                      x >= 2, y >= 1, sig_constraint])
        with self.assertRaises(ValueError):
            m.localsolve(verbosity=0)
        m.pop(1)
        self.assertAlmostEqual(m.localsolve(verbosity=0)["cost"], 1)

    def test_sigconstr_in_sp(self):
        """Tests tight constraint set with localsolve()"""
        x = Variable('x')
        y = Variable('y')
        x_min = Variable('x_{min}', 2)
        y_max = Variable('y_{max}', 0.5)
        with SignomialsEnabled():
            m = Model(x, [TightConstraintSet([x + y >= 1], raiseerror=True),
                          x >= x_min,
                          y <= y_max])
        with self.assertRaises(ValueError):
            m.localsolve(verbosity=0)
        m.substitutions[x_min] = 0.5
        self.assertAlmostEqual(m.localsolve(verbosity=0)["cost"], 0.5)

TESTS = [TestConstraint, TestMonomialEquality, TestSignomialInequality,
         TestTightConstraintSet]

if __name__ == '__main__':
    run_tests(TESTS)
