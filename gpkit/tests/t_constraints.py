"""Unit tests for Constraint, MonomialEquality and SignomialInequality"""
import unittest
from gpkit import Variable, SignomialsEnabled, Posynomial, VectorVariable
from gpkit.nomials import SignomialInequality, PosynomialInequality
from gpkit.nomials import MonomialEquality
from gpkit import Model, ConstraintSet
from gpkit.constraints.tight import Tight
from gpkit.constraints.loose import Loose
from gpkit.tests.helpers import run_tests
from gpkit.exceptions import InvalidGPConstraint
from gpkit.constraints.relax import ConstraintsRelaxed
from gpkit.constraints.bounded import Bounded
import gpkit


class TestConstraint(unittest.TestCase):
    """Tests for Constraint class"""

    def test_uninited_element(self):
        x = Variable("x")

        class SelfPass(Model):
            "A model which contains itself!"
            def setup(self):
                ConstraintSet([self, x <= 1])

        self.assertRaises(ValueError, SelfPass)

    def test_bad_elements(self):
        x = Variable("x")
        with self.assertRaises(ValueError):
            _ = Model(x, [x == "A"])
        with self.assertRaises(ValueError):
            _ = Model(x, [x >= 1, x == "A"])
        with self.assertRaises(ValueError):
            _ = Model(x, [x >= 1, x == "A", x >= 1, ])
        with self.assertRaises(ValueError):
            _ = Model(x, [x == "A", x >= 1])
        v = VectorVariable(2, "v")
        with self.assertRaises(ValueError):
            _ = Model(x, [v == "A"])
        with self.assertRaises(ValueError):
            _ = Model(x, [v <= ["A", "B"]])
        with self.assertRaises(ValueError):
            _ = Model(x, [v >= ["A", "B"]])

    def test_evalfn(self):
        x = Variable("x")
        x2 = Variable("x^2", evalfn=lambda solv: solv[x]**2)
        m = Model(x, [x >= 2])
        m.unique_varkeys = set([x2.key])
        sol = m.solve(verbosity=0)
        self.assertAlmostEqual(sol(x2), sol(x)**2)

    def test_equality_relaxation(self):
        x = Variable("x")
        m = Model(x, [x == 3, x == 4])
        rc = ConstraintsRelaxed(m)
        m2 = Model(rc.relaxvars.prod() * x**0.01, rc)
        self.assertAlmostEqual(m2.solve(verbosity=0)(x), 3, 5)

    def test_constraintget(self):
        x = Variable("x")
        x_ = Variable("x", models=["_"])
        xv = VectorVariable(2, "x")
        xv_ = VectorVariable(2, "x", models=["_"])
        self.assertEqual(Model(x, [x >= 1])["x"], x)
        with self.assertRaises(ValueError):
            _ = Model(x, [x >= 1, x_ >= 1])["x"]
        with self.assertRaises(ValueError):
            _ = Model(x, [x >= 1, xv >= 1])["x"]
        self.assertTrue(all(Model(xv.prod(), [xv >= 1])["x"] == xv))
        with self.assertRaises(ValueError):
            _ = Model(xv.prod(), [xv >= 1, xv_ >= 1])["x"]
        with self.assertRaises(ValueError):
            _ = Model(xv.prod(), [xv >= 1, x_ >= 1])["x"]

    def test_additive_scalar(self):
        """Make sure additive scalars simplify properly"""
        x = Variable('x')
        c1 = 1 >= 10*x
        c2 = 1 >= 5*x + 0.5
        self.assertEqual(type(c1), PosynomialInequality)
        self.assertEqual(type(c2), PosynomialInequality)
        c1posy, = c1.as_posyslt1()
        c2posy, = c2.as_posyslt1()
        self.assertEqual(c1posy.hmap, c2posy.hmap)

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
        self.assertEqual(type((1 >= x).latex()), str)

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
        x = Variable("x", "ft")
        y = Variable("y")
        if gpkit.units:
            self.assertRaises(ValueError, MonomialEquality, x, "=", y)
            self.assertRaises(ValueError, MonomialEquality, y, "=", x)

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

    def test_united_dimensionless(self):
        "Check dimensionless unit-ed variables work"
        x = Variable('x')
        y = Variable('y', 'hr/day')
        c = MonomialEquality(x, "=", y)
        self.assertTrue(isinstance(c, MonomialEquality))


class TestSignomialInequality(unittest.TestCase):
    """Test Signomial constraints"""
    def test_becomes_posy_sensitivities(self):
        # pylint: disable=invalid-name
        # model from #1165
        ujet = Variable("ujet")
        PK = Variable("PK")

        # Constants
        Dp = Variable("Dp", 0.662)
        fBLI = Variable("fBLI", 0.4)
        fsurf = Variable("fsurf", 0.836)
        mdot = Variable("mdot", 1/0.7376)

        with SignomialsEnabled():
            m = Model(PK, [mdot*ujet + fBLI*Dp >= 1,
                           PK >= 0.5*mdot*ujet*(2 + ujet) + fBLI*fsurf*Dp])
        var_senss = m.solve(verbosity=0)["sensitivities"]["constants"]
        self.assertAlmostEqual(var_senss[Dp], -0.16, 2)
        self.assertAlmostEqual(var_senss[fBLI], -0.16, 2)
        self.assertAlmostEqual(var_senss[fsurf], 0.19, 2)
        self.assertAlmostEqual(var_senss[mdot], -0.17, 2)

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
        with self.assertRaises(InvalidGPConstraint):
            _ = sc.as_posyslt1()


class TestLoose(unittest.TestCase):
    """Test loose constraint set"""

    def test_posyconstr_in_gp(self):
        """Tests loose constraint set with solve()"""
        x = Variable('x')
        x_min = Variable('x_{min}', 2)
        m = Model(x, [Loose([x >= x_min], raiseerror=True),
                      x >= 1])
        with self.assertRaises(ValueError):
            m.solve(verbosity=0)
        m.substitutions[x_min] = 0.5
        self.assertAlmostEqual(m.solve(verbosity=0)["cost"], 1)

    def test_posyconstr_in_sp(self):
        x = Variable('x')
        y = Variable('y')
        x_min = Variable('x_min', 1)
        y_min = Variable('y_min', 2)
        with SignomialsEnabled():
            sig_constraint = (x + y >= 3.5)
        m = Model(x*y, [Loose([x >= y], raiseerror=True),
                        x >= x_min, y >= y_min, sig_constraint])
        with self.assertRaises(ValueError):
            m.localsolve(verbosity=0)
        m.substitutions[x_min] = 2
        m.substitutions[y_min] = 1
        self.assertAlmostEqual(m.localsolve(verbosity=0)["cost"], 2.5, 5)


class TestTight(unittest.TestCase):
    """Test tight constraint set"""

    def test_posyconstr_in_gp(self):
        """Tests tight constraint set with solve()"""
        x = Variable('x')
        x_min = Variable('x_{min}', 2)
        m = Model(x, [Tight([x >= 1], raiseerror=True),
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
        m = Model(x*y, [Tight([x >= y], raiseerror=True),
                        x >= 2, y >= 1, sig_constraint])
        with self.assertRaises(ValueError):
            m.localsolve(verbosity=0)
        m.pop(1)
        self.assertAlmostEqual(m.localsolve(verbosity=0)["cost"], 1, 5)

    def test_sigconstr_in_sp(self):
        """Tests tight constraint set with localsolve()"""
        x = Variable('x')
        y = Variable('y')
        x_min = Variable('x_{min}', 2)
        y_max = Variable('y_{max}', 0.5)
        with SignomialsEnabled():
            m = Model(x, [Tight([x + y >= 1], raiseerror=True),
                          x >= x_min,
                          y <= y_max])
        with self.assertRaises(ValueError):
            m.localsolve(verbosity=0)
        m.substitutions[x_min] = 0.5
        self.assertAlmostEqual(m.localsolve(verbosity=0)["cost"], 0.5)


class TestBounded(unittest.TestCase):
    """Test bounded constraint set"""

    def test_substitution_issue905(self):
        x = Variable("x")
        y = Variable("y")
        m = Model(x, [x >= y], {"y": 1})
        bm = Model(m.cost, Bounded(m))
        sol = bm.solve(verbosity=0)
        self.assertAlmostEqual(sol["cost"], 1.0)


TESTS = [TestConstraint, TestMonomialEquality, TestSignomialInequality,
         TestTight, TestLoose, TestBounded]

if __name__ == '__main__':
    run_tests(TESTS)
