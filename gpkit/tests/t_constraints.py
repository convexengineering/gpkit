"Unit tests for Constraint, MonomialEquality and SignomialInequality"
import unittest
import numpy as np
from gpkit import Variable, SignomialsEnabled, Posynomial, VectorVariable
from gpkit.nomials import SignomialInequality, PosynomialInequality
from gpkit.nomials import MonomialEquality
from gpkit import Model, ConstraintSet
from gpkit.constraints.costed import CostedConstraintSet
from gpkit.constraints.tight import Tight
from gpkit.constraints.loose import Loose
from gpkit.tests.helpers import run_tests
from gpkit.exceptions import (InvalidGPConstraint, PrimalInfeasible,
                              DimensionalityError)
from gpkit.constraints.relax import (ConstraintsRelaxed, ConstantsRelaxed,
                                     ConstraintsRelaxedEqually)
from gpkit.constraints.bounded import Bounded
from gpkit.globals import NamedVariables
import gpkit


class TestConstraint(unittest.TestCase):
    "Tests for Constraint class"

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
        with self.assertRaises(TypeError):
            _ = Model(x, [v <= ["A", "B"]])
        with self.assertRaises(TypeError):
            _ = Model(x, [v >= ["A", "B"]])

    def test_evalfn(self):
        x = Variable("x")
        x2 = Variable("x^2", evalfn=lambda solv: solv[x]**2)
        m = Model(x, [x >= 2])
        m.unique_varkeys = set([x2.key])
        sol = m.solve(verbosity=0)
        self.assertAlmostEqual(sol(x2), sol(x)**2)

    def test_relax_list(self):
        x = Variable("x")
        x_max = Variable("x_max", 1)
        x_min = Variable("x_min", 2)
        constraints = [x_min <= x, x <= x_max]
        ConstraintsRelaxed(constraints)
        ConstantsRelaxed(constraints)
        ConstraintsRelaxedEqually(constraints)

    def test_relax_linked(self):
        x = Variable("x")
        x_max = Variable("x_max", 1)
        x_min = Variable("x_min", lambda c: 2*c[x_max])
        zero = Variable("zero", lambda c: 0*c[x_max])
        constraints = ConstraintSet([x_min + zero <= x, x + zero <= x_max])
        _ = ConstantsRelaxed(constraints)
        NamedVariables.reset_modelnumbers()
        include_min = ConstantsRelaxed(constraints, include_only=["x_min"])
        NamedVariables.reset_modelnumbers()
        exclude_max = ConstantsRelaxed(constraints, exclude=["x_max"])
        self.assertEqual(str(include_min), str(exclude_max))

    def test_equality_relaxation(self):
        x = Variable("x")
        m = Model(x, [x == 3, x == 4])
        rc = ConstraintsRelaxed(m)
        m2 = Model(rc.relaxvars.prod() * x**0.01, rc)
        self.assertAlmostEqual(m2.solve(verbosity=0)(x), 3, places=3)

    def test_constraintget(self):
        x = Variable("x")
        x_ = Variable("x", lineage=[("_", 0)])
        xv = VectorVariable(2, "x")
        xv_ = VectorVariable(2, "x", lineage=[("_", 0)])
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
        "Make sure additive scalars simplify properly"
        x = Variable('x')
        c1 = 1 >= 10*x
        c2 = 1 >= 5*x + 0.5
        self.assertEqual(type(c1), PosynomialInequality)
        self.assertEqual(type(c2), PosynomialInequality)
        c1hmap, = c1.as_hmapslt1({})
        c2hmap, = c2.as_hmapslt1({})
        self.assertEqual(c1hmap, c2hmap)

    def test_additive_scalar_gt1(self):
        "1 can't be greater than (1 + something positive)"
        x = Variable('x')

        def constr():
            "method that should raise a ValueError"
            return 1 >= 5*x + 1.1
        self.assertRaises(PrimalInfeasible, constr)

    def test_init(self):
        "Test Constraint __init__"
        x = Variable('x')
        y = Variable('y')
        c = PosynomialInequality(x, ">=", y**2)
        self.assertEqual(c.as_hmapslt1({}), [(y**2/x).hmap])
        self.assertEqual(c.left, x)
        self.assertEqual(c.right, y**2)
        self.assertTrue(">=" in str(c))
        c = PosynomialInequality(x, "<=", y**2)
        self.assertEqual(c.as_hmapslt1({}), [(x/y**2).hmap])
        self.assertEqual(c.left, x)
        self.assertEqual(c.right, y**2)
        self.assertTrue("<=" in str(c))
        self.assertEqual(type((1 >= x).latex()), str)

    def test_oper_overload(self):
        "Test Constraint initialization by operator overloading"
        x = Variable('x')
        y = Variable('y')
        c = (y >= 1 + x**2)
        self.assertEqual(c.as_hmapslt1({}), [(1/y + x**2/y).hmap])
        self.assertEqual(c.left, y)
        self.assertEqual(c.right, 1 + x**2)
        self.assertTrue(">=" in str(c))
        # same constraint, switched operator direction
        c2 = (1 + x**2 <= y)  # same as c
        self.assertEqual(c2.as_hmapslt1({}), c.as_hmapslt1({}))

    def test_sub_tol(self):
        " Test PosyIneq feasibility tolerance under substitutions"
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        PosynomialInequality.feastol = 1e-5
        m = Model(z, [x == z, x >= y], {x: 1, y: 1.0001})
        self.assertRaises(PrimalInfeasible, m.solve, verbosity=0)
        PosynomialInequality.feastol = 1e-3
        self.assertEqual(m.substitutions('x'), m.solve(verbosity=0)('x'))

class TestCostedConstraint(unittest.TestCase):
    "Tests for Costed Constraint class"

    def test_vector_cost(self):
        x = VectorVariable(2, "x")
        self.assertRaises(ValueError, CostedConstraintSet, x, [])
        _ = CostedConstraintSet(np.array(x[0]), [])

class TestMonomialEquality(unittest.TestCase):
    "Test monomial equality constraint class"

    def test_init(self):
        "Test initialization via both operator overloading and __init__"
        x = Variable('x')
        y = Variable('y')
        mono = y**2/x
        # operator overloading
        mec = (x == y**2)
        # __init__
        mec2 = MonomialEquality(x, y**2)
        self.assertTrue(mono.hmap in mec.as_hmapslt1({}))
        self.assertTrue(mono.hmap in mec2.as_hmapslt1({}))
        x = Variable("x", "ft")
        y = Variable("y")
        if gpkit.units:
            self.assertRaises(DimensionalityError, MonomialEquality, x, y)
            self.assertRaises(DimensionalityError, MonomialEquality, y, x)

    def test_vector(self):
        "Monomial Equalities with VectorVariables"
        x = VectorVariable(3, "x")
        self.assertFalse(x == 3)
        self.assertTrue(x == x)  # pylint: disable=comparison-with-itself

    def test_inheritance(self):
        "Make sure MonomialEquality inherits from the right things"
        F = Variable('F')
        m = Variable('m')
        a = Variable('a')
        mec = (F == m*a)
        self.assertTrue(isinstance(mec, MonomialEquality))

    def test_non_monomial(self):
        "Try to initialize a MonomialEquality with non-monomial args"
        x = Variable('x')
        y = Variable('y')

        def constr():
            "method that should raise a TypeError"
            MonomialEquality(x*y, x+y)
        self.assertRaises(TypeError, constr)

    def test_str(self):
        "Test that MonomialEquality.__str__ returns a string"
        x = Variable('x')
        y = Variable('y')
        mec = (x == y)
        self.assertEqual(type(mec.str_without()), str)

    def test_united_dimensionless(self):
        "Check dimensionless unit-ed variables work"
        x = Variable('x')
        y = Variable('y', 'hr/day')
        c = MonomialEquality(x, y)
        self.assertTrue(isinstance(c, MonomialEquality))


class TestSignomialInequality(unittest.TestCase):
    "Test Signomial constraints"

    def test_becomes_posy_sensitivities(self):
        # pylint: disable=invalid-name
        # model from #1165
        ujet = Variable("ujet")
        PK = Variable("PK")
        Dp = Variable("Dp", 0.662)
        fBLI = Variable("fBLI", 0.4)
        fsurf = Variable("fsurf", 0.836)
        mdot = Variable("mdot", 1/0.7376)
        with SignomialsEnabled():
            m = Model(PK, [mdot*ujet + fBLI*Dp >= 1,
                           PK >= 0.5*mdot*ujet*(2 + ujet) + fBLI*fsurf*Dp])
        var_senss = m.solve(verbosity=0)["sensitivities"]["variables"]
        self.assertAlmostEqual(var_senss[Dp], -0.16, 2)
        self.assertAlmostEqual(var_senss[fBLI], -0.16, 2)
        self.assertAlmostEqual(var_senss[fsurf], 0.19, 2)
        self.assertAlmostEqual(var_senss[mdot], -0.17, 2)

        # Linked variable
        Dp = Variable("Dp", 0.662)
        mDp = Variable("-Dp", lambda c: -c[Dp])
        fBLI = Variable("fBLI", 0.4)
        fsurf = Variable("fsurf", 0.836)
        mdot = Variable("mdot", 1/0.7376)
        m = Model(PK, [mdot*ujet >= 1 + fBLI*mDp,
                       PK >= 0.5*mdot*ujet*(2 + ujet) + fBLI*fsurf*Dp])
        var_senss = m.solve(verbosity=0)["sensitivities"]["variables"]
        self.assertAlmostEqual(var_senss[Dp], -0.16, 2)
        self.assertAlmostEqual(var_senss[fBLI], -0.16, 2)
        self.assertAlmostEqual(var_senss[fsurf], 0.19, 2)
        self.assertAlmostEqual(var_senss[mdot], -0.17, 2)

        # fixed negative variable
        Dp = Variable("Dp", 0.662)
        mDp = Variable("-Dp", -0.662)
        fBLI = Variable("fBLI", 0.4)
        fsurf = Variable("fsurf", 0.836)
        mdot = Variable("mdot", 1/0.7376)
        m = Model(PK, [mdot*ujet >= 1 + fBLI*mDp,
                       PK >= 0.5*mdot*ujet*(2 + ujet) + fBLI*fsurf*Dp])
        var_senss = m.solve(verbosity=0)["sensitivities"]["variables"]
        self.assertAlmostEqual(var_senss[Dp] + var_senss[mDp], -0.16, 2)
        self.assertAlmostEqual(var_senss[fBLI], -0.16, 2)
        self.assertAlmostEqual(var_senss[fsurf], 0.19, 2)
        self.assertAlmostEqual(var_senss[mdot], -0.17, 2)

    def test_init(self):
        "Test initialization and types"
        D = Variable('D', units="N")
        x1, x2, x3 = (Variable("x_%s" % i, units="N") for i in range(3))
        with self.assertRaises(TypeError):
            sc = (D >= x1 + x2 - x3)
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
            _ = sc.as_hmapslt1({})


class TestLoose(unittest.TestCase):
    "Test loose constraint set"

    def test_raiseerror(self):
        x = Variable('x')
        x_min = Variable('x_{min}', 2)
        m = Model(x, [Loose([x >= x_min]),
                      x >= 1])
        Loose.raiseerror = True
        self.assertRaises(RuntimeWarning, m.solve, verbosity=0)
        Loose.raiseerror = False

    def test_posyconstr_in_gp(self):
        "Tests loose constraint set with solve()"
        x = Variable('x')
        x_min = Variable('x_{min}', 2)
        m = Model(x, [Loose([x >= x_min]),
                      x >= 1])
        sol = m.solve(verbosity=0)
        self.assertIs(
            sol["warnings"]["Unexpectedly Tight Constraints"][0][1], m[0][0])
        self.assertAlmostEqual(m[0][0].relax_sensitivity, +1, 3)
        m.substitutions[x_min] = 0.5
        self.assertAlmostEqual(m.solve(verbosity=0)["cost"], 1)

    def test_posyconstr_in_sp(self):
        x = Variable('x')
        y = Variable('y')
        x_min = Variable('x_min', 1)
        y_min = Variable('y_min', 2)
        with SignomialsEnabled():
            sig_constraint = (x + y >= 3.5)
        m = Model(x*y, [Loose([x >= y]),
                        x >= x_min, y >= y_min, sig_constraint])
        sol = m.localsolve(verbosity=0)
        self.assertIs(
            sol["warnings"]["Unexpectedly Tight Constraints"][0][1], m[0][0])
        self.assertAlmostEqual(m[0][0].relax_sensitivity, +1, 3)
        m.substitutions[x_min] = 2
        m.substitutions[y_min] = 1
        self.assertAlmostEqual(m.localsolve(verbosity=0)["cost"], 2.5, 5)


class TestTight(unittest.TestCase):
    "Test tight constraint set"

    def test_posyconstr_in_gp(self):
        "Tests tight constraint set with solve()"
        x = Variable('x')
        x_min = Variable('x_{min}', 2)
        m = Model(x, [Tight([x >= 1]),
                      x >= x_min])
        sol = m.solve(verbosity=0)
        self.assertIs(
            sol["warnings"]["Unexpectedly Loose Constraints"][0][1], m[0][0])
        self.assertAlmostEqual(m[0][0].rel_diff, 1, 3)
        m.substitutions[x_min] = 0.5
        self.assertAlmostEqual(m.solve(verbosity=0)["cost"], 1)

    def test_posyconstr_in_sp(self):
        x = Variable('x')
        y = Variable('y')
        with SignomialsEnabled():
            sig_constraint = (x + y >= 0.1)
        m = Model(x*y, [Tight([x >= y]),
                        x >= 2, y >= 1, sig_constraint])
        sol = m.localsolve(verbosity=0)
        self.assertIs(
            sol["warnings"]["Unexpectedly Loose Constraints"][0][1], m[0][0])
        self.assertAlmostEqual(m[0][0].rel_diff, 1, 3)
        m.pop(1)
        self.assertAlmostEqual(m.localsolve(verbosity=0)["cost"], 1, 5)

    def test_sigconstr_in_sp(self):
        "Tests tight constraint set with localsolve()"
        x = Variable('x')
        y = Variable('y')
        x_min = Variable('x_{min}', 2)
        y_max = Variable('y_{max}', 0.5)
        with SignomialsEnabled():
            m = Model(x, [Tight([x + y >= 1]),
                          x >= x_min,
                          y <= y_max])
        sol = m.localsolve(verbosity=0)
        self.assertIs(
            sol["warnings"]["Unexpectedly Loose Constraints"][0][1], m[0][0])
        self.assertGreater(m[0][0].rel_diff, 0.5)
        m.substitutions[x_min] = 0.5
        self.assertAlmostEqual(m.localsolve(verbosity=0)["cost"], 0.5, 5)


class TestBounded(unittest.TestCase):
    "Test bounded constraint set"

    def test_substitution_issue905(self):
        x = Variable("x")
        y = Variable("y")
        m = Model(x, [x >= y], {"y": 1})
        bm = Model(m.cost, Bounded(m))
        sol = bm.solve(verbosity=0)
        self.assertAlmostEqual(sol["cost"], 1.0)
        bm = Model(m.cost, Bounded(m, lower=1e-10))
        sol = bm.solve(verbosity=0)
        self.assertAlmostEqual(sol["cost"], 1.0)
        bm = Model(m.cost, Bounded(m, upper=1e10))
        sol = bm.solve(verbosity=0)
        self.assertAlmostEqual(sol["cost"], 1.0)

TESTS = [TestConstraint, TestMonomialEquality, TestSignomialInequality,
         TestTight, TestLoose, TestBounded, TestCostedConstraint]

if __name__ == "__main__":  # pragma: no cover
    run_tests(TESTS)
