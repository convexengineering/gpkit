"""Test substitution capability across gpkit"""
import pickle
import unittest
import numpy as np
import numpy.testing as npt
from ad import adnumber, ADV
import gpkit
from gpkit import SignomialsEnabled, NamedVariables
from gpkit import Variable, VectorVariable, Model, Signomial
from gpkit.small_scripts import mag
from gpkit.tests.helpers import run_tests
from gpkit.exceptions import UnboundedGP, DimensionalityError

# pylint: disable=invalid-name,attribute-defined-outside-init,unused-variable


class TestNomialSubs(unittest.TestCase):
    """Test substitution for nomial-family objects"""

    def test_vectorized_linked(self):
        class VectorLinked(Model):
            "simple vectorized link"
            def setup(self):
                self.y = y = Variable("y", 1)

                def vectorlink(c):
                    "linked vector function"
                    if isinstance(c[y], ADV):
                        return np.array(c[y])+adnumber([1, 2, 3])
                    return c[y]+np.array([1, 2, 3])
                self.x = x = VectorVariable(3, "x", vectorlink)
        m = VectorLinked()
        self.assertEqual(m.substitutions[m.x[0].key](m.substitutions), 2)
        self.assertEqual(m.gp().substitutions[m.x[0].key], 2)
        self.assertEqual(m.gp().substitutions[m.x[1].key], 3)
        self.assertEqual(m.gp().substitutions[m.x[2].key], 4)

    def test_numeric(self):
        """Basic substitution of numeric value"""
        x = Variable("x")
        p = x**2
        self.assertEqual(p.sub({x: 3}), 9)
        self.assertEqual(p.sub({x.key: 3}), 9)
        self.assertEqual(p.sub({"x": 3}), 9)

    def test_dimensionless_units(self):
        x = Variable('x', 3, 'ft')
        y = Variable('y', 1, 'm')
        if x.units is not None:
            # units are enabled
            self.assertAlmostEqual((x/y).value, 0.9144)

    def test_vector(self):
        x = Variable("x")
        y = Variable("y")
        z = VectorVariable(2, "z")
        p = x*y*z
        self.assertTrue(all(p.sub({x: 1, "y": 2}) == 2*z))
        self.assertTrue(all(p.sub({x: 1, y: 2, "z": [1, 2]}) ==
                            z.sub({z: [2, 4]})))
        self.assertRaises(ValueError, z.sub, {z: [1, 2, 3]})

        xvec = VectorVariable(3, "x", "m")
        xs = xvec[:2].sum()
        for x_ in ["x", xvec]:
            self.assertAlmostEqual(mag(xs.sub({x_: [1, 2, 3]}).c), 3.0)

    def test_variable(self):
        """Test special single-argument substitution for Variable"""
        x = Variable('x')
        y = Variable('y')
        m = x*y**2
        self.assertEqual(x.sub(3), 3)
        # make sure x was not mutated
        self.assertEqual(x, Variable('x'))
        self.assertNotEqual(x.sub(3), Variable('x'))
        # also make sure the old way works
        self.assertEqual(x.sub({x: 3}), 3)
        # and for vectors
        xvec = VectorVariable(3, 'x')
        self.assertEqual(xvec[1].sub(3), 3)

    def test_signomial(self):
        """Test Signomial substitution"""
        D = Variable('D', units="N")
        x = Variable('x', units="N")
        y = Variable('y', units="N")
        a = Variable('a')
        with SignomialsEnabled():
            sc = (a*x + (1 - a)*y - D)
            subbed = sc.sub({a: 0.1})
            self.assertEqual(subbed, 0.1*x + 0.9*y - D)
            self.assertTrue(isinstance(subbed, Signomial))
            subbed = sc.sub({a: 2.0})
            self.assertTrue(isinstance(subbed, Signomial))
            self.assertEqual(subbed, 2*x - y - D)
            _ = a.sub({a: -1}).value  # fix monomial assumptions


class TestModelSubs(unittest.TestCase):
    """Test substitution for Model objects"""

    def test_bad_gp_sub(self):
        x = Variable("x")
        y = Variable("y")
        m = Model(x, [y >= 1], {y: x})
        with self.assertRaises(ValueError):
            m.solve()

    def test_quantity_sub(self):
        if gpkit.units:
            x = Variable("x", 1, "cm")
            y = Variable("y", 1)
            self.assertEqual(x.sub({x: 1*gpkit.units.m}).c.magnitude, 100)
            # NOTE: uncomment the below if requiring Quantity substitutions
            # self.assertRaises(ValueError, x.sub, x, 1)
            self.assertRaises(DimensionalityError, x.sub, {x: 1*gpkit.ureg.N})
            self.assertRaises(DimensionalityError, y.sub, {y: 1*gpkit.ureg.N})
            v = gpkit.VectorVariable(3, "v", "cm")
            subbed = v.sub({v: [1, 2, 3]*gpkit.ureg.m})
            self.assertEqual([z.c.magnitude for z in subbed], [100, 200, 300])
            v = VectorVariable(1, "v", "km")
            v_min = VectorVariable(1, "v_min", "km")
            m = Model(v.prod(), [v >= v_min],
                      {v_min: [2*gpkit.units("nmi")]})
            cost = m.solve(verbosity=0)["cost"]
            self.assertAlmostEqual(cost/3.704, 1.0)
            m = Model(v.prod(), [v >= v_min],
                      {v_min: np.array([2])*gpkit.units("nmi")})
            cost = m.solve(verbosity=0)["cost"]
            self.assertAlmostEqual(cost/3.704, 1.0)

    def test_phantoms(self):
        x = Variable("x")
        x_ = Variable("x", 1, lineage=[("test", 0)])
        xv = VectorVariable(2, "x", [1, 1], lineage=[("vec", 0)])
        m = Model(x, [x >= x_, x_ == xv.prod()])
        m.solve(verbosity=0)
        with self.assertRaises(ValueError):
            _ = m.substitutions["x"]
        with self.assertRaises(KeyError):
            _ = m.substitutions["y"]
        with self.assertRaises(ValueError):
            _ = m["x"]
        self.assertIn(x, m.variables_byname("x"))
        self.assertIn(x_, m.variables_byname("x"))

    def test_persistence(self):
        x = gpkit.Variable("x")
        y = gpkit.Variable("y")
        ymax = gpkit.Variable("y_{max}", 0.1)

        with gpkit.SignomialsEnabled():
            m = gpkit.Model(x, [x >= 1-y, y <= ymax])
            m.substitutions[ymax] = 0.2
            self.assertAlmostEqual(m.localsolve(verbosity=0)["cost"], 0.8, 3)
            m = gpkit.Model(x, [x >= 1-y, y <= ymax])
            with self.assertRaises(UnboundedGP):  # from unbounded ymax
                m.localsolve(verbosity=0)
            m = gpkit.Model(x, [x >= 1-y, y <= ymax])
            m.substitutions[ymax] = 0.1
            self.assertAlmostEqual(m.localsolve(verbosity=0)["cost"], 0.9, 3)

    def test_united_sub_sweep(self):
        A = Variable("A", "USD")
        h = Variable("h", "USD/count")
        Q = Variable("Q", "count")
        Y = Variable("Y", "USD")
        m = Model(Y, [Y >= h*Q + A/Q])
        m.substitutions.update({A: 500*gpkit.units("USD"),
                                h: 35*gpkit.units("USD"),
                                Q: ("sweep", [50, 100, 500])})
        firstcost = m.solve(verbosity=0)["cost"][0]
        self.assertAlmostEqual(1760/firstcost, 1, 5)

    def test_skipfailures(self):
        x = Variable("x")
        x_min = Variable("x_{min}", [1, 2])

        m = Model(x, [x <= 1, x >= x_min])
        sol = m.solve(verbosity=0, skipsweepfailures=True)
        sol.table()
        self.assertEqual(len(sol), 1)

        with self.assertRaises(RuntimeWarning):
            sol = m.solve(verbosity=0, skipsweepfailures=False)

        m.substitutions[x_min][1][0] = 5  # so no sweeps solve
        with self.assertRaises(RuntimeWarning):
            sol = m.solve(verbosity=0, skipsweepfailures=True)

    def test_vector_sweep(self):
        """Test sweep involving VectorVariables"""
        x = Variable("x")
        x_min = Variable("x_min", 1)
        y = VectorVariable(2, "y")
        m = Model(x, [x >= y.prod()])
        m.substitutions.update({y: ('sweep', [[2, 3], [5, 7], [9, 11]])})
        a = m.solve(verbosity=0)["cost"]
        b = [6, 15, 27, 14, 35, 63, 22, 55, 99]
        self.assertTrue(all(abs(a-b)/(a+b) < 1e-7))
        x_min = Variable("x_min", 1)  # constant to check array indexing
        m = Model(x, [x >= y.prod(), x >= x_min])
        m.substitutions.update({y: ('sweep', [[2, 3], [5, 7, 11]])})
        sol = m.solve(verbosity=0)
        a = sol["cost"]
        b = [10, 15, 14, 21, 22, 33]
        self.assertTrue(all(abs(a-b)/(a+b) < 1e-7))
        self.assertEqual(sol["constants"][x_min], 1)
        for i, bi in enumerate(b):
            self.assertEqual(sol.atindex(i)["constants"][x_min], 1)
            ai = m.solution.atindex(i)["cost"]
            self.assertTrue(abs(ai-bi)/(ai+bi) < 1e-7)
        m = Model(x, [x >= y.prod()])
        m.substitutions.update({y: ('sweep', [[2, 3, 9], [5, 7, 11]])})
        self.assertRaises(ValueError, m.solve, verbosity=0)
        m.substitutions.update({y: [2, ("sweep", [3, 5])]})
        a = m.solve(verbosity=0)["cost"]
        b = [6, 10]
        self.assertTrue(all(abs(a-b)/(a+b) < 1e-7))
        # create a numpy float array, then insert a sweep element
        m.substitutions.update({y: [2, 3]})
        m.substitutions.update({y[1]: ("sweep", [3, 5])})
        a = m.solve(verbosity=0)["cost"]
        self.assertTrue(all(abs(a-b)/(a+b) < 1e-7))

    def test_calcconst(self):
        x = Variable("x", "hours")
        t_day = Variable("t_{day}", 12, "hours")
        t_night = Variable("t_{night}", lambda c: 24 - c[t_day], "hours")
        _ = pickle.dumps(t_night)
        m = Model(x, [x >= t_day, x >= t_night])
        sol = m.solve(verbosity=0)
        self.assertAlmostEqual(sol(t_night)/gpkit.ureg.hours, 12)
        m.substitutions.update({t_day: ("sweep", [6, 8, 9, 13])})
        sol = m.solve(verbosity=0)
        npt.assert_allclose(sol["sensitivities"]["variables"][t_day],
                            [-1/3, -0.5, -0.6, +1], 1e-5)
        self.assertEqual(len(sol["cost"]), 4)
        npt.assert_allclose([float(l) for l in
                             (sol(t_day) + sol(t_night))/gpkit.ureg.hours], 24)

    def test_vector_init(self):
        N = 6
        Weight = 50000
        xi_dist = 6*Weight/float(N)*(
            (np.array(range(1, N+1)) - .5/float(N))/float(N) -
            (np.array(range(1, N+1)) - .5/float(N))**2/float(N)**2)

        xi = VectorVariable(N, "xi", xi_dist, "N", "Constant Thrust per Bin")
        P = Variable("P", "N", "Total Power")
        phys_constraints = [P >= xi.sum()]
        objective = P
        eqns = phys_constraints
        m = Model(objective, eqns)
        sol = m.solve(verbosity=0)
        a, b = sol("xi"), xi_dist*gpkit.ureg.N
        self.assertTrue(all(abs(a-b)/(a+b) < 1e-7))

    # pylint: disable=too-many-locals
    def test_model_composition_units(self):
        class Above(Model):
            """A simple upper bound on x

            Lower Unbounded
            ---------------
            x
            """
            def setup(self):
                x = self.x = Variable("x", "ft")
                x_max = Variable("x_{max}", 1, "yard")
                self.cost = 1/x
                return [x <= x_max]

        class Below(Model):
            """A simple lower bound on x

            Upper Unbounded
            ---------------
            x
            """
            def setup(self):
                x = self.x = Variable("x", "m")
                x_min = Variable("x_{min}", 1, "cm")
                self.cost = x
                return [x >= x_min]

        a, b = Above(), Below()
        concatm = Model(a.cost*b.cost, [a, b])
        concat_cost = concatm.solve(verbosity=0)["cost"]
        almostequal = self.assertAlmostEqual
        yard, cm = gpkit.ureg("yard"), gpkit.ureg("cm")
        ft, meter = gpkit.ureg("ft"), gpkit.ureg("m")
        if not isinstance(a["x"].key.units, str):
            almostequal(a.solve(verbosity=0)["cost"], ft/yard, 5)
            almostequal(b.solve(verbosity=0)["cost"], cm/meter, 5)
            almostequal(cm/yard, concat_cost, 5)
        NamedVariables.reset_modelnumbers()
        a1, b1 = Above(), Below()
        self.assertEqual(a1["x"].key.lineage, (("Above", 0),))
        m = Model(a1["x"], [a1, b1, b1["x"] == a1["x"]])
        sol = m.solve(verbosity=0)
        if not isinstance(a1["x"].key.units, str):
            almostequal(sol["cost"], cm/ft, 5)
        a1, b1 = Above(), Below()
        self.assertEqual(a1["x"].key.lineage, (("Above", 1),))
        m = Model(b1["x"], [a1, b1, b1["x"] == a1["x"]])
        sol = m.solve(verbosity=0)
        if not isinstance(b1["x"].key.units, str):
            almostequal(sol["cost"], cm/meter, 5)
        self.assertIn(a1["x"], sol["variables"])
        self.assertIn(b1["x"], sol["variables"])
        self.assertNotIn(a["x"], sol["variables"])
        self.assertNotIn(b["x"], sol["variables"])

    def test_getkey(self):
        class Top(Model):
            """Some high level model

            Upper Unbounded
            ---------------
            y
            """
            def setup(self):
                y = self.y = Variable('y')
                s = Sub()
                sy = s["y"]
                self.cost = y
                return [s, y >= sy, sy >= 1]

        class Sub(Model):
            """A simple sub model

            Upper Unbounded
            ---------------
            y
            """
            def setup(self):
                y = self.y = Variable('y')
                self.cost = y
                return [y >= 2]

        sol = Top().solve(verbosity=0)
        self.assertAlmostEqual(sol['cost'], 2)

    def test_model_recursion(self):
        class Top(Model):
            """Some high level model

            Upper Unbounded
            ---------------
            x

            """
            def setup(self):
                sub = Sub()
                x = self.x = Variable("x")
                self.cost = x
                return sub, [x >= sub["y"], sub["y"] >= 1]

        class Sub(Model):
            """A simple sub model

            Upper Unbounded
            ---------------
            y

            """
            def setup(self):
                y = self.y = Variable('y')
                self.cost = y
                return [y >= 2]

        sol = Top().solve(verbosity=0)
        self.assertAlmostEqual(sol['cost'], 2)

    def test_vector_sub(self):
        x = VectorVariable(3, "x")
        y = VectorVariable(3, "y")
        ymax = VectorVariable(3, "ymax")

        with SignomialsEnabled():
            # issue1077 links to a case that failed for SPs only
            m = Model(x.prod(), [x + y >= 1, y <= ymax])

        m.substitutions["ymax"] = [0.3, 0.5, 0.8]
        m.localsolve(verbosity=0)

    def test_spsubs(self):
        x = Variable("x", 5)
        y = Variable("y", lambda c: 2*c[x])
        z = Variable("z")
        w = Variable("w")

        with SignomialsEnabled():
            cnstr = [z + w >= y*x, w <= y]

        m = Model(z, cnstr)
        m.localsolve(verbosity=0)
        self.assertTrue(m.substitutions["y"], "__call__")

class TestNomialMapSubs(unittest.TestCase):
    "Tests substitutions of nomialmaps"
    def test_monomial_sub(self):
        z = Variable("z")
        w = Variable("w")

        with self.assertRaises(ValueError):
            z.hmap.sub({z.key: w.key}, varkeys=z.varkeys)

    def test_subinplace_zero(self):
        z = Variable("z")
        w = Variable("w")

        p = 2*w + z*w + 2

        self.assertEqual(p.sub({z: -2}), 2)

TESTS = [TestNomialSubs, TestModelSubs, TestNomialMapSubs]

if __name__ == "__main__":  # pragma: no cover
    run_tests(TESTS)
