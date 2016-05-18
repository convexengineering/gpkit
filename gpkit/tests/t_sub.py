"""Test substitution capability across gpkit"""
import unittest
import numpy as np
import numpy.testing as npt
import gpkit
from gpkit import SignomialsEnabled
from gpkit import Variable, VectorVariable, Model, Signomial
from gpkit.small_scripts import mag
from gpkit.tests.helpers import run_tests


class TestNomialSubs(unittest.TestCase):
    """Test substitution for nomial-family objects"""

    def test_numeric(self):
        """Basic substitution of numeric value"""
        x = Variable("x")
        p = x**2
        self.assertEqual(p.sub(x, 3), 9)
        self.assertEqual(p.sub(x.key, 3), 9)
        self.assertEqual(p.sub("x", 3), 9)

    def test_basic(self):
        """Basic substitution, symbolic"""
        x = Variable('x')
        y = Variable('y')
        p = 1 + x**2
        q = p.sub({x: y**2})
        self.assertEqual(q, 1 + y**4)
        self.assertEqual(x.sub({x: y}), y)

    def test_string_mutation(self):
        x = Variable("x", "m")
        descr_before = list(x.exp)[0].descr
        y = x.sub("x", "y")
        descr_after = list(x.exp)[0].descr
        self.assertEqual(descr_before, descr_after)
        x_changed_descr = dict(descr_before)
        x_changed_descr["name"] = "y"
        y_descr = list(y.exp)[0].descr
        self.assertEqual(x_changed_descr["name"], y_descr["name"])
        if not isinstance(descr_before["units"], str):
            self.assertAlmostEqual(x_changed_descr["units"]/y_descr["units"],
                                   1.0)
        self.assertEqual(x.sub("x", x), x)

    def test_quantity_sub(self):
        if gpkit.units:
            x = Variable("x", 1, "cm")
            y = Variable("y", 1)
            self.assertEqual(x.sub(x, 1*gpkit.units.m).c.magnitude, 100)
            # NOTE: uncomment the below if requiring Quantity substitutions
            # self.assertRaises(ValueError, x.sub, x, 1)
            self.assertRaises(ValueError, x.sub, x, 1*gpkit.units.N)
            self.assertRaises(ValueError, y.sub, y, 1*gpkit.units.N)
            v = gpkit.VectorVariable(3, "v", "cm")
            subbed = v.sub(v, [1, 2, 3]*gpkit.units.m)
            self.assertEqual([z.c.magnitude for z in subbed], [100, 200, 300])

    def test_scalar_units(self):
        x = Variable("x", "m")
        xvk = x.key
        y = Variable("y", "km")
        yvk = y.key
        units_exist = bool(x.units)
        for x_ in ["x", xvk, x]:
            for y_ in ["y", yvk, y]:
                if not isinstance(y_, str) and units_exist:
                    expected = 1000.0
                else:
                    expected = 1.0
                self.assertAlmostEqual(expected, mag(x.sub(x_, y_).c))
        if units_exist:
            z = Variable("z", "s")
            self.assertRaises(ValueError, y.sub, y, z)

    def test_dimensionless_units(self):
        x = Variable('x', 3, 'ft')
        y = Variable('y', 1, 'm')
        if x.units is not None:
            # units are enabled
            self.assertAlmostEqual((x/y).value, 0.9144)

    def test_unitless_monomial_sub(self):
        "Tests that dimensionless and undimensioned subs can interact."
        x = Variable("x", "-")
        y = Variable("y")
        self.assertEqual(x.sub(x, y), y)

    def test_vector(self):
        x = Variable("x")
        y = Variable("y")
        z = VectorVariable(2, "z")
        p = x*y*z
        self.assertTrue(all(p.sub({x: 1, "y": 2}) == 2*z))
        self.assertTrue(all(p.sub({x: 1, y: 2, "z": [1, 2]}) ==
                            z.sub(z, [2, 4])))

        xvec = VectorVariable(3, "x", "m")
        xs = xvec[:2].sum()
        for x_ in ["x", xvec]:
            self.assertAlmostEqual(mag(xs.sub(x_, [1, 2, 3]).c), 3.0)

    def test_variable(self):
        """Test special single-argument substitution for Variable"""
        x = Variable('x')
        y = Variable('y')
        m = x*y**2
        self.assertEqual(x.sub(3), 3)
        self.assertEqual(x.sub(y), y)
        self.assertEqual(x.sub(m), m)
        # make sure x was not mutated
        self.assertEqual(x, Variable('x'))
        self.assertNotEqual(x.sub(3), Variable('x'))
        # also make sure the old way works
        self.assertEqual(x.sub({x: 3}), 3)
        self.assertEqual(x.sub({x: y}), y)
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


class TestGPSubs(unittest.TestCase):
    """Test substitution for Model and GP objects"""

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
        self.assertAlmostEqual(firstcost, 1760, 3)

    def test_skipfailures(self):
        x = Variable("x")
        x_min = Variable("x_{min}", [1, 2])

        m = Model(x, [x <= 1, x >= x_min])
        sol = m.solve(verbosity=0, skipsweepfailures=True)
        sol.table()
        self.assertEqual(len(sol), 1)

        m.substitutions[x_min][1][0] = 5
        with self.assertRaises(RuntimeWarning):
            sol = m.solve(verbosity=0, skipsweepfailures=True)

    def test_vector_sweep(self):
        """Test sweep involving VectorVariables"""
        x = Variable("x")
        y = VectorVariable(2, "y")
        m = Model(x, [x >= y.prod()])
        m.substitutions.update({y: ('sweep', [[2, 3], [5, 7], [9, 11]])})
        a = m.solve(verbosity=0)["cost"]
        b = [6, 14, 22, 15, 35, 55, 27, 63, 99]
        # below line fails with changing dictionary keys in py3
        self.assertTrue(all(abs(a-b)/(a+b) < 1e-7))
        m = Model(x, [x >= y.prod()])
        m.substitutions.update({y: ('sweep', [[2, 3], [5, 7, 11]])})
        a = m.solve(verbosity=0)["cost"]
        b = [10, 14, 22, 15, 21, 33]
        self.assertTrue(all(abs(a-b)/(a+b) < 1e-7))
        m = Model(x, [x >= y.prod()])
        m.substitutions.update({y: ('sweep', [[2, 3, 9], [5, 7, 11]])})
        self.assertRaises(ValueError, m.solve, verbosity=0)

    def test_linked_sweep(self):
        def night_hrs(day_hrs):
            "twenty four minus day hours"
            return 24 - day_hrs
        t_day = Variable("t_{day}", "hours")
        t_night = Variable("t_{night}", night_hrs, "hours", args=[t_day])
        x = Variable("x", "hours")
        m = Model(x, [x >= t_day, x >= t_night])
        m.substitutions.update({t_day: ("sweep", [8, 12, 16])})
        sol = m.solve(verbosity=0)
        self.assertEqual(len(sol["cost"]), 3)
        npt.assert_allclose(sol(t_day) + sol(t_night), 24)

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
        solv = sol['variables']
        a = solv["xi"]
        b = xi_dist*gpkit.units.N
        self.assertTrue(all(abs(a-b)/(a+b) < 1e-7))

    def test_model_composition_units(self):
        class Above(Model):
            "A simple upper bound on x"
            def __init__(self):
                x = Variable("x", "ft")
                x_max = Variable("x_{max}", 1, "yard")
                Model.__init__(self, 1/x, [x <= x_max])

        class Below(Model):
            "A simple lower bound on x"
            def __init__(self):
                x = Variable("x", "m")
                x_min = Variable("x_{min}", 1, "cm")
                Model.__init__(self, x, [x >= x_min])

        a, b = Above(), Below()
        concatm = Model(a.cost*b.cost, [a, b])
        concat_cost = concatm.solve(verbosity=0)["cost"]
        if not isinstance(a["x"].key.units, str):
            self.assertAlmostEqual(a.solve(verbosity=0)["cost"], 0.3333333)
            self.assertAlmostEqual(b.solve(verbosity=0)["cost"], 0.01)
            self.assertAlmostEqual(concat_cost, 0.0109361)  # 1 cm/1 yd
        a1, b1 = Above(), Below()
        m = a1.link(b1)
        m.cost = m["x"]
        sol = m.solve(verbosity=0)
        if not isinstance(m["x"].key.units, str):
            expected = (1*gpkit.units.cm/(1*m.cost.units)).to("dimensionless")
            self.assertAlmostEqual(sol["cost"], expected)  # 1 cm/(1 ft or 1 m)
        self.assertIn(m["x"], sol["variables"])
        self.assertIn(a1["x"], sol["variables"])
        self.assertIn(b1["x"], sol["variables"])
        self.assertNotIn(a["x"], sol["variables"])
        self.assertNotIn(b["x"], sol["variables"])

    def test_model_recursion(self):
        class Top(Model):
            "Some high level model"
            def __init__(self):
                x = Variable('x')
                y = Variable('y')
                Model.__init__(self, x, Sub().link([x >= y, y >= 1]))

        class Sub(Model):
            "A simple sub model"
            def __init__(self):
                y = Variable('y')
                Model.__init__(self, y, [y >= 2])

        sol = Top().solve(verbosity=0)
        self.assertAlmostEqual(sol['cost'], 2)


TESTS = [TestNomialSubs, TestGPSubs]

if __name__ == '__main__':
    run_tests(TESTS)
