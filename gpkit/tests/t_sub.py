"""Test substitution capability across gpkit"""
import unittest
import numpy as np
from gpkit import SignomialsEnabled
from gpkit import Variable, VectorVariable, Model, Signomial
from gpkit.small_scripts import mag


class TestNomialSubs(unittest.TestCase):
    """Test substitution for nomial-family objects"""

    def test_numeric(self):
        """Basic substitution of numeric value"""
        x = Variable("x")
        p = x**2
        self.assertEqual(p.sub(x, 3), 9)
        self.assertEqual(p.sub(x.varstrs["x"], 3), 9)
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

    def test_scalar_units(self):
        x = Variable("x", "m")
        xvk = x.varkey
        y = Variable("y", "km")
        yvk = y.varkey
        units_exist = bool(x.units)
        for x_ in ["x", xvk, x]:
            for y_ in ["y", yvk, y]:
                if not isinstance(y_, str) and units_exist:
                    expected = 0.001
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

    def test_vector(self):
        x = Variable("x")
        y = Variable("y")
        z = VectorVariable(2, "z")
        p = x*y*z
        self.assertTrue(all(p.sub({x: 1, "y": 2}) == 2*z))
        self.assertTrue(all(p.sub({x: 1, y: 2, "z": [1, 2]}) ==
                            z.sub(z, [2, 4])))

        x = VectorVariable(3, "x", "m")
        xs = x[:2].sum()
        for x_ in ["x", x]:
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
        x = VectorVariable(3, 'x')
        self.assertEqual(x[1].sub(3), 3)

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

    def test_vector_sweep(self):
        x = Variable("x")
        y = VectorVariable(2, "y")
        m = Model(x, [x >= y.prod()])
        m.substitutions.update({y: ('sweep', [[2, 3], [5, 7, 11]])})
        a = m.solve(verbosity=0)["cost"]
        b = [10, 14, 22, 15, 21, 33]
        # below line fails with changing dictionary keys in py3
        # self.assertTrue(all(abs(a-b)/(a+b) < 1e-7))
        m = Model(x, [x >= y.prod()])
        m.substitutions.update({y: ('sweep',
                                    [[2, 3], [5, 7], [9, 11], [13, 15]])})
        self.assertRaises(ValueError, m.solve)

    def test_vector_init(self):
        N = 6
        Weight = 50000
        xi_dist = 6*Weight/float(N)*(
                    (np.array(range(1, N+1)) - .5/float(N))/float(N) -
                    (np.array(range(1, N+1)) - .5/float(N))**2/float(N)**2
                                    )

        xi = VectorVariable(N, "xi", xi_dist, "N", "Constant Thrust per Bin")
        P = Variable("P", "N", "Total Power")
        phys_constraints = [P >= xi.sum()]
        objective = P
        eqns = phys_constraints
        m = Model(objective, eqns)
        sol = m.solve(verbosity=0)
        solv = sol['variables']
        a = solv["xi"]
        b = xi_dist
        self.assertTrue(all(abs(a-b)/(a+b) < 1e-7))

    # def test_simpleaircraft(self):
    #
    #     class DragModel(Model):
    #         def setup(self):
    #             pi = Variable("\\pi", np.pi, "-",
    #                           "half of the circle constant")
    #             e = Variable("e", 0.95, "-", "Oswald efficiency factor")
    #             S_wetratio = Variable("(\\frac{S}{S_{wet}})", 2.05, "-",
    #                                   "wetted area ratio")
    #             k = Variable("k", 1.2, "-", "form factor")
    #             C_f = Variable("C_f", "-", "skin friction coefficient")
    #             C_D = Variable("C_D", "-", "Drag coefficient of wing")
    #             C_L = Variable("C_L", "-", "Lift coefficent of wing")
    #             A = Variable("A", "-", "aspect ratio")
    #             S = Variable("S", "m^2", "total wing area")
    #             dum = Variable("dum", "-", "dummy variable")
    #
    #             if type(W.varkeys["W"].units) != str:
    #                 CDA0 = Variable("(CDA0)", 310.0, "cm^2",
    #                                 "fuselage drag area")
    #             else:
    #                 CDA0 = Variable("(CDA0)", 0.031, "m^2",
    #                                 "fuselage drag area")
    #
    #             C_D_fuse = CDA0/S
    #             C_D_wpar = k*C_f*S_wetratio
    #             C_D_ind = C_L**2/(pi*A*e)
    #
    #             return (Monomial(1),
    #                     [dum <= 2, dum >= 1, C_f >= 0.074/Re**0.2,
    #                      C_D >= C_D_fuse + C_D_wpar + C_D_ind])
    #
    #     class StructModel(Model):
    #         def setup(self):
    #             N_ult = Variable("N_{ult}", 3.8, "-", "ultimate load factor")
    #             tau = Variable("\\tau", 0.12, "-",
    #                            "airfoil thickness to chord ratio")
    #             W_w = Variable("W_w", "N", "wing weight")
    #             W = Variable("W", "N", "total aircraft weight")
    #
    #             if type(W.varkeys["W"].units) != str:
    #                 W_0 = Variable("W_0", 4.94, "kN",
    #                                "aircraft weight excluding wing")
    #                 W_w_strc = 8.71e-5*N_ult*A**1.5*(W_0*W*S)**0.5/tau/units.m
    #                 W_w_surf = (45.24*units.Pa) * S
    #             else:
    #                 W_0 = Variable("W_0", 4940, "N",
    #                                "aircraft weight excluding wing")
    #                 W_w_strc = 8.71e-5*(N_ult*A**1.5*(W_0*W*S)**0.5)/tau
    #                 W_w_surf = 45.24 * S
    #
    #             return (Monomial(1),
    #                     [W >= W_0 + W_w, W_w >= W_w_surf + W_w_strc])
    #
    #     rho = Variable("\\rho", 1.23, "kg/m^3", "density of air")
    #     mu = Variable("\\mu", 1.78e-5, "kg/m/s", "viscosity of air")
    #     C_Lmax = Variable("C_{L,max}", 1.5, "-", "max CL with flaps down")
    #     V_min = Variable("V_{min}", 22, "m/s", "takeoff speed")
    #     D = Variable("D", "N", "total drag force")
    #     A = Variable("A", "-", "aspect ratio")
    #     S = Variable("S", "m^2", "total wing area")
    #     C_D = Variable("C_D", "-", "Drag coefficient of wing")
    #     C_L = Variable("C_L", "-", "Lift coefficent of wing")
    #     Re = Variable("Re", "-", "Reynold's number")
    #     W = Variable("W", "N", "total aircraft weight")
    #     V = Variable("V", "m/s", "cruising speed")
    #     dum = Variable("dum", "-", "dummy variable")
    #
    #     equations = [D >= 0.5*rho*S*C_D*V**2,
    #                  Re <= (rho/mu)*V*(S/A)**0.5,
    #                  W <= 0.5*rho*S*C_L*V**2,
    #                  W <= 0.5*rho*S*C_Lmax*V_min**2,
    #                  dum >= 1,
    #                  dum <= 2, ]
    #
    #     lol = Variable("W", "N", "lol")
    #
    #     gp = GP(D, equations)
    #     gpl = link([gp, StructModel(name="struct"), DragModel(name="drag")],
    #                {rho: rho, "C_L": C_L, "C_D": C_D, "A": A, "S": S,
    #                 "Re": Re, "W": lol})
    #     self.assertEqual(gpl.varkeys["W"].descr["label"], "lol")
    #     self.assertIn("struct", gpl.varkeys["W_w"].descr["model"])
    #     self.assertIn("dum", gpl.varkeys)
    #
    #     k = GP.model_nums["drag"] - 1
    #     self.assertIn("dum_drag"+(str(k) if k else ""), gpl.varkeys)
    #     gpl2 = link([GP(D, equations), StructModel(name="struct"),
    #                  DragModel(name="drag")],
    #                 {rho: rho, "C_L": C_L, "C_D": C_D, "A": A, "S": S,
    #                  "Re": Re, "W": lol})
    #     self.assertIn("dum_drag"+str(k+1), gpl2.varkeys)
    #
    #     from gpkit.tests.simpleflight import simpleflight_generator
    #     sf = simpleflight_generator(
    #         disableUnits=(type(W.varkeys["W"].units) == str)).gp()
    #
    #     def sorted_solve_array(sol):
    #         return np.array([x[1] for x in
    #                          sorted(sol["variables"].items(),
    #                                 key=lambda x: x[0].name)])
    #     a = sorted_solve_array(sf.solve(verbosity=0))
    #     sol = gpl.solve(verbosity=0)
    #     del sol["variables"]["dum"], sol["variables"]["dum"]
    #     b = sorted_solve_array(sol)
    #     self.assertTrue(all(abs(a-b)/(a+b) < 1e-7))

TESTS = [TestNomialSubs, TestGPSubs]

if __name__ == '__main__':
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
