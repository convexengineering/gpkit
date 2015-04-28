import unittest
import numpy as np
from gpkit import Monomial, Variable, VectorVariable, units, GP, link
from gpkit.small_scripts import mag


class t_NomialSubs(unittest.TestCase):

    def test_Basic(self):
        x = Variable("x")
        p = x**2
        self.assertEqual(p.sub(x, 3), 9)
        self.assertEqual(p.sub(x.varkeys["x"], 3), 9)
        self.assertEqual(p.sub("x", 3), 9)

    def test_StringMutation(self):
        x = Variable("x", "m")
        descr_before = x.exp.keys()[0].descr
        y = x.sub("x", "y")
        descr_after = x.exp.keys()[0].descr
        self.assertEqual(descr_before, descr_after)
        x_changed_descr = dict(descr_before)
        x_changed_descr["name"] = "y"
        y_descr = y.exp.keys()[0].descr
        self.assertEqual(x_changed_descr["name"], y_descr["name"])
        if type(descr_before["units"]) != str:
            self.assertAlmostEqual(x_changed_descr["units"]/y_descr["units"],
                                   1.0)
        self.assertEqual(x.sub("x", x), x)

    def test_ScalarUnits(self):
        x = Variable("x", "m")
        xvk = x.varkeys.values()[0]
        descr_before = x.exp.keys()[0].descr
        y = Variable("y", "km")
        yvk = y.varkeys.values()[0]
        for x_ in ["x", xvk, x]:
            for y_ in ["y", yvk, y]:
                if not isinstance(y_, str) and type(xvk.descr["units"]) != str:
                    expected = 0.001
                else:
                    expected = 1.0
                self.assertAlmostEqual(expected, mag(x.sub(x_, y_).c))
        if type(xvk.descr["units"]) != str:
            z = Variable("z", "s")
            self.assertRaises(ValueError, y.sub, y, z)

    def test_Vector(self):
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


class t_GPSubs(unittest.TestCase):
    def test_VectorSweep(self):
        x = Variable("x")
        y = VectorVariable(2, "y")
        gp = GP(x, [x >= y.prod()])
        gp.sub(y, ('sweep', [[2, 3], [5, 7, 11]]))
        a = gp.solve(printing=False)["cost"]
        b = [10, 14, 22, 15, 21, 33]
        self.assertTrue(all(abs(a-b)/(a+b) < 1e-7))

        gp = GP(x, [x >= y.prod()])

        def bad_sub(gp):
            gp.sub(y, ('sweep', [[2, 3], [5, 7], [9, 11], [13, 15]]))
        self.assertRaises(ValueError, bad_sub, gp)

    def test_VectorInit(self):
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
        gp = GP(objective, eqns)
        sol = gp.solve(printing=False)
        solv = sol['variables']
        a = solv["xi"]
        b = xi_dist
        self.assertTrue(all(abs(a-b)/(a+b) < 1e-7))

    def test_simpleaircraft(self):
        mon = Variable

        class DragModel(GP):
            def setup(self):
                pi = Variable("\\pi", np.pi, "-",
                              "half of the circle constant")
                e = Variable("e", 0.95, "-", "Oswald efficiency factor")
                S_wetratio = Variable("(\\frac{S}{S_{wet}})", 2.05, "-",
                                      "wetted area ratio")
                k = Variable("k", 1.2, "-", "form factor")
                C_f = Variable("C_f", "-", "skin friction coefficient")
                C_D = Variable("C_D", "-", "Drag coefficient of wing")
                C_L = Variable("C_L", "-", "Lift coefficent of wing")
                A = Variable("A", "-", "aspect ratio")
                S = Variable("S", "m^2", "total wing area")
                dum = Variable("dum", "-", "dummy variable")

                if type(W.varkeys["W"].descr["units"]) != str:
                    CDA0 = Variable("(CDA0)", 310.0, "cm^2",
                                    "fuselage drag area")
                else:
                    CDA0 = Variable("(CDA0)", 0.031, "m^2",
                                    "fuselage drag area")

                C_D_fuse = CDA0/S
                C_D_wpar = k*C_f*S_wetratio
                C_D_ind = C_L**2/(pi*A*e)

                return (Monomial(1),
                        [dum <= 2, dum >= 1, C_f >= 0.074/Re**0.2,
                         C_D >= C_D_fuse + C_D_wpar + C_D_ind])

        class StructModel(GP):
            def setup(self):
                N_ult = mon("N_{ult}", 3.8, "-", "ultimate load factor")
                tau = mon("\\tau", 0.12, "-",
                          "airfoil thickness to chord ratio")
                W_w = mon("W_w", "N", "wing weight")
                W = mon("W", "N", "total aircraft weight")

                if type(W.varkeys["W"].descr["units"]) != str:
                    W_0 = mon("W_0", 4.94, "kN",
                              "aircraft weight excluding wing")
                    W_w_strc = 8.71e-5*N_ult*A**1.5*(W_0*W*S)**0.5/tau/units.m
                    W_w_surf = (45.24*units.Pa) * S
                else:
                    W_0 = mon("W_0", 4940, "N",
                              "aircraft weight excluding wing")
                    W_w_strc = 8.71e-5*(N_ult*A**1.5*(W_0*W*S)**0.5)/tau
                    W_w_surf = 45.24 * S

                return (Monomial(1),
                        [W >= W_0 + W_w, W_w >= W_w_surf + W_w_strc])

        rho = mon("\\rho", 1.23, "kg/m^3", "density of air")
        mu = mon("\\mu", 1.78e-5, "kg/m/s", "viscosity of air")
        C_Lmax = mon("C_{L,max}", 1.5, "-", "max CL with flaps down")
        V_min = mon("V_{min}", 22, "m/s", "takeoff speed")
        D = mon("D", "N", "total drag force")
        A = mon("A", "-", "aspect ratio")
        S = mon("S", "m^2", "total wing area")
        C_D = mon("C_D", "-", "Drag coefficient of wing")
        C_L = mon("C_L", "-", "Lift coefficent of wing")
        Re = mon("Re", "-", "Reynold's number")
        W = mon("W", "N", "total aircraft weight")
        V = mon("V", "m/s", "cruising speed")
        dum = mon("dum", "-", "dummy variable")

        equations = [D >= 0.5*rho*S*C_D*V**2,
                     Re <= (rho/mu)*V*(S/A)**0.5,
                     W <= 0.5*rho*S*C_L*V**2,
                     W <= 0.5*rho*S*C_Lmax*V_min**2,
                     dum >= 1,
                     dum <= 2, ]

        lol = mon("W", "N", "lol")

        gp = GP(D, equations)
        gpl = link([gp, StructModel(name="struct"), DragModel(name="drag")],
                   {rho: rho, "C_L": C_L, "C_D": C_D, "A": A, "S": S,
                    "Re": Re, "W": lol})
        self.assertEqual(gpl.varkeys["W"].descr["label"], "lol")
        self.assertIn("struct", gpl.varkeys["W_w"].descr["model"])
        self.assertIn("dum", gpl.varkeys)

        k = GP.model_nums["drag"] - 1
        self.assertIn("dum_drag"+(str(k) if k else ""), gpl.varkeys)
        gpl2 = link([GP(D, equations), StructModel(name="struct"),
                     DragModel(name="drag")],
                    {rho: rho, "C_L": C_L, "C_D": C_D, "A": A, "S": S,
                     "Re": Re, "W": lol})
        self.assertIn("dum_drag"+str(k+1), gpl2.varkeys)

        from simpleflight import simpleflight_generator
        sf = simpleflight_generator(
            disableUnits=(type(W.varkeys["W"].descr["units"]) == str)).gp()

        def sorted_solve_array(sol):
            return np.array([x[1] for x in
                             sorted(sol["variables"].items(),
                                    key=lambda x: x[0].name)])
        a = sorted_solve_array(sf.solve(printing=False))
        sol = gpl.solve(printing=False)
        del sol["variables"]["dum"], sol["variables"]["dum"]
        b = sorted_solve_array(sol)
        self.assertTrue(all(abs(a-b)/(a+b) < 1e-7))

tests = [t_NomialSubs, t_GPSubs]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
