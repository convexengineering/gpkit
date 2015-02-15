import unittest
import numpy as np
from gpkit import (Monomial, Posynomial, PosyArray, Variable, VarKey,
                   VectorVariable, units, GP, link)
from gpkit.small_scripts import mag


class t_NomialSubs(unittest.TestCase):

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
            self.assertAlmostEqual(x_changed_descr["units"]/y_descr["units"], 1.0)
        self.assertEqual(x.sub("x", x), x)

    def test_Scalar(self):
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

    def test_Vector(self):
        x = VectorVariable(3, "x", "m")
        xs = x[:2].sum()
        for x_ in ["x", x]:
            self.assertAlmostEqual(mag(xs.sub(x_, [1, 2, 3]).c), 3.0)

class t_GPSubs(unittest.TestCase):
    def test_VectorSweep(self):
        x = Variable("x")
        y = VectorVariable(2, "y")
        gp = GP(x, [x >= y.prod()])
        gp.sub(y, ('sweep', [[2, 3], [5, 7]]))
        a = gp.solve()["cost"]
        b = [10, 14, 15, 21]
        self.assertTrue(all(abs(a-b)/(a+b) < 1e-7))

    def test_simpleaircraft(self):
        mon = Variable
        vec = VectorVariable

        class DragModel(GP):
            def setup(self):
                pi = mon("\\pi", np.pi, "-", "half of the circle constant")
                e = mon("e", 0.95, "-", "Oswald efficiency factor")
                S_wetratio = mon("(\\frac{S}{S_{wet}})", 2.05, "-", "wetted area ratio")
                k = mon("k", 1.2, "-", "form factor")
                C_f = mon("C_f", "-", "skin friction coefficient")
                C_D = mon("C_D", "-", "Drag coefficient of wing")
                C_L = mon("C_L", "-", "Lift coefficent of wing")
                A = mon("A", "-", "aspect ratio")
                S = mon("S", "m^2", "total wing area")

                if type(W.varkeys["W"].descr["units"]) != str:
                    CDA0 = mon("(CDA0)", 310.0, "cm^2", "fuselage drag area")
                else:
                    CDA0 = mon("(CDA0)", 0.031, "m^2", "fuselage drag area")

                C_D_fuse = CDA0/S
                C_D_wpar = k*C_f*S_wetratio
                C_D_ind = C_L**2/(pi*A*e)

                return Monomial(1), [C_f >= 0.074/Re**0.2, C_D >= C_D_fuse + C_D_wpar + C_D_ind]

        class StructModel(GP):
            def setup(self):
                N_ult = mon("N_{ult}", 3.8, "-", "ultimate load factor")
                tau = mon("\\tau", 0.12, "-", "airfoil thickness to chord ratio")
                W_w = mon("W_w", "N", "wing weight")
                W = mon("W", "N", "total aircraft weight")

                if type(W.varkeys["W"].descr["units"]) != str:
                    W_0 = mon("W_0", 4.94, "kN", "aircraft weight excluding wing")
                    W_w_strc = 8.71e-5*(N_ult*A**1.5*(W_0*W*S)**0.5)/tau / units.m
                    W_w_surf = (45.24*units.Pa) * S
                else:
                    W_0 = mon("W_0", 4940, "N", "aircraft weight excluding wing")
                    W_w_strc = 8.71e-5*(N_ult*A**1.5*(W_0*W*S)**0.5)/tau
                    W_w_surf = 45.24 * S

                return Monomial(1), [W >= W_0 + W_w, W_w >= W_w_surf + W_w_strc]

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

        equations  = [D >= 0.5*rho*S*C_D*V**2,
                      Re <= (rho/mu)*V*(S/A)**0.5,
                      W <= 0.5*rho*S*C_L*V**2,
                      W <= 0.5*rho*S*C_Lmax*V_min**2,]

        lol = mon("W", "N", "lol")

        gp = GP(D, equations)
        gpl = link([gp, StructModel(name="struct"), DragModel(name="drag")],
                   {rho: rho, "C_L": C_L, "C_D": C_D, "A": A, "S": S, "Re": Re, "W": lol})
        self.assertEqual(gpl.varkeys["W"].descr["label"], "lol")

        from simpleflight import simpleflight_generator
        sf = simpleflight_generator(disableUnits=(type(W.varkeys["W"].descr["units"])==str)).gp()
        def sorted_solve_array(gp):
            return np.array(map(lambda x: x[1],
                            sorted(gp.solve(printing=False)["variables"].items(),
                                   key=lambda x: x[0].name)))
        a = sorted_solve_array(sf)
        b = sorted_solve_array(gpl)
        self.assertTrue(all(abs(a-b)/(a+b) < 1e-7))

tests = [t_NomialSubs, t_GPSubs]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
