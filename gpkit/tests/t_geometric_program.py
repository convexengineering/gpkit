"""Tests for GP and SP classes"""
import math
import unittest
import numpy as np
from gpkit import GP, SP, Monomial, settings, VectorVariable, Variable
from gpkit.small_classes import CootMatrix
import gpkit

NDIGS = {"cvxopt": 5, "mosek": 7, "mosek_cli": 5}
# name: decimal places of accuracy


class TestGP(unittest.TestCase):
    """
    Test the GP class.
    This TestCase gets run once for each installed solver.
    """
    name = "TestGP_"
    # solver and ndig get set in loop at bottom this file, a bit hacky
    solver = None
    ndig = None

    def test_trivial_gp(self):
        """
        Create and solve a trivial GP:
            minimize    x + 2y
            subject to  xy >= 1

        The global optimum is (x, y) = (sqrt(2), 1/sqrt(2)).
        """
        x = Monomial('x')
        y = Monomial('y')
        prob = GP(cost=(x + 2*y),
                  constraints=[x*y >= 1],
                  solver=self.solver)
        sol = prob.solve(printing=False)["variables"]
        self.assertAlmostEqual(sol["x"], math.sqrt(2.), self.ndig)
        self.assertAlmostEqual(sol["y"], 1/math.sqrt(2.), self.ndig)
        self.assertAlmostEqual(sol["x"] + 2*sol["y"],
                               2*math.sqrt(2),
                               self.ndig)

    def test_trivial_vector_gp(self):
        """
        Create and solve a trivial GP with VectorVariables
        """
        x = VectorVariable(2, 'x')
        y = VectorVariable(2, 'y')
        prob = GP(cost=(sum(x) + 2*sum(y)),
                  constraints=[x*y >= 1],
                  solver=self.solver)
        sol = prob.solve(printing=False)['variables']
        self.assertEqual(sol['x'].shape, (2,))
        self.assertEqual(sol['y'].shape, (2,))
        for x, y in zip(sol['x'], sol['y']):
            self.assertAlmostEqual(x, math.sqrt(2.), self.ndig)
            self.assertAlmostEqual(y, 1/math.sqrt(2.), self.ndig)

    def simpleflight_test_core(self, gp):
        gp.solver = self.solver
        sol = gp.solve(printing=False)
        freevarcheck = dict(A=8.46,
                            C_D=0.0206,
                            C_f=0.0036,
                            C_L=0.499,
                            Re=3.68e+06,
                            S=16.4,
                            W=7.34e+03,
                            V=38.2,
                            W_w=2.40e+03)
        # sensitivity values from p. 34 of W. Hoburg's thesis
        consenscheck = {"(\\frac{S}{S_{wet}})": 0.4300,
                        "e": -0.4785,
                        "\\pi": -0.4785,
                        "V_{min}": -0.3691,
                        "k": 0.4300,
                        "\mu": 0.0860,
                        "(CDA0)": 0.0915,
                        "C_{L,max}": -0.1845,
                        "\\tau": -0.2903,
                        "N_{ult}": 0.2903,
                        "W_0": 1.0107,
                        "\\rho": -0.2275}
        for key in freevarcheck:
            sol_rat = sol["variables"][key]/freevarcheck[key]
            self.assertTrue(abs(1-sol_rat) < 1e-2)
        for key in consenscheck:
            sol_rat = sol["sensitivities"]["variables"][key]/consenscheck[key]
            self.assertTrue(abs(1-sol_rat) < 1e-2)

    def test_simpleflight(self):
        from gpkit.tests.simpleflight import simpleflight_generator
        sf = simpleflight_generator()
        self.simpleflight_test_core(sf.gp())

    def test_mdd_example(self):
        Cl = Variable("Cl", 0.5, "-", "Lift Coefficient")
        Mdd = Variable("Mdd", "-", "Drag Divergence Mach Number")
        gp1 = GP(1/Mdd, [1 >= 5*Mdd + 0.5, Mdd >= 0.00001])
        gp2 = GP(1/Mdd, [1 >= 5*Mdd + 0.5])
        gp3 = GP(1/Mdd, [1 >= 5*Mdd + Cl, Mdd >= 0.00001])
        sol1 = gp1.solve(printing=False, solver=self.solver)
        sol2 = gp2.solve(printing=False, solver=self.solver)
        sol3 = gp3.solve(printing=False, solver=self.solver)
        self.assertEqual(gp1.A, CootMatrix(row=[0, 1, 2],
                                           col=[0, 0, 0],
                                           data=[-1, 1, -1]))
        self.assertEqual(gp2.A, CootMatrix(row=[0, 1],
                                           col=[0, 0],
                                           data=[-1, 1]))
        # order of variables with a posynomial is not stable
        #   (though monomial order is)
        equiv1 = gp3.A == CootMatrix(row=[0, 2, 3],
                                     col=[0, 0, 0],
                                     data=[-1, 1, -1])
        equiv2 = gp3.A == CootMatrix(row=[0, 1, 3],
                                     col=[0, 0, 0],
                                     data=[-1, 1, -1])
        self.assertTrue(equiv1 or equiv2)
        self.assertAlmostEqual(sol1(Mdd), sol2(Mdd))
        self.assertAlmostEqual(sol1(Mdd), sol3(Mdd))
        self.assertAlmostEqual(sol2(Mdd), sol3(Mdd))

    def test_additive_constants(self):
        x = Variable('x')
        gp = GP(1/x, [1 >= 5*x + 0.5, 1 >= 10*x])
        gp.genA()
        self.assertEqual(gp.cs[1], gp.cs[2])
        self.assertEqual(gp.A.data[1], gp.A.data[2])

    def test_zeroing(self):
        L = Variable("L")
        k = Variable("k", 0)
        gpkit.enable_signomials()
        sol = GP(1/L, [L-5*k <= 10]).solve(printing=False, solver=self.solver)
        self.assertAlmostEqual(sol(L), 10, self.ndig)
        gpkit.disable_signomials()

    def test_composite_objective(self):
        L = Variable("L")
        W = Variable("W")
        eqns = [L >= 1, W >= 1,
                L*W == 10]
        obj = gpkit.composite_objective(L+W, W**-1 * L**-3, sub={L: 1, W: 1})
        sol = GP(obj, eqns).solve(printing=False, solver=self.solver)
        a = sol["sensitivities"]["variables"]["w_{CO}"].flatten()
        b = np.array([0, 0.98809322, 0.99461408, 0.99688676, 0.99804287,
                      0.99874303, 0.99921254, 0.99954926, 0.99980255, 1])
        self.assertTrue((abs(a-b)/(a+b+1e-7) < 1e-7).all())

    def test_singular(self):
        """
        Create and solve GP with a singular A matrix
        """
        if self.solver == 'cvxopt':
            # cvxopt can't solve this problem
            # (see https://github.com/cvxopt/cvxopt/issues/36)
            return
        x = Variable('x')
        y = Variable('y')
        gp = GP(y*x, [y*x >= 12])
        sol = gp.solve(solver=self.solver, printing=False)
        self.assertAlmostEqual(sol["cost"], 12)


class TestSP(unittest.TestCase):
    """test case for SP class -- gets run for each installed solver"""
    name = "TestSP_"
    solver = None
    ndig = None

    def test_trivial_sp(self):
        x = Variable('x')
        y = Variable('y')
        gpkit.enable_signomials()
        sp = SP(x, [x >= 1-y, y <= 0.1])
        sol = sp.localsolve(printing=False, solver=self.solver)
        self.assertAlmostEqual(sol["variables"]["x"], 0.9, self.ndig)
        sp = SP(x, [x >= 0.1, x+y >= 1, y <= 0.1])
        sol = sp.localsolve(printing=False, solver=self.solver)
        self.assertAlmostEqual(sol["variables"]["x"], 0.9, self.ndig)
        gpkit.disable_signomials()

    def test_united_interp(self):
        gpkit.enable_signomials()
        L = Variable("L", "m", "Length")
        W = Variable("W", "m", "Width")
        Obj = Variable("Obj", "1/m^4", "Objective")

        a = Variable("a", np.linspace(0, 1, 10), "-", "Pareto Sweep Variable")

        eqns = [L >= gpkit.units.m, W >= gpkit.units.m,
                L*W == 10*gpkit.units.m**2,
                Obj >= a*(2*L+2*W)*gpkit.units.m**-5 + (1-a)*(12*W**-1*L**-3)]

        sp = SP(Obj, eqns)
        if self.solver != "mosek_cli":
            # the mosek_cli solver takes 5s on this problem!
            sol = sp.localsolve(printing=False, solver=self.solver)
            solv = sol["variables"]
            self.assertAlmostEqual(solv["L"][3], 3.276042, 5)
        gpkit.disable_signomials()

    def test_issue180(self):
        gpkit.enable_signomials()
        L = Variable("L")
        Lmax = Variable("L_{max}", 10)
        W = Variable("W")
        Wmax = Variable("W_{max}", 10)
        A = Variable("A", 10)
        Obj = Variable("Obj")
        a_val = 0.01
        a = Variable("a", a_val)
        eqns = [L <= Lmax,
                W <= Wmax,
                L*W >= A,
                Obj >= a*(2*L + 2*W) + (1-a)*(12 * W**-1 * L**-3)]
        sp = SP(Obj, eqns)
        spsol = sp.localsolve(printing=False, solver=self.solver)
        gpkit.disable_signomials()
        # now solve as GP
        eqns[-1] = (Obj >= a_val*(2*L + 2*W) + (1-a_val)*(12 * W**-1 * L**-3))
        gp = GP(Obj, eqns)
        gpsol = gp.solve(printing=False, solver=self.solver)
        self.assertAlmostEqual(spsol['cost'], gpsol['cost'])


TEST_CASES = [TestGP, TestSP]

TESTS = []
for testcase in TEST_CASES:
    for solver in settings["installed_solvers"]:
        if solver:
            test = type(testcase.__name__+"_"+solver,
                        (testcase,), {})
            setattr(test, "solver", solver)
            setattr(test, "ndig", NDIGS[solver])
            TESTS.append(test)

if __name__ == "__main__":
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
