import math
import unittest
from gpkit import GP, Monomial, settings, VectorVariable, Variable

NDIGS = {"cvxopt": 5, "mosek": 7, "mosek_cli": 7}
# name: decimal places of accuracy


class t_GP(unittest.TestCase):
    name = "t_"

    def test_trivial_gp(self):
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
        from simpleflight import simpleflight_generator
        sf = simpleflight_generator()
        self.simpleflight_test_core(sf.gp())

    def test_simpleflight_nounits(self):
        from simpleflight import simpleflight_generator
        sf = simpleflight_generator(disableUnits=True)
        self.simpleflight_test_core(sf.gp())

    def test_Mddtest(self):
        Cl = Variable("Cl", 0.5, "-", "Lift Coefficient")
        Mdd = Variable("Mdd", "-", "Drag Divergence Mach Number")
        gp1 = GP(1/Mdd, [1 >= 5*Mdd + 0.5, Mdd >= 0.00001])
        gp2 = GP(1/Mdd, [1 >= 5*Mdd + 0.5])
        gp3 = GP(1/Mdd, [1 >= 5*Mdd + Cl, Mdd >= 0.00001])
        sol1 = gp1.solve(printing=False)
        sol2 = gp2.solve(printing=False)
        sol3 = gp3.solve(printing=False)
        self.assertAlmostEqual(sol1(Mdd), sol2(Mdd))
        self.assertAlmostEqual(sol1(Mdd), sol3(Mdd))
        self.assertAlmostEqual(sol2(Mdd), sol3(Mdd))

    def test_additive_constants(self):
        x = Variable('x')
        gp = GP(1/x, [1 >= 5*x + 0.5, 1 >= 10*x])
        self.assertEqual(gp.cs[1], gp.cs[2])
        self.assertEqual(gp.A.data[1], gp.A.data[2])


testcases = [t_GP]

tests = []
for testcase in testcases:
    for solver in settings["installed_solvers"]:
        if solver:
            test = type(testcase.__name__+"_"+solver,
                        (testcase,), {})
            setattr(test, "solver", solver)
            setattr(test, "ndig", NDIGS[solver])
            tests.append(test)

if __name__ == "__main__":
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
