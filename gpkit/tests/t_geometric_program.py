import math
import unittest
from gpkit import GP, Monomial, settings

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
        sol = prob.solve()["variables"]
        self.assertAlmostEqual(sol["x"], math.sqrt(2.), self.ndig)
        self.assertAlmostEqual(sol["y"], 1/math.sqrt(2.), self.ndig)
        self.assertAlmostEqual(sol["x"] + 2*sol["y"], 2*math.sqrt(2), self.ndig)

    def test_simpleflight(self):
        import simpleflight
        gp = simpleflight.gp()
        gp.solver = self.solver
        sol = gp.solve()
        freevarcheck = dict(A=8.46,
                            C_D=0.0206,
                            C_f=0.0036,
                            C_L=0.499,
                            Re=3.68e+06,
                            S=16.4,
                            W=7.34e+03,
                            V=38.2,
                            W_w=2.40e+03)
        # sensitivity values from p. 34 of Woody"s thesis
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
            self.assertTrue(abs(1-sol["variables"][key]/freevarcheck[key]) < 1e-2)
        for key in consenscheck:
            self.assertTrue(abs(1-sol["sensitivities"]["variables"][key]/consenscheck[key]) < 1e-2)

testcases = [t_GP]

tests = []
for testcase in testcases:
    for solver in settings["installed_solvers"]:
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
