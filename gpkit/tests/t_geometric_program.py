import math
import unittest
from gpkit import GP, Monomial

SOLVERS = {'cvxopt': 5, 'mosek': 7, 'mosek_cli': 7}
# name: decimal places of accuracy


class t_GP(unittest.TestCase):
    name = "t_"

    def test_trivial_gp(self):
        x = Monomial('x')
        y = Monomial('y')
        prob = GP(cost=(x + 2*y),
                  constraints=[x*y >= 1],
                  solver=self.solver)
        sol = prob.solve()
        self.assertAlmostEqual(sol['x'], math.sqrt(2.), self.ndig)
        self.assertAlmostEqual(sol['y'], 1/math.sqrt(2.), self.ndig)
        self.assertAlmostEqual(sol['x'] + 2*sol['y'], 2*math.sqrt(2), self.ndig)

    def test_simpleflight(self):
        import simpleflight_gps
        gp = simpleflight_gps.single()
        gp.solver = self.solver
        sol = gp.solve()
        solcheck = dict(A=8.46,
                        C_D=0.0206,
                        C_f=0.0036,
                        C_L=0.499,
                        Re=3.68e+06,
                        S=16.4,
                        W=7.34e+03,
                        V=38.2,
                        W_w=2.40e+03)
        for key in solcheck:
            self.assertTrue(abs(1-sol[key]/solcheck[key]) < 1e-2)

testcases = [t_GP]

tests = []
for testcase in testcases:
    for solver, ndig in SOLVERS.items():
        test = type(testcase.__name__+"_"+solver,
                    (testcase,), {})
        setattr(test, 'solver', solver)
        setattr(test, 'ndig', ndig)
        tests.append(test)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
