import unittest
from gpkit import Variable, GP
from gpkit.geometric_program import GPSolutionArray


class t_GPSolutionArray(unittest.TestCase):

    def test_call(self):
        A = Variable('A', '-', 'Test Variable')
        prob = GP(A, [A >= 1])
        sol = prob.solve(printing=False)
        self.assertTrue(isinstance(sol(A), float))
        self.assertAlmostEqual(sol(A), 1.0, 10)


tests = [t_GPSolutionArray]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
