import unittest
from gpkit import Monomial, Posynomial
from gpkit.variables import Variable, VectorVariable
from gpkit.posyarray import PosyArray


class t_variable(unittest.TestCase):

    def test_monify(self):
        x = Variable('x', 'dummy variable')
        y = Variable('y', 'dummy variable')
        self.assertEqual(x, Monomial('x'))
        self.assertEqual(y, Monomial('y'))


class t_vectorvariable(unittest.TestCase):

    def test_vectify(self):
        x = VectorVariable(3, 'x', 'dummy variable')
        x_0 = Monomial('x_0')
        x_1 = Monomial('x_1')
        x_2 = Monomial('x_2')
        x2 = PosyArray([x_0, x_1, x_2])
        self.assertEqual(x, x2)


tests = [t_variable, t_vectorvariable]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
