import unittest
from gpkit import Monomial, Posynomial
from gpkit.utils import monify, vectify
from gpkit.array import array


class t_monify(unittest.TestCase):

    def test_monify(self):
        x, y = monify('x y')
        self.assertEqual(x, Monomial('x'))
        self.assertEqual(y, Monomial('y'))


class t_vectify(unittest.TestCase):

    def test_vectify(self):
        x = vectify('x', 3)
        x2 = array(monify('x_0 x_1 x_2'))
        self.assertEqual(x, x2)


tests = [t_monify, t_vectify]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
