import unittest
from gpkit import Monomial, Posynomial
from gpkit.utils import monify, vectify
from gpkit.array import array


class t_array(unittest.TestCase):

    def test_array_mult(self):
        x = vectify('x', 3)
        x0, x1, x2 = monify('x0 x1 x2')
        p = x0**2 + x1**2 + x2**2
        self.assertEqual(x.dot(x), p)
        m = array([[x0**2, x0*x1, x0*x2],
                   [x0*x1, x1**2, x1*x2],
                   [x0*x2, x1*x2, x2**2]])
        self.assertEqual(x.outer(x), m)

    def test_elementwise_mult(self):
        x = vectify('x', 3)
        x0, x1, x2 = monify('x0 x1 x2')
        # multiplication
        v = array([1, 2, 3]).T
        p = array([1*x0, 2*x1, 3*x2]).T
        self.assertEqual(x*v, p)
        # division
        p2 = array([x0, x1/2, x2/3]).T
        self.assertEqual(x/v, p2)
        # power
        p3 = array([x0**2, x1**2, x2**2]).T
        self.assertEqual(x**2, p3)

    def test_constraint_gen(self):
        x = vectify('x', 3)
        x0, x1, x2 = monify('x0 x1 x2')
        v = array([1, 2, 3]).T
        p = [x0, x1/2, x2/3]
        self.assertEqual(x <= v, p)
        self.assertEqual(x < v, p)

    def test_substition(self):
        x = vectify('x', 3)
        c = {'x': [1, 2, 3]}
        self.assertEqual(x.sub(c), array([1, 2, 3]))
        p = x**2
        self.assertEqual(p.sub(c), array([1, 4, 9]))
        d = p.sum()
        # note the ugly Posy(Mono) down there...
        # this should be fixed when Mono inherits from Poly!
        self.assertEqual(d.sub(c), Posynomial([Monomial({},14)]))


tests = [t_array]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
