import unittest
from gpkit import Monomial, Posynomial, PosyArray, VectorVariable
import gpkit


class t_PosyArray(unittest.TestCase):

    def test_array_mult(self):
        x = VectorVariable(3, 'x', label='dummy variable')
        x_0 = Monomial('x', idx=0, length=3, label='dummy variable')
        x_1 = Monomial('x', idx=1, length=3, label='dummy variable')
        x_2 = Monomial('x', idx=2, length=3, label='dummy variable')
        p = x_0**2 + x_1**2 + x_2**2
        self.assertEqual(x.dot(x), p)
        m = PosyArray([[x_0**2, x_0*x_1, x_0*x_2],
                       [x_0*x_1, x_1**2, x_1*x_2],
                       [x_0*x_2, x_1*x_2, x_2**2]])
        self.assertEqual(x.outer(x), m)

    def test_elementwise_mult(self):
        m = Monomial('m')
        x = VectorVariable(3, 'x', label='dummy variable')
        x_0 = Monomial('x', idx=0, length=3, label='dummy variable')
        x_1 = Monomial('x', idx=1, length=3, label='dummy variable')
        x_2 = Monomial('x', idx=2, length=3, label='dummy variable')
        # multiplication with numbers
        v = PosyArray([2, 2, 3]).T
        p = PosyArray([2*x_0, 2*x_1, 3*x_2]).T
        self.assertEqual(x*v, p)
        # division with numbers
        p2 = PosyArray([x_0/2, x_1/2, x_2/3]).T
        self.assertEqual(x/v, p2)
        # power
        p3 = PosyArray([x_0**2, x_1**2, x_2**2]).T
        self.assertEqual(x**2, p3)
        # multiplication with monomials
        p = PosyArray([m*x_0, m*x_1, m*x_2]).T
        self.assertEqual(x*m, p)
        # division with monomials
        p2 = PosyArray([x_0/m, x_1/m, x_2/m]).T
        self.assertEqual(x/m, p2)

    def test_constraint_gen(self):
        x = VectorVariable(3, 'x', label='dummy variable')
        x_0 = Monomial('x', idx=0, length=3, label='dummy variable')
        x_1 = Monomial('x', idx=1, length=3, label='dummy variable')
        x_2 = Monomial('x', idx=2, length=3, label='dummy variable')
        v = PosyArray([1, 2, 3]).T
        p = [x_0, x_1/2, x_2/3]
        self.assertEqual(x <= v, p)

    def test_substition(self):
        x = VectorVariable(3, 'x', label='dummy variable')
        c = {x: [1, 2, 3]}
        s = PosyArray([Monomial({}, e) for e in [1, 2, 3]])
        self.assertEqual(x.sub(c), s)
        p = x**2
        s2 = PosyArray([Monomial({}, e) for e in [1, 4, 9]])
        self.assertEqual(p.sub(c), s2)
        d = p.sum()
        self.assertEqual(d.sub(c), Monomial({}, 14))

    def test_units(self):
        # inspired by gpkit issue #106
        c = VectorVariable(5, "c", "m", "Local Chord")
        if gpkit.units:
            constraints = (c == 1*gpkit.units.m)
        else:
            constraints = (c == 1)
        self.assertEqual(len(constraints), 5)


tests = [t_PosyArray]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
