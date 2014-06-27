import unittest
from gpkit.monomial import Monomial
from gpkit.monomial import monify
from gpkit.posynomial import Posynomial


class Test_Monomial(unittest.TestCase):

    def setUp(self):
        pass

    def test_init(self):
        m = Monomial(['x', 'y'], 5, [1, -1])
        m2 = Monomial({'x': 1, 'y': -1}, 5)
        self.assertEqual(m.vars, set(['x', 'y']))
        self.assertEqual(m.exps, {'x': 1, 'y': -1})
        self.assertEqual(m.c, 5)
        self.assertEqual(m, m2)

        # default c and a
        m = Monomial('x')
        self.assertEqual(m.vars, set(['x']))
        self.assertEqual(m.exps, {'x': 1})
        self.assertEqual(m.c, 1)

    def test_repr(self):
        m = Monomial(['x', 'y'], 5, [2, -1])
        self.assertEqual(type(m.__repr__()), str)
        self.assertEqual(Monomial('x').__repr__(), 'x')

    def test_latex(self):
        m = Monomial(['x', 'y'], 5, [2, -1]).latex()
        self.assertEqual(type(m), str)
        self.assertTrue('$' in m, '$ not in %s' % m)
        self.assertEqual(Monomial('x', 5).latex(), '$5x$')

    def test_eq_ne(self):
        # simple one
        x = Monomial('x')
        y = Monomial('y')
        self.assertFalse(x == y)
        self.assertTrue(x != y)

        xx = Monomial('x')
        self.assertTrue(x == xx)
        self.assertFalse(x != xx)

        self.assertTrue(x == x)
        self.assertFalse(x != x)

        # scalar fails on type comparison even though c matches
        x = Monomial([], c=1)
        y = 1
        self.assertFalse(x == y)

        # several vars
        m1 = Monomial(['a', 'b', 'c'], c=5, a=[3, 2, 1])
        m2 = Monomial(['a', 'b', 'c'], c=5., a=[3, 2, 1])
        m3 = Monomial(['a', 'b', 'c'], c=6, a=[3, 2, 1])
        m4 = Monomial(['a', 'b'], c=5, a=[3, 2])
        self.assertTrue(m1 == m2)
        self.assertTrue(m1 != m3)
        self.assertTrue(m1 != m4)

    def test_div(self):
        x = Monomial(['x', 'y'], 36, [1, -1])
        # divide by scalar
        self.assertEqual(x/9, Monomial(['x', 'y'], 4, [1, -1]))
        # divide by Monomial
        y = x / Monomial('z')
        self.assertEqual(y, Monomial(['x', 'y', 'z'], 36, [1, -1, -1]))
        # make sure x unchanged
        self.assertEqual(x, Monomial(['x', 'y'], 36, [1, -1]))
        # mixed new and old vars
        z = x / Monomial(['x', 't'], .5, [-1, 2])
        self.assertEqual(z, Monomial(['x', 'y', 't'], 72, [2, -1, -2]))

    def test_mul(self):
        x = Monomial(['x', 'y'], 4, [1, -1])
        # divide by scalar
        self.assertEqual(x*9, Monomial(['x', 'y'], 36, [1, -1]))
        # divide by Monomial
        y = x * Monomial('z')
        self.assertEqual(y, Monomial(['x', 'y', 'z'], 4, [1, -1, 1]))
        # make sure x unchanged
        self.assertEqual(x, Monomial(['x', 'y'], 4, [1, -1]))
        # mixed new and old vars
        z = x * Monomial(['x', 't'], .5, [-1, 2])
        self.assertEqual(z, Monomial(['x', 'y', 't'], 2, [0, -1, 2]))

    def test_pow(self):
        x = Monomial(['x', 'y'], 4, [1, -1])
        # identity
        self.assertEqual(x/x, Monomial({}, 1))
        # square
        self.assertEqual(x*x, x**2)
        # divide
        y = Monomial(['x', 'y'], 5, [2, 3])
        self.assertEqual(x/y, x*y**-1)
        # make sure x unchanged
        self.assertEqual(x, Monomial(['x', 'y'], 4, [1, -1]))


class Test_Posynomial(unittest.TestCase):

    def test_basic(self):
        x, y = monify('x y')
        ms = [Monomial({'x': 1, 'y': 2}, 3.14),
              Monomial('y', 0.5),
              Monomial({'x': 3, 'y': 1}, -6),
              Monomial({}, 2)]
        p = Posynomial(ms)
        # check creation
        self.assertEqual(p.monomials, set(ms))
        # check arithmetic
        p2 = 3.14*x*y**2 + y/2 - x**3*6*y + 2
        self.assertEqual(p, p2)


class Test_monify(unittest.TestCase):

    def test_monify(self):
        x, y = monify('x y')
        self.assertEqual(x, Monomial('x'))
        self.assertEqual(y, Monomial('y'))


if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for t in [Test_Monomial, Test_monify, Test_Posynomial]:
        suite.addTests(loader.loadTestsFromTestCase(t))
    unittest.TextTestRunner(verbosity=2).run(suite)
