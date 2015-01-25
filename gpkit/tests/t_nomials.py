import math
import unittest
from gpkit import Monomial, Posynomial


class t_Monomial(unittest.TestCase):

    def setUp(self):
        pass

    def test_init(self):
        m = Monomial({'x': 2, 'y': -1}, 5)
        m2 = Monomial({'x': 2, 'y': -1}, 5)
        self.assertEqual(m.var_locs, {'x': [0], 'y': [0]})
        self.assertEqual(m.exp, {'x': 2, 'y': -1})
        self.assertEqual(m.c, 5)
        self.assertEqual(m, m2)

        # default c and a
        m = Monomial('x')
        self.assertEqual(m.var_locs, {'x': [0]})
        self.assertEqual(m.exp, {'x': 1})
        self.assertEqual(m.c, 1)

        # single (string) var with non-default c
        m = Monomial('tau', .1)
        self.assertEqual(m.var_locs, {'tau': [0]})
        self.assertEqual(m.exp, {'tau': 1})
        self.assertEqual(m.c, .1)

        # variable names not compatible with python namespaces
        crazy_varstr = 'what the !!!/$\**?'
        m = Monomial({'x': 1, crazy_varstr: .5}, 25)
        self.assertTrue(crazy_varstr in m.exp)

        # non-positive c raises
        self.assertRaises(ValueError, Monomial, 'x', -2)
        self.assertRaises(ValueError, Monomial, {'x': 2}, -1.)
        self.assertRaises(ValueError, Monomial, 'x', 0)
        self.assertRaises(ValueError, Monomial, 'x', 0.0)

        # can create nameless Monomials
        x1 = Monomial()
        x2 = Monomial()
        V = Monomial('V')
        vel = Monomial('V')
        self.assertNotEqual(x1, x2)
        self.assertEqual(V, vel)

        # test label kwarg
        x = Monomial('x', label='dummy variable')
        self.assertEqual(x.exp.keys()[0].descr['label'], 'dummy variable')

    def test_repr(self):
        m = Monomial({'x': 2, 'y': -1}, 5)
        self.assertEqual(type(m.__repr__()), str)
        self.assertEqual(Monomial('x').__repr__(), 'gpkit.Monomial(x)')

    def test_latex(self):
        m = Monomial({'x': 2, 'y': -1}, 5)._latex()
        self.assertEqual(type(m), str)
        self.assertEqual(Monomial('x', 5)._latex(), '5x')

    def test_eq_ne(self):
        # simple one
        x = Monomial('x')
        y = Monomial('y')
        self.assertNotEqual(x, y)
        self.assertFalse(x == y)

        xx = Monomial('x')
        self.assertEqual(x, xx)
        self.assertFalse(x != xx)

        self.assertEqual(x, x)
        self.assertFalse(x != x)

        # scalar fails on type comparison even though c matches
        # NOTE(bqpd): not anymore!
        x = Monomial({}, 1)
        y = 1
        self.assertEqual(x, y)
        self.assertEqual(x, Monomial({}))

        # several vars
        m1 = Monomial({'a': 3, 'b': 2, 'c': 1}, 5)
        m2 = Monomial({'a': 3, 'b': 2, 'c': 1}, 5)
        m3 = Monomial({'a': 3, 'b': 2, 'c': 1}, 6)
        m4 = Monomial({'a': 3, 'b': 2}, 5)
        self.assertEqual(m1, m2)
        self.assertNotEqual(m1, m3)
        self.assertNotEqual(m1, m4)

    def test_div(self):
        x = Monomial('x')
        y = Monomial('y')
        z = Monomial('z')
        t = Monomial('t')
        a = 36*x/y
        # sanity check
        self.assertEqual(a, Monomial({'x': 1, 'y': -1}, 36))
        # divide by scalar
        self.assertEqual(a/9, 4*x/y)
        # divide by Monomial
        b = a / z
        self.assertEqual(b, 36*x/y/z)
        # make sure x unchanged
        self.assertEqual(a, Monomial({'x': 1, 'y': -1}, 36))
        # mixed new and old vars
        c = a / (0.5*t**2/x)
        self.assertEqual(c, Monomial({'x': 2, 'y': -1, 't': -2}, 72))

    def test_mul(self):
        x = Monomial({'x': 1, 'y': -1}, 4)
        # test integer division
        self.assertEqual(x/5, Monomial({'x': 1, 'y': -1}, 0.8))
        # divide by scalar
        self.assertEqual(x*9, Monomial({'x': 1, 'y': -1}, 36))
        # divide by Monomial
        y = x * Monomial('z')
        self.assertEqual(y, Monomial({'x': 1, 'y': -1, 'z': 1}, 4))
        # make sure x unchanged
        self.assertEqual(x, Monomial({'x': 1, 'y': -1}, 4))
        # mixed new and old vars
        z = x * Monomial({'x': -1, 't': 2}, .5)
        self.assertEqual(z, Monomial({'x': 0, 'y': -1, 't': 2}, 2))

    def test_pow(self):
        x = Monomial({'x': 1, 'y': -1}, 4)
        self.assertEqual(x, Monomial({'x': 1, 'y': -1}, 4))
        # identity
        self.assertEqual(x/x, Monomial({}, 1))
        # square
        self.assertEqual(x*x, x**2)
        # divide
        y = Monomial({'x': 2, 'y': 3}, 5)
        self.assertEqual(x/y, x*y**-1)
        # make sure x unchanged
        self.assertEqual(x, Monomial({'x': 1, 'y': -1}, 4))

    def test_numerical_precision(self):
        # not sure what to test here, placeholder for now
        c1, c2 = 1/700., 123e8
        m1 = Monomial({'x': 2, 'y': 1}, c1)
        m2 = Monomial({'y': -1, 'z': 3/2.}, c2)
        self.assertEqual(math.log((m1**4 * m2**3).c),
                         4*math.log(c1) + 3*math.log(c2))


class t_Posynomial(unittest.TestCase):

    def test_init(self):
        x = Monomial('x')
        y = Monomial('y')
        ms = [Monomial({'x': 1, 'y': 2}, 3.14),
              Monomial('y', 0.5),
              Monomial({'x': 3, 'y': 1}, 6),
              Monomial({}, 2)]
        exps, cs = [], []
        for m in ms:
            cs += m.cs.tolist()
            exps += m.exps
        p = Posynomial(exps, cs)
        # check arithmetic
        p2 = 3.14*x*y**2 + y/2 + x**3*6*y + 2
        self.assertEqual(p, p2)

        p = Posynomial(({'m': 1, 'v': 2},
                        {'m': 1, 'g': 1, 'h': 1}),
                       (0.5, 1))
        self.assertTrue(all(isinstance(x, float) for x in p.cs))
        self.assertEqual(len(p.exps), 2)
        self.assertEqual(set(p.var_locs), set(('m', 'g', 'h', 'v')))
        self.assertEqual(p.var_locs['g'], p.var_locs['h'])
        self.assertNotEqual(p.var_locs['g'], p.var_locs['v'])
        self.assertEqual(len(p.var_locs['m']), 2)
        self.assertTrue(all(len(p.var_locs[key]) == 1 for key in ('ghv')))

    def test_simplification(self):
        x = Monomial('x')
        y = Monomial('y')
        p1 = x + y + y + (x+y) + (y+x**2) + 3*x
        p2 = 4*y + x**2 + 5*x
        self.assertEqual(p1, p2)

    def test_posyposy_mult(self):
        x = Monomial('x')
        y = Monomial('y')
        p = x**2 + 2*y*x + y**2
        self.assertEqual((x+y)**2, p)
        p2 = 2*x**2 + 2*y*x + y**2*x + y**3
        self.assertEqual((x+y)*(2*x+y**2), p2)

    def test_constraint_gen(self):
        x = Monomial('x')
        y = Monomial('y')
        p = x**2 + 2*y*x + y**2
        self.assertEqual(p <= 1, p)
        self.assertEqual(p <= x, p/x)

    def test_integer_division(self):
        x = Monomial('x')
        y = Monomial('y')
        p = 4*x + y
        self.assertEqual(p/3, p/3.)
        self.assertTrue(((p/3).cs == [4./3., 1./3.]).all())

# test substitution

tests = [t_Posynomial, t_Monomial]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
