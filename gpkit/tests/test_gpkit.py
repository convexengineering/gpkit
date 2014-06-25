import unittest
from gpkit.monomial import Monomial


class Test_Monomial(unittest.TestCase):

    def setUp(self):
        pass

    def test_init(self):
        m = Monomial(['x','y'], 5, [1, -1])
        self.assertEqual(m.vars, set(['x', 'y']))
        self.assertEqual(m.a, [1, -1])
        self.assertEqual(m.c, 5)
        
        # default c and a
        m = Monomial('x')
        self.assertEqual(m.vars, set(['x']))
        self.assertEqual(m.a, [1])
        self.assertEqual(m.c, 1)

    def test_repr(self):
        m = Monomial(['x', 'y'], 5, [2, -1])
        self.assertEqual(type(m.__repr__()), str)
        self.assertEqual(Monomial('x').__repr__(), 'x')

    def test_latex(self):
        m = Monomial(['x', 'y'], 5, [2, -1]).latex()
        self.assertEqual(type(m), str)
        self.assertTrue('$'in m, '$ not in %s' % m)
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
        m1 = Monomial(['a','b','c'], c=5, a=[3,2,1])
        m2 = Monomial(['a','b','c'], c=5., a=[3,2,1])
        m3 = Monomial(['a','b','c'], c=6, a=[3,2,1])
        m4 = Monomial(['a','b'], c=5, a=[3,2])
        self.assertTrue(m1 == m2)
        self.assertTrue(m1 != m3)
        self.assertTrue(m1 != m4)

    def test_div(self):
        x = Monomial(['x', 'y'], 36, [1, -1])
        # divide by scalar
        self.assertEqual(x/9, Monomial(['x', 'y'], 4, [1, -1]))
        # divide by Monomial
        y = x/Monomial('z')
        self.assertEqual(y, Monomial(['x', 'y', 'z'], 36, [1, -1, -1]))
        # make sure x unchanged
        self.assertEqual(x, Monomial(['x', 'y'], 36, [1, -1]))
        # mixed new and old vars
        z = x/Monomial(['x', 't'], .5, [-1, 2])
        self.assertEqual(z, Monomial(['x','y','t'], 72, [2,-1,-2]))

    def test_mul(self):
        x = Monomial(['x', 'y'], 4, [1, -1])
        # divide by scalar
        self.assertEqual(x*9, Monomial(['x', 'y'], 36, [1, -1]))
        # divide by Monomial
        y = x*Monomial('z')
        self.assertEqual(y, Monomial(['x', 'y', 'z'], 4, [1, -1, 1]))
        # make sure x unchanged
        self.assertEqual(x, Monomial(['x', 'y'], 4, [1, -1]))
        # mixed new and old vars
        z = x*Monomial(['x', 't'], .5, [-1, 2])
        self.assertEqual(z, Monomial(['x','y','t'], 2, [0,-1,2]))

    def test_pow(self):
        x = Monomial(['x', 'y'], 4, [1, -1])
        # divide by scalar
        self.assertEqual(x*9, Monomial(['x', 'y'], 36, [1, -1]))
        # divide by Monomial
        y = x*Monomial('z')
        self.assertEqual(y, Monomial(['x', 'y', 'z'], 4, [1, -1, 1]))
        # make sure x unchanged
        self.assertEqual(x, Monomial(['x', 'y'], 4, [1, -1]))
        # mixed new and old vars
        z = x*Monomial(['x', 't'], .5, [-1, 2])
        self.assertEqual(z, Monomial(['x','y','t'], 2, [0,-1,2]))


class TestAnotherThing(unittest.TestCase):

    def test_placeholder(self):
        pass

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for t in [Test_Monomial, TestAnotherThing]:
        suite.addTests(loader.loadTestsFromTestCase(t))
    unittest.TextTestRunner(verbosity=2).run(suite)

