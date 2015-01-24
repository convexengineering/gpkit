import unittest
from gpkit import (Monomial, Posynomial, PosyArray, Variable, VarKey,
                   VectorVariable)


class t_VarKey(unittest.TestCase):

    def test_init(self):
        # test type
        x = VarKey('x')
        self.assertEqual(type(x), VarKey)
        # test no args
        x = VarKey()
        self.assertEqual(type(x), VarKey)
        y = VarKey(x)
        self.assertEqual(x, y)
        # test special 'name' keyword overwriting behavior
        x = VarKey('x', flavour='vanilla')
        self.assertEqual(x.name, 'x')
        x = VarKey(name='x')
        self.assertEqual(x.name, 'x')
        self.assertRaises(ValueError, lambda: VarKey('x', name='y'))
        self.assertRaises(ValueError, lambda: VarKey(x, name='y'))

    def test_eq_neq(self):
        # no args
        x1 = VarKey()
        x2 = VarKey()
        self.assertTrue(x1 != x2)
        self.assertFalse(x1 == x2)
        self.assertEqual(x1, x1)
        V = VarKey('V')
        vel = VarKey('V')
        self.assertTrue(V == vel)
        self.assertFalse(V != vel)
        self.assertEqual(vel, vel)

    def test_repr(self):
        for k in ('x', '$x$', 'var_name', 'var name', '\\theta', '$\pi_{10}$'):
            var = VarKey(k)
            self.assertEqual(repr(var), k)
        # not sure what this means, but I want to know if it changes
        for num in (2, 2.0):
            v = VarKey(num)
            self.assertEqual(v, VarKey(str(num)))

    def test_dict_key(self):
        # make sure variables are well-behaved dict keys
        v = VarKey()
        x = VarKey('$x$')
        d = {v: 1273, x: 'foo'}
        self.assertEqual(d[v], 1273)
        self.assertEqual(d[x], 'foo')
        d = {VarKey(): None, VarKey(): 12}
        self.assertEqual(len(d), 2)


class t_Variable(unittest.TestCase):

    def test_init(self):
        v = Variable('v')
        self.assertTrue(isinstance(v, Variable))
        self.assertTrue(isinstance(v, Monomial))
        # test that operations on Variable cast to Monomial
        self.assertTrue(isinstance(3*v, Monomial))
        self.assertFalse(isinstance(3*v, Variable))


class t_VectorVariable(unittest.TestCase):

    def test_init(self):
        # test 1
        n = 3
        v = VectorVariable(n, 'v', label='dummy variable')
        v_mult = 3*v
        self.assertEqual(type(v), VectorVariable)
        for i in range(n):
            self.assertTrue(isinstance(v[i], Variable))
            self.assertTrue(isinstance(v[i], Monomial))
            # test that operations on Variable cast to Monomial
            self.assertTrue(isinstance(v_mult[i], Monomial))
            self.assertFalse(isinstance(v_mult[i], Variable))

        # test 2
        x = VectorVariable(3, 'x', label='dummy variable')
        x_0 = Monomial('x', idx=0, length=3, label='dummy variable')
        x_1 = Monomial('x', idx=1, length=3, label='dummy variable')
        x_2 = Monomial('x', idx=2, length=3, label='dummy variable')
        x2 = PosyArray([x_0, x_1, x_2])
        self.assertEqual(x, x2)


tests = [t_VarKey, t_Variable, t_VectorVariable]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
