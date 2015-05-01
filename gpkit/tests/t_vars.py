import unittest
import numpy as np
from gpkit import (Monomial, Posynomial, PosyArray, Variable, VarKey,
                   VectorVariable, ArrayVariable)


class T_VarKey(unittest.TestCase):

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


class T_Variable(unittest.TestCase):

    def test_init(self):
        v = Variable('v')
        self.assertTrue(isinstance(v, Variable))
        self.assertTrue(isinstance(v, Monomial))
        # test that operations on Variable cast to Monomial
        self.assertTrue(isinstance(3*v, Monomial))
        self.assertFalse(isinstance(3*v, Variable))

    def test_value(self):
        a = Variable('a')
        b = Variable('b', value=4)
        c = a**2 + b
        self.assertEqual(b.value, 4)
        self.assertTrue(isinstance(b.value, float))
        p1 = c.value
        p2 = a**2 + 4
        ps1 = [list(exp.keys())for exp in p1.exps]
        ps2 = [list(exp.keys())for exp in p2.exps]
        #print("%s, %s" % (ps1, ps2))  # python 3 dict reordering
        self.assertEqual(p1, p2)
        self.assertEqual(a.value, a)


class T_VectorVariable(unittest.TestCase):

    def test_init(self):
        # test 1
        n = 3
        v = VectorVariable(n, 'v', label='dummy variable')
        v_mult = 3*v
        for i in range(n):
            self.assertTrue(isinstance(v[i], Variable))
            self.assertTrue(isinstance(v[i], Monomial))
            # test that operations on Variable cast to Monomial
            self.assertTrue(isinstance(v_mult[i], Monomial))
            self.assertFalse(isinstance(v_mult[i], Variable))

        # test 2
        x = VectorVariable(3, 'x', label='dummy variable')
        x_0 = Monomial('x', idx=(0,), shape=(3,), label='dummy variable')
        x_1 = Monomial('x', idx=(1,), shape=(3,), label='dummy variable')
        x_2 = Monomial('x', idx=(2,), shape=(3,), label='dummy variable')
        x2 = PosyArray([x_0, x_1, x_2])
        self.assertEqual(x, x2)

        # test inspired by issue 137
        N = 20
        x_arr = np.arange(0, 5., 5./N) + 1e-6
        x = VectorVariable(N, 'x', x_arr, 'm', "Beam Location")

    def test_array_variable(self):
        """
        Make sure ArrayVariable is a shortcut to VectorVariable
        (I want to know if this changes).
        """
        self.assertTrue(ArrayVariable is VectorVariable)


class TestArrayVariable(unittest.TestCase):

    def test_shape(self):
        x = ArrayVariable((2, 3), 'x')
        self.assertEqual(x.shape, (2, 3))

    def test_ndim(self):
        x = ArrayVariable((3, 4), 'x')
        self.assertEqual(x.ndim, 2)

    def test_left_right(self):
        x = ArrayVariable((2, 3), 'x')
        self.assertRaises(NotImplementedError, lambda: x.left)
        self.assertRaises(NotImplementedError, lambda: x.right)

    def test_sum(self):
        x = ArrayVariable((2, 3), 'x')
        rowsums = x.sum(axis=1)
        self.assertTrue(isinstance(rowsums, PosyArray))
        self.assertEqual(len(rowsums), 2)
        colsums = x.sum(axis=0)
        self.assertTrue(isinstance(colsums, PosyArray))
        self.assertEqual(len(colsums), 3)

    def test_str(self):
        x = ArrayVariable((2, 4), 'x')
        strx = str(x)
        self.assertEqual(strx.count("["), 3)
        self.assertEqual(strx.count("]"), 3)

    def test_getitem(self):
        x = ArrayVariable((2, 4), 'x')
        self.assertTrue(isinstance(x[0][0], Monomial))
        self.assertTrue(isinstance(x[0, 0], Monomial))


TESTS = [T_VarKey, T_Variable, T_VectorVariable]
TESTS += [TestArrayVariable]    # avoiding merge conflict -- fix this

if __name__ == '__main__':
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
