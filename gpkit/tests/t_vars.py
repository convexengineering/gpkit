"""Test VarKey, Variable, VectorVariable, and ArrayVariable classes"""
from __future__ import print_function
import unittest
import sys
import numpy as np
from gpkit import (Monomial, NomialArray, Variable, VarKey,
                   VectorVariable, ArrayVariable)
import gpkit
from gpkit.nomials import Variable as PlainVariable

if sys.version_info >= (3, 0):
    unicode = str  # pylint:disable=redefined-builtin,invalid-name


class TestVarKey(unittest.TestCase):
    """TestCase for the VarKey class"""

    def test_init(self):
        """Test VarKey initialization"""
        # test no-name init
        _ = ArrayVariable(1)
        # test protected field
        with self.assertRaises(ValueError):
            _ = ArrayVariable(1, idx=5)
        # test type
        x = VarKey('x')
        self.assertEqual(type(x), VarKey)
        # test no args
        x = VarKey()
        self.assertEqual(type(x), VarKey)
        y = VarKey(**x.descr)
        self.assertEqual(x, y)
        # test special 'name' keyword overwriting behavior
        x = VarKey('x', flavour='vanilla')
        self.assertEqual(x.name, 'x')
        x = VarKey(name='x')
        self.assertEqual(x.name, 'x')
        # pylint: disable=redundant-keyword-arg
        self.assertRaises(TypeError, lambda: VarKey('x', name='y'))
        self.assertIsInstance(x.latex(), str)
        self.assertIsInstance(x.latex_unitstr(), unicode)

    def test_ast(self): # pylint: disable=too-many-statements
        t = Variable("t")
        u = Variable("u")
        v = Variable("v")
        w = Variable("w")
        x = VectorVariable(3, "x")
        y = VectorVariable(3, "y")
        z = VectorVariable(3, "z")
        a = VectorVariable((3, 2), "a")

        print(w >= x)
        self.assertEqual(str(3*(x + y)*z), "3*(x[:] + y[:])*z[:]")
        nni = 3
        ii = np.tile(np.arange(1., nni+1.), a.shape[1:]+(1,)).T
        self.assertEqual(str(w*NomialArray(ii)/nni)[:4], "w*[[")
        self.assertEqual(str(w*NomialArray(ii)/nni)[-4:], "]]/3")
        self.assertEqual(str(NomialArray(ii)*w/nni)[:2], "[[")
        self.assertEqual(str(NomialArray(ii)*w/nni)[-6:], "]]*w/3")
        self.assertEqual(str(w*ii/nni)[:4], "w*[[")
        self.assertEqual(str(w*ii/nni)[-4:], "]]/3")
        self.assertEqual(str(w*(ii/nni))[:4], "w*[[")
        self.assertEqual(str(w*(ii/nni))[-2:], "]]")
        self.assertEqual(str(w >= (x[0]*t + x[1]*u)/v),
                         "w >= (x[0]*t + x[1]*u)/v")
        self.assertEqual(str(x), "x[:]")
        self.assertEqual(str(x*2), "x[:]*2")
        self.assertEqual(str(2*x), "2*x[:]")
        self.assertEqual(str(x + 2), "x[:] + 2")
        self.assertEqual(str(2 + x), "2 + x[:]")
        self.assertEqual(str(x/2), "x[:]/2")
        self.assertEqual(str(2/x), "2/x[:]")
        if sys.version_info <= (3, 0):
            self.assertEqual(str(x**3), "x[:]^3")
        self.assertEqual(str(-x), "-x[:]")
        self.assertEqual(str(x/y/z), "x[:]/y[:]/z[:]")
        self.assertEqual(str(x/(y/z)), "x[:]/(y[:]/z[:])")
        self.assertEqual(str(x >= y), "x[:] >= y[:]")
        self.assertEqual(str(x >= y + z), "x[:] >= y[:] + z[:]")
        self.assertEqual(str(x[:2]), "x[:2]")
        self.assertEqual(str(x[:]), "x[:]")
        self.assertEqual(str(x[1:]), "x[1:]")
        self.assertEqual(str(y * [1, 2, 3]), "y[:]*[1, 2, 3]")
        self.assertEqual(str(x[:2] == (y*[1, 2, 3])[:2]),
                         "x[:2] = (y[:]*[1, 2, 3])[:2]")
        self.assertEqual(str(y + [1, 2, 3]), "y[:] + [1, 2, 3]")
        self.assertEqual(str(x == y + [1, 2, 3]), "x[:] = y[:] + [1, 2, 3]")
        self.assertEqual(str(x >= y + [1, 2, 3]), "x[:] >= y[:] + [1, 2, 3]")
        self.assertEqual(str(a[:, 0]), "a[:,0]")
        self.assertEqual(str(a[2, :]), "a[2,:]")
        g = 1 + 3*a[2, 0]**2
        gstrbefore = str(g)
        g.ast = None
        gstrafter = str(g)
        if sys.version_info <= (3, 0):
            self.assertEqual(gstrbefore, gstrafter)

    def test_eq_neq(self):
        """Test boolean equality operators"""
        # no args
        vk1 = VarKey()
        vk2 = VarKey()
        self.assertTrue(vk1 != vk2)
        self.assertFalse(vk1 == vk2)
        self.assertEqual(vk1, vk1)
        V = VarKey('V')
        vel = VarKey('V')
        self.assertTrue(V == vel)
        self.assertFalse(V != vel)
        self.assertEqual(vel, vel)
        x1 = Variable("x", 3, "m")
        x2 = Variable("x", 2, "ft")
        x3 = Variable("x", 2, "m")
        if gpkit.units:
            self.assertNotEqual(x2.key, x3.key)
        else:  # units don't distinguish variables when they're disabled
            self.assertEqual(x2.key, x3.key)
        self.assertEqual(x1.key, x3.key)

    def test_repr(self):
        """Test __repr__ method"""
        for k in ('x', '$x$', 'var_name', 'var name', r"\theta", r'$\pi_{10}$'):
            var = VarKey(k)
            self.assertEqual(repr(var), k)

    def test_dict_key(self):
        """make sure variables are well-behaved dict keys"""
        v = VarKey()
        x = VarKey('$x$')
        d = {v: 1273, x: 'foo'}
        self.assertEqual(d[v], 1273)
        self.assertEqual(d[x], 'foo')
        d = {VarKey(): None, VarKey(): 12}
        self.assertEqual(len(d), 2)

    def test_units_attr(self):
        """Make sure VarKey objects have a units attribute"""
        x = VarKey('x')
        for vk in (VarKey(), x, VarKey(**x.descr), VarKey(units='m')):
            self.assertTrue("units" in vk.descr)

class TestVariable(unittest.TestCase):
    """TestCase for the Variable class"""

    def test_init(self):
        """Test Variable initialization"""
        v = Variable('v')
        self.assertTrue(isinstance(v, PlainVariable))
        self.assertTrue(isinstance(v, Monomial))
        # test that operations on Variable cast to Monomial
        self.assertTrue(isinstance(3*v, Monomial))
        self.assertFalse(isinstance(3*v, PlainVariable))

    def test_value(self):
        """Detailed tests for value kwarg of __init__"""
        a = Variable('a')
        b = Variable('b', value=4)
        c = a**2 + b
        self.assertEqual(b.value, 4)
        self.assertTrue(isinstance(b.value, float))
        p1 = c.value
        p2 = a**2 + 4
        self.assertEqual(p1, p2)
        self.assertEqual(a.value, a)

    def test_hash(self):
        x1 = Variable("x", "-", "first x")
        x2 = Variable("x", "-", "second x")
        self.assertEqual(hash(x1), hash(x2))
        p1 = Variable("p", "psi", "first pressure")
        p2 = Variable("p", "psi", "second pressure")
        self.assertEqual(hash(p1), hash(p2))
        xu = Variable("x", "m", "x with units")
        if gpkit.units:
            self.assertNotEqual(hash(x1), hash(xu))
        else:  # units don't distinguish variables when they're disabled
            self.assertEqual(hash(x1), hash(xu))

    def test_unit_parsing(self):
        x = Variable("x", "s^0.5/m^0.5")
        y = Variable("y", "(m/s)^-0.5")
        self.assertEqual(x.units, y.units)

    def test_to(self):
        if gpkit.units:
            x = Variable("x", "ft")
            self.assertEqual(x.to("inch").c.magnitude, 12)

    def test_eq_ne(self):
        # test for #1138
        W = Variable("W", 5, "lbf", "weight of 1 bag of sugar")
        self.assertTrue(W != W.key)
        self.assertTrue(W.key != W)
        self.assertFalse(W == W.key)
        self.assertFalse(W.key == W)


class TestVectorVariable(unittest.TestCase):
    """TestCase for the VectorVariable class.
    Note: more relevant tests in t_posy_array."""

    def test_init(self):
        """Test VectorVariable initialization"""
        # test 1
        n = 3
        v = VectorVariable(n, 'v', label='dummy variable')
        self.assertTrue(isinstance(v, NomialArray))
        v_mult = 3*v
        for i in range(n):
            self.assertTrue(isinstance(v[i], PlainVariable))
            self.assertTrue(isinstance(v[i], Monomial))
            # test that operations on Variable cast to Monomial
            self.assertTrue(isinstance(v_mult[i], Monomial))
            self.assertFalse(isinstance(v_mult[i], PlainVariable))

        # test 2
        x = VectorVariable(3, 'x', label='dummy variable')
        x_0 = Variable('x', idx=(0,), shape=(3,), label='dummy variable')
        x_1 = Variable('x', idx=(1,), shape=(3,), label='dummy variable')
        x_2 = Variable('x', idx=(2,), shape=(3,), label='dummy variable')
        x2 = NomialArray([x_0, x_1, x_2])
        self.assertEqual(x, x2)

        # test inspired by issue 137
        N = 20
        x_arr = np.arange(0, 5., 5./N) + 1e-6
        x = VectorVariable(N, 'x', x_arr, 'm', "Beam Location")

    def test_constraint_creation_units(self):
        v = VectorVariable(2, "v", "m/s")
        c = (v >= 40*gpkit.units("ft/s"))
        c2 = (v >= np.array([20, 30])*gpkit.units("ft/s"))
        if gpkit.units:
            self.assertTrue(c.right.units)
            self.assertTrue(NomialArray(c2.right).units)
        else:
            self.assertEqual(type(c.right), int)
            self.assertEqual(type(c2.right), np.ndarray)


class TestArrayVariable(unittest.TestCase):
    """TestCase for the ArrayVariable class"""

    def test_is_vector_variable(self):
        """
        Make sure ArrayVariable is a shortcut to VectorVariable
        (we want to know if this changes).
        """
        self.assertTrue(ArrayVariable is VectorVariable)

    def test_str(self):
        """Make sure string looks something like a numpy array"""
        x = ArrayVariable((2, 4), 'x')
        self.assertEqual(str(x), "x[:]")


class TestVectorize(unittest.TestCase):
    """TestCase for gpkit.vectorize"""

    def test_shapes(self):
        with gpkit.Vectorize(3):
            with gpkit.Vectorize(5):
                y = gpkit.Variable("y")
                x = gpkit.VectorVariable(2, "x")
            z = gpkit.VectorVariable(7, "z")

        self.assertEqual(y.shape, (5, 3))
        self.assertEqual(x.shape, (2, 5, 3))
        self.assertEqual(z.shape, (7, 3))


TESTS = [TestVarKey, TestVariable, TestVectorVariable, TestArrayVariable,
         TestVectorize]

if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
