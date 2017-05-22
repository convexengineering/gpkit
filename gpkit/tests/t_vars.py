"""Test VarKey, Variable, VectorVariable, and ArrayVariable classes"""
import unittest
import numpy as np
from gpkit import (Monomial, NomialArray, Variable, VarKey,
                   VectorVariable, ArrayVariable)
import gpkit
from gpkit.nomials import Variable as PlainVariable


class TestVarKey(unittest.TestCase):
    """TestCase for the VarKey class"""

    def test_init(self):
        """Test VarKey initialization"""
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
        # pylint: disable=redundant-keyword-arg
        self.assertRaises(TypeError, lambda: VarKey('x', name='y'))
        # pylint: disable=redundant-keyword-arg
        self.assertRaises(TypeError, lambda: VarKey(x, name='y'))

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
        self.assertNotEqual(x2.key, x3.key)
        # do we want these to collide?
        self.assertEqual(x1.key, x3.key)

    def test_repr(self):
        """Test __repr__ method"""
        for k in ('x', '$x$', 'var_name', 'var name', r"\theta", r'$\pi_{10}$'):
            var = VarKey(k)
            self.assertEqual(repr(var), k)
        # not sure what this means, but I want to know if it changes
        for num in (2, 2.0):
            v = VarKey(num)
            self.assertEqual(v, VarKey(str(num)))

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
        for vk in (VarKey(), x, VarKey(x), VarKey(units='m')):
            self.assertTrue(hasattr(vk, 'units'))


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
        """Hashes should collide independent of units"""
        x1 = Variable("x", "-", "first x")
        x2 = Variable("x", "-", "second x")
        self.assertEqual(hash(x1), hash(x2))
        p1 = Variable("p", "psi", "first pressure")
        p2 = Variable("p", "psi", "second pressure")
        self.assertEqual(hash(p1), hash(p2))
        xu = Variable("x", "m", "x with units")
        if gpkit.units:
            self.assertNotEqual(hash(x1), hash(xu))
        else:
            self.assertEqual(hash(x1), hash(xu))

    def test_unit_parsing(self):
        x = Variable("x", "s^0.5/m^0.5")
        y = Variable("y", "(m/s)^-0.5")
        self.assertEqual(x.units, y.units)


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
        x_0 = Monomial('x', idx=(0,), shape=(3,), label='dummy variable')
        x_1 = Monomial('x', idx=(1,), shape=(3,), label='dummy variable')
        x_2 = Monomial('x', idx=(2,), shape=(3,), label='dummy variable')
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
        strx = str(x)
        self.assertEqual(strx.count("["), 3)
        self.assertEqual(strx.count("]"), 3)


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
