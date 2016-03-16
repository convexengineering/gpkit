"""Tests for NomialArray class"""
import unittest
import numpy as np
from gpkit import Monomial, Posynomial, NomialArray, VectorVariable
import gpkit


class TestNomialArray(unittest.TestCase):
    """TestCase for the NomialArray class.
    Also tests VectorVariable, since VectorVariable returns a NomialArray
    """

    def test_shape(self):
        x = VectorVariable((2, 3), 'x')
        self.assertEqual(x.shape, (2, 3))

    def test_ndim(self):
        x = VectorVariable((3, 4), 'x')
        self.assertEqual(x.ndim, 2)

    def test_array_mult(self):
        x = VectorVariable(3, 'x', label='dummy variable')
        x_0 = Monomial('x', idx=(0,), shape=(3,), label='dummy variable')
        x_1 = Monomial('x', idx=(1,), shape=(3,), label='dummy variable')
        x_2 = Monomial('x', idx=(2,), shape=(3,), label='dummy variable')
        p = x_0**2 + x_1**2 + x_2**2
        self.assertEqual(x.dot(x), p)
        m = NomialArray([[x_0**2, x_0*x_1, x_0*x_2],
                         [x_0*x_1, x_1**2, x_1*x_2],
                         [x_0*x_2, x_1*x_2, x_2**2]])
        self.assertEqual(x.outer(x), m)

    def test_elementwise_mult(self):
        m = Monomial('m')
        x = VectorVariable(3, 'x', label='dummy variable')
        x_0 = Monomial('x', idx=(0,), shape=(3,), label='dummy variable')
        x_1 = Monomial('x', idx=(1,), shape=(3,), label='dummy variable')
        x_2 = Monomial('x', idx=(2,), shape=(3,), label='dummy variable')
        # multiplication with numbers
        v = NomialArray([2, 2, 3]).T
        p = NomialArray([2*x_0, 2*x_1, 3*x_2]).T
        self.assertEqual(x*v, p)
        # division with numbers
        p2 = NomialArray([x_0/2, x_1/2, x_2/3]).T
        self.assertEqual(x/v, p2)
        # power
        p3 = NomialArray([x_0**2, x_1**2, x_2**2]).T
        self.assertEqual(x**2, p3)
        # multiplication with monomials
        p = NomialArray([m*x_0, m*x_1, m*x_2]).T
        self.assertEqual(x*m, p)
        # division with monomials
        p2 = NomialArray([x_0/m, x_1/m, x_2/m]).T
        self.assertEqual(x/m, p2)

    def test_constraint_gen(self):
        x = VectorVariable(3, 'x', label='dummy variable')
        x_0 = Monomial('x', idx=(0,), shape=(3,), label='dummy variable')
        x_1 = Monomial('x', idx=(1,), shape=(3,), label='dummy variable')
        x_2 = Monomial('x', idx=(2,), shape=(3,), label='dummy variable')
        v = NomialArray([1, 2, 3]).T
        p = [x_0, x_1/2, x_2/3]
        self.assertEqual((x <= v).as_posyslt1(), p)

    def test_substition(self):
        x = VectorVariable(3, 'x', label='dummy variable')
        c = {x: [1, 2, 3]}
        self.assertEqual(x.sub(c), [Monomial({}, e) for e in [1, 2, 3]])
        p = x**2
        self.assertEqual(p.sub(c), [Monomial({}, e) for e in [1, 4, 9]])
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

    def test_left_right(self):
        x = VectorVariable(10, 'x')
        xL = x.left
        xR = x.right
        self.assertEqual(xL[0], 0)
        self.assertEqual(xL[1], x[0])
        self.assertEqual(xR[-1], 0)
        self.assertEqual(xR[0], x[1])
        self.assertEqual((xL + xR)[1:-1], x[2:] + x[:-2])

        x = VectorVariable((2, 3), 'x')
        self.assertRaises(NotImplementedError, lambda: x.left)
        self.assertRaises(NotImplementedError, lambda: x.right)

    def test_sum(self):
        x = VectorVariable(5, 'x')
        p = x.sum()
        self.assertTrue(isinstance(p, Posynomial))
        self.assertEqual(p, sum(x))

        x = VectorVariable((2, 3), 'x')
        rowsum = x.sum(axis=1)
        colsum = x.sum(axis=0)
        self.assertTrue(isinstance(rowsum, NomialArray))
        self.assertTrue(isinstance(colsum, NomialArray))
        self.assertEqual(rowsum[0], sum(x[0]))
        self.assertEqual(colsum[0], sum(x[:, 0]))
        self.assertEqual(len(rowsum), 2)
        self.assertEqual(len(colsum), 3)

    def test_getitem(self):
        x = VectorVariable((2, 4), 'x')
        self.assertTrue(isinstance(x[0][0], Monomial))
        self.assertTrue(isinstance(x[0, 0], Monomial))

    def test_prod(self):
        x = VectorVariable(3, 'x')
        m = x.prod()
        self.assertTrue(isinstance(m, Monomial))
        self.assertEqual(m, x[0]*x[1]*x[2])
        self.assertEqual(m, np.prod(x))

    def test_outer(self):
        x = VectorVariable(3, 'x')
        y = VectorVariable(3, 'y')
        self.assertEqual(np.outer(x, y), x.outer(y))
        self.assertEqual(np.outer(y, x), y.outer(x))
        self.assertTrue(isinstance(x.outer(y), NomialArray))

    def test_empty(self):
        x = VectorVariable(3, 'x')
        # have to create this using slicing, to get object dtype
        empty_posy_array = x[:0]
        self.assertEqual(empty_posy_array.sum(), 0)
        self.assertEqual(empty_posy_array.prod(), 1)
        self.assertFalse(isinstance(empty_posy_array.sum(), (bool, np.bool_)))
        self.assertFalse(isinstance(empty_posy_array.prod(), (bool, np.bool_)))
        self.assertEqual(len(empty_posy_array), 0)
        self.assertEqual(empty_posy_array.ndim, 1)


TESTS = [TestNomialArray]

if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
