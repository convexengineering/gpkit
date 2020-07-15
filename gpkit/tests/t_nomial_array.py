"""Tests for NomialArray class"""
import unittest
import warnings as pywarnings
import numpy as np
from gpkit import Variable, Posynomial, NomialArray, VectorVariable, Monomial
from gpkit.constraints.set import ConstraintSet
from gpkit.exceptions import DimensionalityError
import gpkit


class TestNomialArray(unittest.TestCase):
    """TestCase for the NomialArray class.
    Also tests VectorVariable, since VectorVariable returns a NomialArray
    """

    def test_shape(self):
        x = VectorVariable((2, 3), 'x')
        self.assertEqual(x.shape, (2, 3))
        self.assertIsInstance(x.str_without(), str)
        self.assertIsInstance(x.latex(), str)

    def test_ndim(self):
        x = VectorVariable((3, 4), 'x')
        self.assertEqual(x.ndim, 2)

    def test_array_mult(self):
        x = VectorVariable(3, 'x', label='dummy variable')
        x_0 = Variable('x', idx=(0,), shape=(3,), label='dummy variable')
        x_1 = Variable('x', idx=(1,), shape=(3,), label='dummy variable')
        x_2 = Variable('x', idx=(2,), shape=(3,), label='dummy variable')
        p = x_0**2 + x_1**2 + x_2**2
        self.assertEqual(x.dot(x), p)
        m = NomialArray([[x_0**2, x_0*x_1, x_0*x_2],
                         [x_0*x_1, x_1**2, x_1*x_2],
                         [x_0*x_2, x_1*x_2, x_2**2]])
        self.assertEqual(x.outer(x), m)

    def test_elementwise_mult(self):
        m = Variable('m')
        x = VectorVariable(3, 'x', label='dummy variable')
        x_0 = Variable('x', idx=(0,), shape=(3,), label='dummy variable')
        x_1 = Variable('x', idx=(1,), shape=(3,), label='dummy variable')
        x_2 = Variable('x', idx=(2,), shape=(3,), label='dummy variable')
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
        self.assertIsInstance(v.str_without(), str)
        self.assertIsInstance(v.latex(), str)
        self.assertIsInstance(p.str_without(), str)
        self.assertIsInstance(p.latex(), str)

    def test_constraint_gen(self):
        x = VectorVariable(3, 'x', label='dummy variable')
        x_0 = Variable('x', idx=(0,), shape=(3,), label='dummy variable')
        x_1 = Variable('x', idx=(1,), shape=(3,), label='dummy variable')
        x_2 = Variable('x', idx=(2,), shape=(3,), label='dummy variable')
        v = NomialArray([1, 2, 3]).T
        p = [x_0, x_1/2, x_2/3]
        constraint = ConstraintSet([x <= v])
        self.assertEqual(list(constraint.as_hmapslt1({})), [e.hmap for e in p])

    def test_substition(self):  # pylint: disable=no-member
        x = VectorVariable(3, 'x', label='dummy variable')
        c = {x: [1, 2, 3]}
        self.assertEqual(x.sub(c), [Monomial({}, e) for e in [1, 2, 3]])
        p = x**2
        self.assertEqual(p.sub(c), [Monomial({}, e) for e in [1, 4, 9]])  # pylint: disable=no-member
        d = p.sum()
        self.assertEqual(d.sub(c), Monomial({}, 14))  # pylint: disable=no-member

    def test_units(self):
        # inspired by gpkit issue #106
        c = VectorVariable(5, "c", "m", "Local Chord")
        constraints = (c == 1*gpkit.units.m)
        self.assertEqual(len(constraints), 5)
        # test an array with inconsistent units
        with pywarnings.catch_warnings():  # skip the UnitStrippedWarning
            pywarnings.simplefilter("ignore")
            mismatch = NomialArray([1*gpkit.units.m, 1*gpkit.ureg.ft, 1.0])
        self.assertRaises(DimensionalityError, mismatch.sum)
        self.assertEqual(mismatch[:2].sum().c, 1.3048*gpkit.ureg.m)  # pylint:disable=no-member
        self.assertEqual(mismatch.prod().c, 1*gpkit.ureg.m*gpkit.ureg.ft)  # pylint:disable=no-member

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
        pows = NomialArray([x[0], x[0]**2, x[0]**3])
        self.assertEqual(pows.prod(), x[0]**6)

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
        self.assertRaises(ValueError, empty_posy_array.sum)
        self.assertRaises(ValueError, empty_posy_array.prod)
        self.assertEqual(len(empty_posy_array), 0)
        self.assertEqual(empty_posy_array.ndim, 1)


TESTS = [TestNomialArray]

if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=wrong-import-position
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
