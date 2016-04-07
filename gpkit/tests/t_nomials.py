"""Tests for Monomial, Posynomial, and Signomial classes"""
import math
import unittest
from gpkit import Variable, Monomial, Posynomial, Signomial
from gpkit import units, SignomialsEnabled


class TestMonomial(unittest.TestCase):
    """TestCase for the Monomial class"""

    def test_init(self):
        "Test multiple ways to create a Monomial"
        m = Monomial({'x': 2, 'y': -1}, 5)
        m2 = Monomial({'x': 2, 'y': -1}, 5)
        x, y = m.varkeys.map("x y".split())
        self.assertEqual(m.varlocs, {x: [0], y: [0]})
        self.assertEqual(m.exp, {x: 2, y: -1})
        self.assertEqual(m.c, 5)
        self.assertEqual(m, m2)

        # default c and a
        m = Monomial('x')
        x = m.varkeys.map("x")
        self.assertEqual(m.varlocs, {x: [0]})
        self.assertEqual(m.exp, {x: 1})
        self.assertEqual(m.c, 1)

        # single (string) var with non-default c
        m = Monomial('tau', .1)
        tau = m.varkeys.map(["tau"])
        self.assertEqual(m.varlocs, {tau: [0]})
        self.assertEqual(m.exp, {tau: 1})
        self.assertEqual(m.c, .1)

        # variable names not compatible with python namespaces
        crazy_varstr = 'what the !!!/$**?'
        m = Monomial({'x': 1, crazy_varstr: .5}, 25)
        crazy_varkey = m.varkeys.map([crazy_varstr])
        self.assertTrue(crazy_varkey in m.exp)

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
        self.assertEqual(list(x.exp)[0].descr['label'], 'dummy variable')

    def test_repr(self):
        "Simple tests for __repr__, which prints more than str"
        m = Monomial({'x': 2, 'y': -1}, 5)
        r = m.__repr__()
        self.assertEqual(type(r), str)
        self.assertEqual(Monomial('x').__repr__(), 'gpkit.Monomial(x)')

    def test_latex(self):
        "Test latex string creation"
        m = Monomial({'x': 2, 'y': -1}, 5).latex()
        self.assertEqual(type(m), str)
        self.assertEqual(Monomial('x', 5).latex(), '5x')

    def test_str_with_units(self):
        "Make sure __str__() works when units are involved"
        S = Monomial('S', units='m^2')
        rho = Monomial('rho', units='kg/m^3')
        x = rho*S
        xstr = str(x)
        self.assertEqual(type(xstr), str)
        self.assertTrue('S' in xstr and 'rho' in xstr)

    def test_eq_ne(self):
        "Test equality and inequality comparators"
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

        x = Monomial({}, 1)
        self.assertEqual(x, 1)
        self.assertEqual(x, Monomial({}))

        # several vars
        m1 = Monomial({'a': 3, 'b': 2, 'c': 1}, 5)
        m2 = Monomial({'a': 3, 'b': 2, 'c': 1}, 5)
        m3 = Monomial({'a': 3, 'b': 2, 'c': 1}, 6)
        m4 = Monomial({'a': 3, 'b': 2}, 5)
        self.assertEqual(m1, m2)
        self.assertNotEqual(m1, m3)
        self.assertNotEqual(m1, m4)

        # numeric
        self.assertEqual(Monomial(3), 3)
        self.assertEqual(Monomial(3), Monomial(3))
        self.assertNotEqual(Monomial(3), 2)
        self.assertNotEqual(Monomial('x'), 3)
        self.assertNotEqual(Monomial(3), Monomial('x'))

    def test_div(self):
        "Test Monomial division"
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
        "Test monomial multiplication"
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

        x0 = Monomial('x0')
        self.assertEqual(0.0, 0.0*x0)
        x1 = Monomial('x1')
        n_hat = [1, 0]
        p = n_hat[0]*x0 + n_hat[1]*x1
        self.assertEqual(p, x0)

    def test_pow(self):
        "Test Monomial exponentiation"
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
        "not sure what to test here, placeholder for now"
        c1, c2 = 1/700., 123e8
        m1 = Monomial({'x': 2, 'y': 1}, c1)
        m2 = Monomial({'y': -1, 'z': 3/2.}, c2)
        self.assertEqual(math.log((m1**4 * m2**3).c),
                         4*math.log(c1) + 3*math.log(c2))

    def test_units(self):
        "make sure multiplication with units works (issue 492)"
        # have had issues where Quantity.__mul__ causes wrong return type
        m = 1.2 * units.ft * Variable('x')**2
        self.assertTrue(isinstance(m, Monomial))
        self.assertEqual(m.units, 1*units.ft)
        # also multiply at the end, though this has not been a problem
        m = 0.5 * Variable('x')**2 * units.kg
        self.assertTrue(isinstance(m, Monomial))
        self.assertEqual(m.units, 1*units.kg)


class TestSignomial(unittest.TestCase):
    """TestCase for the Signomial class"""

    def test_init(self):
        "Test Signomial construction"
        x = Monomial('x')
        y = Monomial('y')
        with SignomialsEnabled():
            self.assertEqual(str(1 - x - y**2 - 1), "-x + -y**2")
            self.assertEqual((1 - x/y**2).latex(), "-\\frac{x}{y^{2}} + 1")
        self.assertRaises(TypeError, lambda: x-y)

    def test_eq_ne(self):
        "Test Signomial equality and inequality operators"
        x = Variable('x')
        xu = Variable('x', units="ft")
        with SignomialsEnabled():
            self.assertEqual(x - x**2, -x**2 + x)
            self.assertNotEqual(-x, -xu)
            # numeric
            self.assertEqual(Signomial(0), 0)
            self.assertNotEqual(Signomial(0), 1)
            self.assertEqual(Signomial(-3), -3)
            self.assertNotEqual(Signomial(-3), 3)


class TestPosynomial(unittest.TestCase):
    """TestCase for the Posynomial class"""

    def test_init(self):
        "Test Posynomial construction"
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
        m, g, h, v = p.varkeys.map("m g h v".split())
        self.assertTrue(all(isinstance(x, float) for x in p.cs))
        self.assertEqual(len(p.exps), 2)
        self.assertEqual(set(p.varlocs), set([m, g, h, v]))
        self.assertEqual(p.varlocs[g], p.varlocs[h])
        self.assertNotEqual(p.varlocs[g], p.varlocs[v])
        self.assertEqual(len(p.varlocs[m]), 2)
        self.assertTrue(all(len(p.varlocs[key]) == 1 for key in [g, h, v]))

    def test_eq(self):
        """Test Posynomial __eq__"""
        x = Variable('x')
        y = Variable('y')
        self.assertTrue((1 + x) == (1 + x))
        self.assertFalse((1 + x) == 2*(1 + x))
        self.assertFalse((1 + x) == 0.5*(1 + x))
        self.assertFalse((1 + x) == (1 + y))
        x = Variable('x', value=3)
        y = Variable('y', value=2)
        self.assertEqual((1 + x**2).value, (4 + y + y**2).value)

    def test_eq_units(self):
        p1 = Variable('x') + Variable('y')
        p2 = Variable('x') + Variable('y')
        p1u = Variable('x', units="m") + Variable('y', units="m")
        p2u = Variable('x', units="m") + Variable('y', units="m")
        self.assertEqual(p1, p2)
        self.assertEqual(p1u, p2u)
        self.assertFalse(p1 == p1u)
        self.assertNotEqual(p1, p1u)


    def test_simplification(self):
        "Make sure like monomial terms get automatically combined"
        x = Monomial('x')
        y = Monomial('y')
        p1 = x + y + y + (x+y) + (y+x**2) + 3*x
        p2 = 4*y + x**2 + 5*x
        # ps1 = [list(exp.keys())for exp in p1.exps]
        # ps2 = [list(exp.keys())for exp in p2.exps]
        # print("%s, %s" % (ps1, ps2))  # python 3 dict reordering
        self.assertEqual(p1, p2)

    def test_posyposy_mult(self):
        "Test multiplication of Posynomial with Posynomial"
        x = Monomial('x')
        y = Monomial('y')
        p1 = x**2 + 2*y*x + y**2
        p2 = (x+y)**2
        # ps1 = [list(exp.keys())for exp in p1.exps]
        # ps2 = [list(exp.keys())for exp in p2.exps]
        # print("%s, %s" % (ps1, ps2))  # python 3 dict reordering
        self.assertEqual(p1, p2)
        p1 = (x+y)*(2*x+y**2)
        p2 = 2*x**2 + 2*y*x + y**2*x + y**3
        # ps1 = [list(exp.keys())for exp in p1.exps]
        # ps2 = [list(exp.keys())for exp in p2.exps]
        # print("%s, %s" % (ps1, ps2))  # python 3 dict reordering
        self.assertEqual(p1, p2)

    def test_constraint_gen(self):
        "Test creation of Constraints via operator overloading"
        x = Monomial('x')
        y = Monomial('y')
        p = x**2 + 2*y*x + y**2
        self.assertEqual((p <= 1).as_posyslt1(), [p])
        self.assertEqual((p <= x).as_posyslt1(), [p/x])

    def test_integer_division(self):
        "Make sure division by integer doesn't use Python integer division"
        x = Monomial('x')
        y = Monomial('y')
        p = 4*x + y
        self.assertEqual(p/3, p/3.)
        equiv1 = all((p/3).cs == [1./3., 4./3.])
        equiv2 = all((p/3).cs == [4./3., 1./3.])
        self.assertTrue(equiv1 or equiv2)

    def test_diff(self):
        "Test differentiation (!!)"
        x = Variable('x')
        y = Variable('y')
        self.assertEqual(x.diff(x), 1)
        self.assertEqual(x.diff(y), 0)
        self.assertEqual((y**2).diff(y), 2*y)
        self.assertEqual((x + y**2).diff(y), 2*y)
        self.assertEqual((x + y**2).diff('x'), 1)
        self.assertEqual((x + x*y**2).diff(y), 2*x*y)
        self.assertEqual((2*y).diff(y), 2)
        # test with units
        x = Variable('x', units='ft')
        d = (3*x**2).diff(x)
        self.assertEqual(d, 6*x)
        # test negative exponent
        d = (1 + 1/y).diff(y)
        with SignomialsEnabled():
            expected = -y**-2
        self.assertEqual(d, expected)

    def test_mono_lower_bound(self):
        "Test monomial approximation"
        x = Variable('x')
        y = Variable('y')
        p = y**2 + 1
        self.assertRaises(TypeError, lambda: y.mono_lower_bound({y: 1}))
        # TODO: remove pylint warning below after Nomials refactor
        # pylint is confused because it thinks p is a Signomial
        # pylint: disable=no-member
        self.assertEqual(p.mono_lower_bound({y: 1}), 2*y)
        self.assertEqual(p.mono_lower_bound({y: 0}), 1)
        self.assertEqual((x*y**2 + 1).mono_lower_bound({y: 1, x: 1}),
                         2*y*x**0.5)
        # test with units
        d = Variable('d', units='ft')
        h = Variable('h', units='ft')
        p = (d*h**2 + h*d**2)
        m = p.mono_lower_bound({d: 1, h: 1})
        self.assertEqual(m, 2*(d*h)**1.5)

# test substitution

TESTS = [TestPosynomial, TestMonomial, TestSignomial]

if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
