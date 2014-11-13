import unittest
from gpkit import Monomial, Posynomial, monovector, PosyArray, Variable


class t_Variable(unittest.TestCase):

    def test_init(self):
        # test type
        x = Variable('x')
        self.assertTrue(isinstance(x, Variable))
        # test no args
        x = Variable()
        self.assertTrue(isinstance(x, Variable))

    def test_eq_neq(self):
        # no args
        x1 = Variable()
        x2 = Variable()
        self.assertTrue(x1 != x2)
        self.assertFalse(x1 == x2)
        V = Variable('V')
        vel = Variable('V')
        self.assertTrue(V == vel)
        self.assertFalse(V != vel)


class t_utils(unittest.TestCase):

    def test_monify(self):
        x = Monomial('x', label='dummy variable')
        self.assertEqual(x.exp.keys()[0].descr["label"], 'dummy variable')

    def test_vectify(self):
        x = monovector(3, 'x', label='dummy variable')
        x_0 = Monomial('x', idx=0, length=3, label='dummy variable')
        x_1 = Monomial('x', idx=1, length=3, label='dummy variable')
        x_2 = Monomial('x', idx=2, length=3, label='dummy variable')
        x2 = PosyArray([x_0, x_1, x_2])
        self.assertEqual(x, x2)


tests = [t_utils, t_Variable]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
