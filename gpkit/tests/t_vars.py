import unittest
from gpkit import Monomial, Posynomial, monovector, PosyArray


class t_variable(unittest.TestCase):

    def test_monify(self):
        x = Monomial('x', label='dummy variable')
        self.assertEqual(x.exp.keys()[0].descr["label"], 'dummy variable')


class t_vectorvariable(unittest.TestCase):

    def test_vectify(self):
        x = monovector(3, 'x', label='dummy variable')
        x_0 = Monomial('x', idx=0, length=3, label='dummy variable')
        x_1 = Monomial('x', idx=1, length=3, label='dummy variable')
        x_2 = Monomial('x', idx=2, length=3, label='dummy variable')
        x2 = PosyArray([x_0, x_1, x_2])
        self.assertEqual(x, x2)


tests = [t_variable, t_vectorvariable]

if __name__ == '__main__':
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(suite)
