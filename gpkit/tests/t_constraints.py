import unittest
from gpkit import Variable
from gpkit.nomials import Constraint


class T_Constraint(unittest.TestCase):

    def test_additive_scalar(self):
        x = Variable('x')
        c1 = 1 >= 10*x
        c2 = 1 >= 5*x + 0.5
        self.assertEqual(type(c1), Constraint)
        self.assertEqual(type(c2), Constraint)
        self.assertEqual(c1.cs, c2.cs)
        self.assertEqual(c1.exps, c2.exps)

    def test_additive_scalar_gt1(self):
        x = Variable('x')

        def constr():
            return (1 >= 5*x + 1.1)
        self.assertRaises(ValueError, constr)


class T_MonoEQConstraint(unittest.TestCase):

    def test_placeholder(self):
        pass


_TESTS = [T_Constraint, T_MonoEQConstraint]

if __name__ == '__main__':
    _SUITE = unittest.TestSuite()
    _LOADER = unittest.TestLoader()

    for t in _TESTS:
        _SUITE.addTests(_LOADER.loadTestsFromTestCase(t))

    unittest.TextTestRunner(verbosity=2).run(_SUITE)
