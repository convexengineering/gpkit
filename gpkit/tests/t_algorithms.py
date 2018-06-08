"""Unit tests for relaxation algorithms"""
import unittest

from gpkit import Variable, Model, SignomialsEnabled
from gpkit.algorithms.relaxations import RelaxedConstantsModel
from gpkit.tests.helpers import run_tests


class TestRelaxed(unittest.TestCase):
    """Tests for different relaxations"""
    def test_constants_relaxation(self):
        ujet = Variable("ujet")
        PK = Variable("PK")

        # Constants
        Dp = Variable("Dp", 0.662)
        fBLI = Variable("fBLI")
        fsurf = Variable("fsurf", 0.836)
        mdot = Variable("mdot", 1/0.7376)

        with SignomialsEnabled():
            m = Model(PK, [mdot*ujet + fBLI*Dp >= 1,
                           PK >= 0.5*mdot*ujet*(2 + ujet) + fBLI*fsurf*Dp])
        RCm = RelaxedConstantsModel(m)
        sol = m.localsolve(verbosity=0)
        RCsol = RCm.localsolve(verbosity=0)
        self.assertAlmostEqual(sol['cost'], RCsol['cost'], 3)

TESTS = [TestRelaxed]

if __name__ == '__main__':
    run_tests(TESTS)
