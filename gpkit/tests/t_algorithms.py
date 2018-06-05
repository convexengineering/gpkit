"""Unit tests for algorithms using SimPleAC and """
import unittest

from gpkit import Variable, Model, VectorVariable
from gpkit.algorithms.relaxations import ConstraintsRelaxed, ConstantsRelaxed
import gpkit
from gpkit.tests.helpers import run_tests

# Importing testing models
#from gpkitmodels.SP.SimPleAC.SimPleAC import SimPleAC


class TestRelaxed(unittest.TestCase):
    def dummytest(self):
        self.assertAlmostEqual(1,1)

#     """Tests for different relaxations"""
#     def test_SP_relaxation(self):
#         m = SimPleAC()
#         m.cost = m['W_f']
#         RelaxedConstraints = ConstraintsRelaxed(m)
#         RelaxedConstants = R
#
#         self.assertAlmostEqual(m2.solve(verbosity=0)(x), 3, 5)

TESTS = [TestRelaxed]

if __name__ == '__main__':
    run_tests(TESTS)
