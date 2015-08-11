"""Tests for small_classes.py and small_scripts.py"""
import unittest
import numpy as np
from gpkit.small_classes import HashVector

class TestHashVector(unittest.TestCase):
    """TestCase for the HashVector class"""

    def test_init(self):
        """Make sure HashVector acts like a dict"""
        # args and kwargs
        hv = HashVector([(2, 3), (1, 10)], dog='woof')
        self.assertTrue(isinstance(hv, dict))
        self.assertEqual(hv, {2: 3, 1: 10, 'dog': 'woof'})
        # no args
        self.assertEqual(HashVector(), {})
        # creation from dict
        self.assertEqual(HashVector({'x': 7}), {'x': 7})

    def test_neg(self):
        """Test negation"""
        hv = HashVector(x=7, y=0, z=-1)
        self.assertEqual(-hv, {'x': -7, 'y': 0, 'z': 1})

    def test_pow(self):
        """Test exponentiation"""
        hv = HashVector(x=4, y=0, z=1)
        self.assertEqual(hv**0.5, {'x': 2, 'y': 0, 'z': 1})

    def test_mul_add(self):
        """Test multiplication and addition"""
        a = HashVector(x=1, y=7)
        b = HashVector()
        c = HashVector(x=3, z=4)
        # multiplication and addition by scalars
        r = a*0
        self.assertEqual(r, HashVector(x=0, y=0))
        self.assertTrue(isinstance(r, HashVector))
        r = a - 2
        self.assertEqual(r, HashVector(x=-1, y=5))
        self.assertTrue(isinstance(r, HashVector))
        # multiplication and addition by dicts
        self.assertEqual(a + b, a)
        self.assertEqual(a + b + c, HashVector(x=4, y=7, z=4))
        self.assertEqual(a * b * c, HashVector())
        self.assertEqual(a * {'x': 6, 'k': 4}, HashVector(x=6))

TESTS = [TestHashVector]


if __name__ == '__main__':
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
