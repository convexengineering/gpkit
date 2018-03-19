"""Test KeyDict class"""
import unittest
import numpy as np
from gpkit import Variable, VectorVariable
import gpkit
from gpkit.keydict import KeyDict
from gpkit.tests.helpers import run_tests


class TestKeyDict(unittest.TestCase):
    """TestCase for the KeyDict class"""

    def test_setattr(self):
        kd = KeyDict()
        x = Variable("x", models=["test"])
        kd[x] = 1
        self.assertIn(x, kd)
        self.assertEqual(set(kd), set([x.key]))

    def test_getattr(self):
        kd = KeyDict()
        x = Variable("x", models=["motor"])
        kd[x] = 52
        self.assertEqual(kd[x], 52)
        self.assertEqual(kd[x.key], 52)
        self.assertEqual(kd["x"], 52)
        self.assertEqual(kd["x_motor"], 52)
        self.assertNotIn("x_someothermodelname", kd)

    def test_failed_getattr(self):
        kd = KeyDict()
        with self.assertRaises(KeyError):
            _ = kd["waldo"]
            # issue 893: failed __getitem__ caused state change
        self.assertNotIn("waldo", kd)
        waldo = Variable("waldo")
        kd.update({waldo: 5})
        res = kd["waldo"]
        self.assertEqual(res, 5)
        self.assertIn("waldo", kd)

    def test_dictlike(self):
        kd = KeyDict()
        kd["a string key"] = "a string value"
        self.assertTrue(isinstance(kd, dict))
        self.assertEqual(kd.keys(), ["a string key"])

    def test_vector(self):
        v = VectorVariable(3, "v")
        kd = KeyDict()
        kd[v] = np.array([2, 3, 4])
        self.assertTrue(all(kd[v] == kd[v.key]))
        self.assertTrue(all(kd["v"] == np.array([2, 3, 4])))
        self.assertEqual(v[0].key.idx, (0,))
        self.assertEqual(kd[v][0], kd[v[0]])
        self.assertEqual(kd[v][0], 2)
        kd[v[0]] = 6
        self.assertEqual(kd[v][0], kd[v[0]])
        self.assertEqual(kd[v][0], 6)
        self.assertTrue(all(kd[v] == np.array([6, 3, 4])))
        v = VectorVariable(3, "v", "m")
        kd[v] = np.array([2, 3, 4])
        if gpkit.units:
            kd[v[0]] = gpkit.units("inch")


TESTS = [TestKeyDict]


if __name__ == '__main__':
    run_tests(TESTS)
