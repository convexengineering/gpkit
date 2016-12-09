"Runs each file listed in pwd/TESTS as a test"

import unittest
import os
import re
import importlib
from gpkit import settings
from gpkit.tests.helpers import generate_example_tests, new_test


class TestFiles(unittest.TestCase):
    "Stub to be filled with files in pwd/TEST"
    pass


def clean(s):
    """Parses string into valid python variable name

    https://stackoverflow.com/questions/3303312/
    how-do-i-convert-a-string-to-a-valid-variable-name-in-python
    """
    s = re.sub('[^0-9a-zA-Z_]', '', s)  # Remove invalid characters
    # Remove leading characters until we find a letter or underscore
    s = re.sub('^[^a-zA-Z_]+', '', s)
    return s


def add_filetest(TestClass, path):
    path = path.strip()
    print "adding test for", repr(path)

    def test_fn(self):
        mod = importlib.import_module(path[:-3])
        if not hasattr(mod, "test"):
            self.fail("file '%s' had no `test` function." % path)
        mod.test()

    setattr(TestClass, "test_"+clean(path), test_fn)


SOLVERS = settings["installed_solvers"]


def newtest_fn(name, solver, import_dict, path):
    "Doubly nested callbacks to run the test with `getattr(self, name)()`"
    return new_test(name, solver, import_dict, path,
                    testfn=(lambda name, import_dict, path:
                            lambda self: getattr(self, name)()))


def run(filename="TESTS", xmloutput=False):
    with open(filename, "r") as f:
        for path in f:
            add_filetest(TestFiles, path)
    TESTS = generate_example_tests("", [TestFiles], SOLVERS,
                                   newtest_fn=newtest_fn)
    from gpkit.tests.run_tests import run
    run(tests=TESTS, unitless=False, xmloutput=xmloutput)
