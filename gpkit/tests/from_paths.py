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


def clean(string):
    """Parses string into valid python variable name

    https://stackoverflow.com/questions/3303312/
    how-do-i-convert-a-string-to-a-valid-variable-name-in-python
    """
    string = re.sub('[^0-9a-zA-Z_]', '', string)  # Remove invalid characters
    # Remove leading characters until we find a letter or underscore
    string = re.sub('^[^a-zA-Z_]+', '', string)
    return string


def add_filetest(testclass, path):
    """Add test that imports the given path and runs its test() function

    TODO: make work for subdirectories, using os.chdir"""
    path = path.strip()
    print "adding test for", repr(path)

    def test_fn(self):
        top_level = os.getcwd()
        try:
            dirname = os.path.dirname(path)
            if dirname:
                os.chdir(os.path.dirname(path))
            mod = importlib.import_module(os.path.basename(path)[:-3])
        finally:
            os.chdir(top_level)
        if not hasattr(mod, "test"):
            self.fail("file '%s' had no `test` function." % path)
        mod.test()

    setattr(testclass, "test_"+clean(path), test_fn)


SOLVERS = settings["installed_solvers"]


def newtest_fn(name, solver, import_dict, path):
    "Doubly nested callbacks to run the test with `getattr(self, name)()`"
    return new_test(name, solver, import_dict, path,
                    testfn=(lambda name, import_dict, path:
                            lambda self: getattr(self, name)()))  # pylint:disable=undefined-variable


def run(filename="TESTS", xmloutput=False):
    "Parse and run paths from a given file for each solver"
    with open(filename, "r") as f:
        for path in f:
            add_filetest(TestFiles, path)
    tests = generate_example_tests("", [TestFiles], SOLVERS,
                                   newtest_fn=newtest_fn)
    from gpkit.tests.run_tests import run as run_
    run_(tests=tests, unitless=False, xmloutput=xmloutput)
