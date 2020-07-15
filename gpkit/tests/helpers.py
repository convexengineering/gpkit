"""Convenience classes and functions for unit testing"""
import unittest
import sys
import os
import importlib


def generate_example_tests(path, testclasses, solvers=None, newtest_fn=None):
    """
    Mutate TestCase class so it behaves as described in TestExamples docstring

    Arguments
    ---------
    path : str
        directory containing example modules to test
    testclass : class
        class that inherits from `unittest.TestCase`
    newtest_fn : function
        function that returns new tests. defaults to import_test_and_log_output
    solvers : iterable
        solvers to run for; or only for default if solvers is None
    """
    import_dict = {}
    if newtest_fn is None:
        newtest_fn = new_test
    tests = []
    for testclass in testclasses:
        if os.path.isdir(path):
            sys.path.insert(0, path)
        for fn in dir(testclass):
            if fn[:5] == "test_":
                name = fn[5:]
                old_test = getattr(testclass, fn)
                setattr(testclass, name, old_test)  # move to a non-test fn
                delattr(testclass, fn)  # delete the old old_test
                for solver in solvers:
                    new_name = "test_%s_%s" % (name, solver)
                    new_fn = newtest_fn(name, solver, import_dict, path)
                    setattr(testclass, new_name, new_fn)
        tests.append(testclass)
    return tests


def new_test(name, solver, import_dict, path, testfn=None):
    """logged_example_testcase with a NewDefaultSolver"""
    if testfn is None:
        testfn = logged_example_testcase

    def test(self):
        # pylint: disable=missing-docstring
        # No docstring because it'd be uselessly the same for each example

        import gpkit
        with NewDefaultSolver(solver):
            testfn(name, import_dict, path)(self)

        # clear modelnums to ensure deterministic script-like output!
        gpkit.globals.NamedVariables.reset_modelnumbers()

        # check all global state is falsy
        for globname, global_thing in [
                ("model numbers", gpkit.globals.NamedVariables.modelnums),
                ("lineage", gpkit.NamedVariables.lineage),
                ("signomials enabled", gpkit.SignomialsEnabled),
                ("vectorization", gpkit.Vectorize.vectorization),
                ("namedvars", gpkit.NamedVariables.namedvars)]:
            if global_thing:  # pragma: no cover
                raise ValueError("global attribute %s should have been"
                                 " falsy after the test, but was instead %s"
                                 % (globname, global_thing))
    return test


def logged_example_testcase(name, imported, path):
    """Returns a method for attaching to a unittest.TestCase that imports
    or reloads module 'name' and stores in imported[name].
    Runs top-level code, which is typically a docs example, in the process.

    Returns a method.
    """
    def test(self):
        # pylint: disable=missing-docstring
        # No docstring because it'd be uselessly the same for each example
        filepath = ("".join([path, os.sep, "%s_output.txt" % name])
                    if name not in imported else None)
        with StdoutCaptured(logfilepath=filepath):
            imported[name] = importlib.import_module(name)
        getattr(self, name)(imported[name])
    return test


def run_tests(tests, xmloutput=None, verbosity=2):
    """Default way to run tests, to be used in __main__.

    Arguments
    ---------
    tests: iterable of unittest.TestCase
    xmloutput: string or None
        if not None, generate xml output for continuous integration,
        with name given by the input string
    verbosity: int
        verbosity level for unittest.TextTestRunner
    """
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for t in tests:
        suite.addTests(loader.loadTestsFromTestCase(t))
    if xmloutput:
        import xmlrunner  # pylint: disable=import-error
        xmlrunner.XMLTestRunner(output=xmloutput).run(suite)
    else:  # pragma: no cover
        unittest.TextTestRunner(verbosity=verbosity).run(suite)


class NullFile:
    "A fake file interface that does nothing"
    def write(self, string):
        "Do not write, do not pass go."

    def close(self):
        "Having not written, cease."


class NewDefaultSolver:
    "Creates an environment with a different default solver"
    def __init__(self, solver):
        self.solver = solver
        self.prev_default_solver = None

    def __enter__(self):
        "Change default solver."
        import gpkit
        self.prev_default_solver = gpkit.settings["default_solver"]
        gpkit.settings["default_solver"] = self.solver

    def __exit__(self, *args):
        "Reset default solver."
        import gpkit
        gpkit.settings["default_solver"] = self.prev_default_solver


class StdoutCaptured:
    "Puts everything that would have printed to stdout in a log file instead"
    def __init__(self, logfilepath=None):
        self.logfilepath = logfilepath
        self.original_stdout = None
        self.original_unit_printing = None

    def __enter__(self):
        "Capture stdout"
        self.original_stdout = sys.stdout
        sys.stdout = (open(self.logfilepath, mode="w")
                      if self.logfilepath else NullFile())

    def __exit__(self, *args):
        "Return stdout"
        sys.stdout.close()
        sys.stdout = self.original_stdout
