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
    if solvers is None:
        import gpkit
        solvers = [gpkit.settings["default_solver"]]
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
        # clear MODELNUMS to ensure determinate script-like output!
        for key in set(gpkit.MODELNUM_LOOKUP):
            del gpkit.MODELNUM_LOOKUP[key]

        with NewDefaultSolver(solver):
            testfn(name, import_dict, path)(self)

        # check all other global state besides MODELNUM_LOOKUP
        #   is falsy (which should mean blank)
        for globname, global_thing in [("models", gpkit.MODELS),
                                       ("modelnums", gpkit.MODELNUMS),
                                       ("vectorization", gpkit.VECTORIZATION),
                                       ("namedvars", gpkit.NAMEDVARS)]:
            if global_thing:
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
            if name not in imported:
                imported[name] = importlib.import_module(name)
            else:
                reload(imported[name])
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
    else:
        unittest.TextTestRunner(verbosity=verbosity).run(suite)


class NullFile(object):
    "A fake file interface that does nothing"
    def write(self, string):
        "Do not write, do not pass go."
        pass

    def close(self):
        "Having not written, cease."
        pass


class NewDefaultSolver(object):
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


class StdoutCaptured(object):
    "Puts everything that would have printed to stdout in a log file instead"
    def __init__(self, logfilepath=None):
        self.logfilepath = logfilepath
        self.original_stdout = None

    def __enter__(self):
        "Capture stdout"
        self.original_stdout = sys.stdout
        logfile = (open(self.logfilepath, "w") if self.logfilepath
                   else NullFile())
        sys.stdout = logfile

    def __exit__(self, *args):
        "Return stdout"
        sys.stdout.close()
        sys.stdout = self.original_stdout
