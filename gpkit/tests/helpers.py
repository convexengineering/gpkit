"""Convenience classes and functions for unit testing"""
import unittest
import sys
import os


def run_tests(tests, xmloutput=None, verbosity=2):
    """Default way to run tests, to be called in __main__ methods.

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
        import xmlrunner
        xmlrunner.XMLTestRunner(output=xmloutput).run(suite)
    else:
        unittest.TextTestRunner(verbosity=verbosity).run(suite)


class NullFile(object):
    """A fake file interface that does nothing"""
    def write(self, string):
        pass

    def close(self):
        pass


class StdoutCaptured(object):
    "Puts everything that would have printed to stdout in a log file instead"
    def __init__(self, logfilepath=None):
        self.logfilepath = logfilepath
        self.original_stdout = None

    def __enter__(self):
        self.original_stdout = sys.stdout
        logfile = (open(self.logfilepath, "w") if self.logfilepath
                   else NullFile())
        sys.stdout = logfile

    def __exit__(self, *args):
        sys.stdout.close()
        sys.stdout = self.original_stdout
