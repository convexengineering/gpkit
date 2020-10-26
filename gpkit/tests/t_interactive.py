"""Tests for interactive plotting tools"""
import unittest
from gpkit.interactive.plotting import treemap
import gpkit


class TestPlotting(unittest.TestCase):
    """TestCase for gpkit.interactive.plotting"""
    def test_treemap(self):
        # m = Model() TODO
        fig = treemap(m)
        # fig.show(renderer="browser")


TESTS = [TestPlotting]


if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=wrong-import-position
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
