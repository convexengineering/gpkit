"""Standard Python setup script for gpkit"""
import os
from distutils.core import setup

LONG_DESCRIPTION = """
GPkit is a Python package for defining and manipulating
geometric programming models,
abstracting away the backend solver.
Supported solvers are
`MOSEK <http://mosek.com>`_
and `CVXopt <http://cvxopt.org/>`_.

`Documentation <http://gpkit.rtfd.org/>`_

`Citing GPkit <http://gpkit.rtfd.org/en/latest/citinggpkit.html>`_
"""

LICENSE = """The MIT License (MIT)

Copyright (c) 2018 Edward Burnell

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

# create blank settings file to replace anything cached
settings = os.sep.join([os.path.dirname(__file__), "gpkit", "env", "settings"])
try:
    with open(settings, "w") as f:
        f.write("installed_solvers :  ")
except IOError:
    pass

setup(
    name="gpkit",
    description="Package for defining and manipulating geometric "
                "programming models.",
    author="Edward Burnell",
    author_email="gpkit@mit.edu",
    url="https://www.github.com/convexengineering/gpkit",
    install_requires=["numpy >= 1.12.1", "pint >= 0.7", "scipy", "ad",
                      "ctypesgen", "cvxopt"],
    version="0.7.0.0",
    packages=["gpkit", "gpkit.tools", "gpkit.interactive", "gpkit.constraints",
              "gpkit.nomials", "gpkit.tests", "gpkit._mosek", "gpkit._pint"],
    package_data={"gpkit": ["env/settings"],
                  "gpkit._pint": ["*.txt"]},
    license=LICENSE,
    long_description=LONG_DESCRIPTION,
)
