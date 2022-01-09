"""Standard Python setup script for gpkit"""
import os
from distutils.core import setup


LICENSE = """The MIT License (MIT)

Copyright (c) 2020 Edward Burnell

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
THIS_DIR = os.path.dirname(__file__)
try:
    with open(os.sep.join([THIS_DIR, "gpkit", "env", "settings"]), "w") as f:
        f.write("installed_solvers :  ")
except IOError:
    pass

# read the README file
with open(os.path.join(THIS_DIR, "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="gpkit",
    description="Package for defining and manipulating geometric "
                "programming models.",
    author="Edward Burnell",
    author_email="gpkit@mit.edu",
    url="https://www.github.com/convexengineering/gpkit",
    python_requires=">=3.5.2",
    install_requires=["numpy >= 1.16.4", "pint >= 0.8.1", "plotly",
                      "scipy", "adce", "cvxopt >= 1.1.8",
                      "matplotlib"],
    version="1.1.0",
    packages=["gpkit", "gpkit.tools", "gpkit.interactive", "gpkit.constraints",
              "gpkit.nomials", "gpkit.tests", "gpkit.solvers"],
    package_data={"gpkit": ["env/settings"]},
    license=LICENSE,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
)
