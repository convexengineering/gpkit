from __future__ import print_function

import os
import sys
import shutil

# custom build script
if sys.argv[1] in ["build", "install"]:
    import gpkit.build

from distutils.core import setup

long_description = """
GPkit
*****

GPkit is a Python package for defining and manipulating
geometric programming models,
abstracting away the backend solver.
Supported solvers are
`MOSEK <http://mosek.com>`_
and `CVXopt <http://cvxopt.org/>`_.

`Documentation <http://gpkit.readthedocs.org/>`_

`Install instructions <http://gpkit.readthedocs.org/en/latest/installation.html>`_

`Examples <http://gpkit.readthedocs.org/en/latest/examples.html>`_

`Glossary <http://gpkit.readthedocs.org/en/latest/glossary.html>`_

If you use GPkit, please cite it using the following bibtex::
```
@Misc{gpkit,
    author = {MIT Convex Optimization Group},
    title = {GPkit},
    howpublished = {\url{https://github.com/convexopt/gpkit}},
    year = {2015},
    note = {Release 0.1}
    }
```

"""

license = """The MIT License (MIT)

Copyright (c) 2015 MIT Convex Optimization Group

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

setup(
    name='gpkit',
    description='Package for defining and manipulating geometric programming models.',
    author='Convex Optimization Group at MIT ACDL',
    author_email='convex@mit.edu',
    url='https://www.github.com/convexopt/gpkit',
    install_requires=['numpy', 'ctypesgen>=0.r125'],
    version='0.1.0',
    packages=['gpkit', 'gpkit._mosek', 'gpkit.tests', 'gpkit.interactive'],
    package_data={'gpkit': ['env/*'],
                  'gpkit._mosek': ['lib/*']},
    license=license,
    long_description=long_description,
)
