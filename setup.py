from __future__ import print_function

import os
import sys
import shutil

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

`Documentation <http://gpkit.readthedocs.org/>`_ \\
`Install instructions <http://gpkit.readthedocs.org/en/latest/installation.html>`_ \\
`Examples <http://gpkit.readthedocs.org/en/latest/examples.html>`_ \\
`Glossary <http://gpkit.readthedocs.org/en/latest/glossary.html>`_

"""

setup(
    name='gpkit',
    version='0.0',
    packages=['gpkit', 'gpkit._mosek', 'gpkit.tests', 'gpkit.interactive'],
    package_data={'gpkit': ['env/*'],
                  'gpkit._mosek': ['lib/*']},
    license=open('LICENSE').read(),
    long_description=open('README.md').read(),
)

# custom build script
if sys.argv[1] == "build":
    import build
