from __future__ import print_function

import os
import sys
import shutil

from distutils.core import setup

setup(
    name='gpkit',
    version='0.6dev',
    install_requires=['numpy', 'ctypesgen'],
    packages=['gpkit', 'gpkit._mosek', 'gpkit.tests', 'gpkit.interactive'],
    package_data={'gpkit': ['env/*', 'examples/*'],
                  'gpkit._mosek': ['lib/*']},
    license=open('LICENSE').read(),
    long_description=open('README.md').read(),
)

# custom build script
import build
