import os
import sys
import shutil

from distutils.core import setup

setup(
    name='gpkit',
    version='0.1dev',
    install_requires=['numpy', 'ctypesgen'],
    packages=['gpkit', 'gpkit._mosek', 'gpkit.tests'],
    package_data={'gpkit': ['env/*', 'examples/*'],
                  'gpkit._mosek': ['lib/*']},
    license=open('LICENSE').read(),
    long_description=open('README.md').read(),
)


def pathjoin(*args):
    return os.sep.join(args)

print "Building gpkit...\n"
settings = {}
envpath = pathjoin("gpkit", "env")
if os.path.isdir(envpath): shutil.rmtree(envpath)
os.makedirs(envpath)
print "Replaced the directory gpkit/env\n"

print "Attempting to find and build solvers:\n"
from build_solvers import installed_solvers
if not installed_solvers:
    sys.stderr.write("Can't find any solvers!\n")
    sys.exit(70)

# Choose default solver #
solver_priorities = ["mosek", "mosek_cli", "cvxopt"]
for solver in solver_priorities:
    if solver in installed_solvers:
        settings["defaultsolver"] = solver
        break
print "Choosing %s as the default solver" % settings["defaultsolver"]

# Write settings #
settingspath = envpath + os.sep + "settings"
with open(settingspath, "w") as f:
    for setting, value in settings.iteritems():
        f.write("%s %s\n" % (setting, value))
    f.write("\n")
