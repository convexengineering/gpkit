"GP and SP modeling package"
__version__ = "0.7.0"

from .build import build
from ._pint import units, ureg, DimensionalityError
from .globals import settings
from .globals import SignomialsEnabled, SIGNOMIALS_ENABLED
from .globals import Vectorize, VECTORIZATION
from .globals import (NamedVariables, MODELS, MODELNUMS, MODELNUM_LOOKUP,
                      NAMEDVARS, reset_modelnumbers)
from .varkey import VarKey
from .nomials import Monomial, Posynomial, Signomial, NomialArray
from .nomials import VectorizableVariable as Variable
# NOTE above: the Variable the user sees is not the Variable used internally
from .nomials import VectorVariable, ArrayVariable
from .constraints.gp import GeometricProgram
from .constraints.sgp import SequentialGeometricProgram
from .constraints.sigeq import SignomialEquality
from .constraints.set import ConstraintSet
from .constraints.model import Model
from .tools.docstring import parse_variables

GPBLU = "#59ade4"
GPCOLORS = ["#59ade4", "#FA3333"]

if "just built!" in settings:
    from .tests.run_tests import run
    run(verbosity=1)
    print("""
GPkit is now installed with solver(s) %s
To incorporate new solvers at a later date, run `gpkit.build()`.

If any tests didn't pass, please post the output above
(starting from "Found no installed solvers, beginning a build.")
to gpkit@mit.edu or https://github.com/convexengineering/gpkit/issues/new
so we can prevent others from having these errors.

The same goes for any other bugs you encounter with GPkit:
send 'em our way, along with any interesting models, speculative features,
comments, discussions, or clarifications you feel like sharing.

Finally, we hope you find our documentation (https://gpkit.readthedocs.io/)
and engineering-design models (https://github.com/convexengineering/gplibrary/)
to be useful resources for your own applications.

Enjoy!
""" % settings["installed_solvers"])
