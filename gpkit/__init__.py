"GP and SP modeling package"
__version__ = "0.7.0"

from ._pint import units, ureg, DimensionalityError
from .globals import settings
from .globals import SignomialsEnabled, SIGNOMIALS_ENABLED
from .globals import Vectorize, VECTORIZATION
from .globals import (NamedVariables, MODELS, MODELNUMS, MODELNUM_LOOKUP,
                      NAMEDVARS)
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
