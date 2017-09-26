"Contains nomials, inequalities, and arrays"
from .array import NomialArray
from .core import Nomial
from .math import Monomial, Posynomial, Signomial
from .math import MonomialEquality, PosynomialInequality
from .math import SignomialInequality
from .math import SingleSignomialEquality
from .variables import Variable, ArrayVariable, VectorizableVariable
from .map import NomialMap
from .substitution import parse_subs

# TEMPORARY SHORTCUTS
from .data import NomialData
VectorVariable = ArrayVariable
