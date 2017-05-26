"Contains nomials, inequalities, and arrays"
from .array import NomialArray
from .core import Nomial
from .math import Monomial, Posynomial, Signomial
from .math import MonomialEquality, PosynomialInequality
from .math import SignomialInequality
from .math import SingleSignomialEquality
from .variables import Variable, ArrayVariable, VectorizableVariable
from .map import parse_subs, NomialMap

# TEMPORARY SHORTCUTS
from .data import NomialData
VectorVariable = ArrayVariable
