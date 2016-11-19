"Contains nomials, inequalities, and arrays"
from .array import NomialArray
from .nomial_core import Nomial
from .nomial_math import Monomial, Posynomial, Signomial
from .nomial_math import MonomialEquality, PosynomialInequality
from .nomial_math import SignomialInequality
from .nomial_math import SingleSignomialEquality
from .variables import Variable, ArrayVariable, VectorizableVariable
from .substitution import parse_subs

# TEMPORARY SHORTCUTS
from .data import NomialData
VectorVariable = ArrayVariable
