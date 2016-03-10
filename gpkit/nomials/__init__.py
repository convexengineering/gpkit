"Contains nomials, inequalities, and arrays"
from .array import NomialArray
from .nomial_core import Nomial
from .nomial_math import Monomial, Posynomial, Signomial
from .nomial_math import MonomialEquality, PosynomialInequality
from .nomial_math import SignomialInequality
from .variables import Variable, ArrayVariable

# TEMPORARY SHORTCUTS
VectorVariable = ArrayVariable
from .data import NomialData
