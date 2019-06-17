"Contains nomials, inequalities, and arrays"
from .array import NomialArray
from .core import Nomial
from .data import NomialData
from .map import NomialMap
from .math import Monomial, Posynomial, Signomial
from .math import MonomialEquality, PosynomialInequality
from .math import SignomialInequality, SingleSignomialEquality
from .substitution import parse_subs
from .variables import Variable, ArrayVariable, VectorizableVariable

VectorVariable = ArrayVariable
