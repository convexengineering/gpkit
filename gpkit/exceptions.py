"GPkit-specific Exception classes"
from . import DimensionalityError  # pylint: disable=unused-import


class InvalidGPConstraint(Exception):
    "Raised when a non-GP-compatible constraint is used in a GP"


class InvalidPosynomial(Exception):
    "Raised when a Posynomial would be created with a negative coefficient"

class Infeasible(Exception):
    "Raised when a model does not solve"

class UnknownInfeasible(Infeasible):
    "Raised when a model does not solve for unknown reasons"

class PrimalInfeasible(Infeasible):
    "Raised when a model returns a certificate of primal infeasibility"

class DualInfeasible(Infeasible):
    "Raised when a model returns a certificate of dual infeasibility"

class UnboundedGP(Exception):
    "Raise when a GP is not fully bounded"
