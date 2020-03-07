"GPkit-specific Exception classes"
from . import DimensionalityError  # pylint: disable=unused-import

class MathematicallyInvalid(TypeError):
    "Raised whenever something violates a mathematical definition."

class InvalidPosynomial(MathematicallyInvalid):
    "Raised when a Posynomial would be created with a negative coefficient"

class InvalidGPConstraint(MathematicallyInvalid):
    "Raised when a non-GP-compatible constraint is used in a GP"

class InvalidSGP(MathematicallyInvalid):
    "Raised when a non-GP-compatible constraint is used in a GP"


class UnnecessarySGP(ValueError):
    "Raised when an SGP is fully GP-compatible"

class UnboundedGP(ValueError):
    "Raise when a GP is not fully bounded"


class Infeasible(RuntimeWarning):
    "Raised when a model does not solve"

class UnknownInfeasible(Infeasible):
    "Raised when a model does not solve for unknown reasons"

class PrimalInfeasible(Infeasible):
    "Raised when a model returns a certificate of primal infeasibility"

class DualInfeasible(Infeasible):
    "Raised when a model returns a certificate of dual infeasibility"
