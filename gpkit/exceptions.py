"GPkit-specific Exception classes"
from . import DimensionalityError  # pylint: disable=unused-import

class MathematicallyInvalid(TypeError):
    "Raised whenever something violates a mathematical definition."

class InvalidPosynomial(MathematicallyInvalid):
    "Raised if a Posynomial would be created with a negative coefficient"

class InvalidGPConstraint(MathematicallyInvalid):
    "Raised if a non-GP-compatible constraint is used in a GP"

class InvalidSGPConstraint(MathematicallyInvalid):
    "Raised if a non-SGP-compatible constraint is used in an SGP"


class UnnecessarySGP(ValueError):
    "Raised if an SGP is fully GP-compatible"

class UnboundedGP(ValueError):
    "Raise if a GP is not fully bounded"


class InvalidLicense(RuntimeWarning):
    "Raised if a solver's license is missing, invalid, or expired."


class Infeasible(RuntimeWarning):
    "Raised if a model does not solve"

class UnknownInfeasible(Infeasible):
    "Raised if a model does not solve for unknown reasons"

class PrimalInfeasible(Infeasible):
    "Raised if a model returns a certificate of primal infeasibility"

class DualInfeasible(Infeasible):
    "Raised if a model returns a certificate of dual infeasibility"
