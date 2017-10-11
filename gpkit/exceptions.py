"GPkit-specific Exception classes"

from . import DimensionalityError  # pylint: disable=unused-import


class InvalidGPConstraint(Exception):
    "Raised when a non-GP-compatible constraint is used in a GP"
    pass
