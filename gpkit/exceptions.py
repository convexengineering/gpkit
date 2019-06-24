"GPkit-specific Exception classes"
from __future__ import unicode_literals

from . import DimensionalityError  # pylint: disable=unused-import


class InvalidGPConstraint(Exception):
    "Raised when a non-GP-compatible constraint is used in a GP"
    pass


class InvalidPosynomial(Exception):
    "Raised when a Posynomial would be created with a negative coefficient"
    pass
