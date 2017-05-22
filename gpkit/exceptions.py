"GPkit-specific Exception classes"


class InvalidGPConstraint(Exception):
    "Raised when a non-GP-compatible constraint is used in a GP"
    pass
