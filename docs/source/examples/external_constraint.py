"Can be found in gpkit/docs/source/examples/external_constraint.py"
from gpkit.exceptions import InvalidGPConstraint
from external_function import external_code


class ExternalConstraint:
    "Class for external calling"

    def __init__(self, x, y):
        # We need a GPkit variable defined to return in our constraint.  The
        # easiest way to do this is to read in the parameters of interest in
        # the initiation of the class and store them here.
        self.x = x
        self.y = y

    def as_hmapslt1(self, _):
        "Ensures this is treated as an SGP constraint"
        raise InvalidGPConstraint("ExternalConstraint cannot solve as a GP.")

    def as_gpconstr(self, x0):
        "Returns locally-approximating GP constraint"
        # Creating a default constraint for the first solve
        if not x0:
            return (self.y >= self.x)
        # Otherwise calls external code at the current position...
        x_star = x0[self.x]
        res = external_code(x_star)
        # ...and returns a linearized constraint
        posynomial_constraint = (self.y >= res*self.x/x_star)
        posynomial_constraint.generated_by = self
        return posynomial_constraint
