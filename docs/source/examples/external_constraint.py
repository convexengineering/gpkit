"Can be found in gpkit/docs/source/examples/external_constraint.py"
from gpkit.exceptions import InvalidGPConstraint
from external_function import external_code


class ExternalConstraint(object):
    "Class for external calling"
    varkeys = {}

    def __init__(self, x, y):
        # We need a GPkit variable defined to return in our constraint.  The
        # easiest way to do this is to read in the parameters of interest in
        # the initiation of the class and store them here.
        self.x = x
        self.y = y

    def as_posyslt1(self, _):
        "Ensures this is treated as an SGP constraint"
        raise InvalidGPConstraint("ExternalConstraint cannot solve as a GP.")

    def as_gpconstr(self, x0):
        "Returns locally-approximating GP constraint"

        # Unpacking the GPkit variables
        x = self.x
        y = self.y

        # Creating a default constraint for the first solve
        if not x0:
            return (y >= x)

        # Returns constraint updated with new call to the external code
        else:
            # Unpack Design Variables at the current point
            x_star = x0["x"]

            # Call external code
            res = external_code(x_star)

        # Return linearized constraint
        return (y >= res*x/x_star)
