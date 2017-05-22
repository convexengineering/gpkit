"Can be found in gpkit/docs/source/examples/external_constraint.py"
from gpkit import ConstraintSet
from gpkit.exceptions import InvalidGPConstraint
from external_function import external_code


class ExternalConstraint(ConstraintSet):
    "Class for external calling"
    # Overloading the __init__ function here permits the constraint class to be
    # called more cleanly at the top level GP.
    def __init__(self, x, y, **kwargs):

        # Calls the ConstriantSet __init__ function
        super(ExternalConstraint, self).__init__([], **kwargs)

        # We need a GPkit variable defined to return in our constraint.  The
        # easiest way to do this is to read in the parameters of interest in
        # the initiation of the class and store them here.
        self.x = x
        self.y = y

    # Prevents the ExternalConstraint class from solving in a GP, thus forcing
    # iteration
    def as_posyslt1(self, substitutions=None):
        raise InvalidGPConstraint("ExternalConstraint cannot solve as a GP.")

    # Returns the ExternalConstraint class as a GP compatible constraint when
    # requested by the GPkit solver
    def as_gpconstr(self, x0):

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
