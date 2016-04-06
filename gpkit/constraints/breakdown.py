"""Define the Breakdown class"""
from .set import ConstraintSet
from .. import Variable


class Breakdown(ConstraintSet):
    """ConstraintSet of a summing tree. Can graph the breakdown of a solution.

    Arguments
    ---------
    input_dict : dictionary specifying variable names and values
    units : string, None for unitless operation

    """
    def __init__(self, input_dict, units):
        self.input_dict = input_dict  # TODO: linked-list tree? low priority.
        self.depth = 0
        varlist, constraints = self._recurse(input_dict, units)
        self.root, = varlist
        ConstraintSet.__init__(self, constraints)

    def _recurse(self, input_dict, units):
        "Recursive function to generate gpkit Vars for each input weight"
        varlist = []
        constraints = []
        for key, value in sorted(input_dict.items()):
            if isinstance(value, dict):
                # there's another level to this breakdown
                var = Variable(key, units)
                # going down...
                self.depth += 1
                subvariables, subconstraints = self._recurse(value, units)
                # this depth's variable is greater than the sum of subvariables
                constraints.append(var >= sum(subvariables))
                # subvariables may have further depths
                constraints.extend(subconstraints)
            elif isinstance(value, list):
                var = Variable(key, *value)
            else:
                var = Variable(key, value)
            varlist.append(var)
        return varlist, constraints

    def make_diagram(self, sol, sidelength=12, height=20):
        """Make an SVG representation of the tree for a given solution.

        Default diagram width is 12cm, height is 20cm.
        """
        from ..interactive.svg import make_diagram
        # set defualt parameters for the drawing
        self.sidelength = sidelength  # TODO:remove
        self.height = height  # TODO:remove
        make_diagram(self, sol)
