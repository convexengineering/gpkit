"""Define the Breakdown class"""
from .set import ConstraintSet
from .. import Variable

class Breakdown(ConstraintSet):
    """ConstraintSet of a summing tree. Can graph the breakdown of a solution.

    Arguments
    ---------
    input_dict : dictionary specifying variable names and values

    units : ;default units string, '-' for unitless operation

    filename: the name of the svg file produced, must end in .svg
    """
    def __init__(self, input_dict, units):
        self.input_dict = input_dict
        self.depth = 0
        self.units = units
        varlist, constraints = self._recurse(input_dict)
        self.root, = varlist
        ConstraintSet.__init__(self, constraints)

    def _recurse(self, input_dict):
        "Recursive function to generate gpkit Vars for each input weight"
        varlist = []
        constraints = []
        for key, value in sorted(input_dict.items()):
            if isinstance(value, dict):
                # there's another level to this breakdown
                var = Variable(key, self.units)
                # going down...
                self.depth += 1
                subvariables, subconstraints = self._recurse(value)
                # this depth's variable is greater than the sum of subvariables
                constraints.append(var >= sum(subvariables))
                # subvariables may have further depths
                constraints.extend(subconstraints)
            elif isinstance(value, list):
                var = Variable(key, value[0], value[1])
            else:
                var = Variable(key, value, self.units)
            varlist.append(var)
        return varlist, constraints

    def make_diagram(self, sol, filename, sidelength=12, height=20):
        """Make an SVG representation of the tree for a given solution.
        Default diagram width is 12cm, height is 20cm.
        """
        from ..interactive.svg import BDmake_diagram
        # set defualt parameters for the drawing
        BDmake_diagram(sol, self.depth, filename, sidelength, height, self.input_dict, self.units)
