"""Define the Breakdown class"""
from .set import ConstraintSet
from .. import Variable


class Breakdown(ConstraintSet):
    """ConstraintSet of a summing tree. Can graph the breakdown of a solution.

    Arguments
    ---------
    input_dict : dictionary specifying variable names and values
    units : string, None for unitless operation

    Default diagram width is 12cm, height is 20cm. Use set_diagramHeight and
    set_diagramWidth to adjust these.
    """
    def __init__(self, input_dict, units):
        #define a variable to count number of "levels" in the dict, used for drawing
        self.levels = 0
        #create the list of variables to make constraints out of
        self.constr = []
        self.varlist = []
        #initialize units
        self.units = units
        #call recursive function to create gp constraints
        self.recurse(input_dict)
        #initialize input_dict
        self.input_dict = input_dict
        #set defualt parameters for the drawing
        self.sidelength = 12
        self.height = 20

        constraints = [1 >= sum(constr[1])/constr[0] for constr in self.constr]
        ConstraintSet.__init__(self, constraints)

    def recurse(self, input_dict):
        "Recursive function to generate gpkit Vars for each input weight"
        order = input_dict.keys()
        i = 0
        hold = []
        while i < len(order):
            if isinstance(input_dict[order[i]], dict):
                #increment the number of "levels"
                self.levels = self.levels+1
                #create the variable
                var = Variable(order[i], self.units)
                self.varlist.append(var)
                #need to recurse again
                variables = self.recurse(input_dict[order[i]])
                j = 0
                varhold = 0
                while j < len(variables):
                    varhold = varhold+variables[j]
                    j = j+1
                hold.append(var)
                self.constr.append([var, variables])
            elif isinstance(input_dict[order[i]], list):
                #need to create a var of name dict entry
                var = Variable(order[i], *input_dict[order[i]])
                self.varlist.append(var)
                hold.append(var)
            else:
                #create a var
                var = Variable(order[i], input_dict[order[i]])
                self.varlist.append(var)
                hold.append(var)
            i = i+1
        return hold

    def make_diagram(self, sol):
        "Make an SVG representation of the tree for a given solution."
        from ..interactive.svg import make_diagram
        make_diagram(self, sol)
