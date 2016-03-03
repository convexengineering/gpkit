"""Define the Breakdown class"""
import numpy.testing as npt
from gpkit import Model, Variable

class Breakdown(Model):
    """
    model to make graphical breakdowns of weights, etc.
    takes input as a dict, variable arguments provided in a list. Also takes the
    chosen units in a string as a constructor argument. To run without units input
    None for varstring
    """
    def __init__(self, input_dict,varstring):
        """
        class constructor - initialize all global variables, generate gpkit Vars
        """
        #create the list of variables to make constraints out of
        self.constr = []
        self.varlist = []
        #initialize varstring
        self.varstring=varstring
        #call recursive function to create gp constraints
        self.recurse(input_dict)
        #initialize sol for stability purposes
        self.sol = None


    def solve_method(self):
        """
        method constructs gp constraints and objective, solves the model to
        determine the value of each weight
        """
        model = Model(self.make_objective(), self.make_constraints())
        self.sol = model.solve(verbosity=0)
        return self.sol

    def recurse(self, input_dict):
        """
        recursive function to generate gpkit Vars for each input weight
        """
        order = input_dict.keys()
        i = 0
        hold = []
        while i < len(order):
            if isinstance(input_dict[order[i]], dict):
                #create the variable
                var = Variable(order[i], self.varstring)
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

    def make_constraints(self):
        """
        method to generate gp constraints out of previously created Vars
        """
        i = 0
        constraints = []
        while i < len(self.constr):
            j = 0
            posy = 0
            while j < len(self.constr[i][1]):
                posy = posy+self.constr[i][1][j]
                j = j+1
            constraints.append(1 >= posy/self.constr[i][0])
            i += 1
        return constraints

    def make_objective(self):
        """
        return the first variable that is created, this is what should be minimized
        """
        return self.varlist[0]

    def test(self):
        """
        test method
        """
        npt.assert_almost_equal(self.sol('w'), 15, decimal=5)
        npt.assert_almost_equal(self.sol('w1'), 11, decimal=5)
        npt.assert_almost_equal(self.sol('w2'), 3, decimal=5)
        npt.assert_almost_equal(self.sol('w3'), 1, decimal=5)
        npt.assert_almost_equal(self.sol('w11'), 3, decimal=5)
        npt.assert_almost_equal(self.sol('w12'), 8, decimal=5)
        npt.assert_almost_equal(self.sol('w121'), 2, decimal=5)
        npt.assert_almost_equal(self.sol('w122'), 6, decimal=5)
        npt.assert_almost_equal(self.sol('w21'), 1, decimal=5)
        npt.assert_almost_equal(self.sol('w22'), 2, decimal=5)

if __name__ == "__main__":
    TEST = {'w': {'w1': {'w11':[3,"N"], 'w12':{'w121':[2,"N"], 'w122':[6,"N"]}},
                  'w2': {'w21':[1,"N"], 'w22':[2,"N"]}, 'w3':[1,"N"]}}
    BD = Breakdown(TEST, "N")
    BD.solve_method()
    BD.test()
