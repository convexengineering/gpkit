"""Define the Breakdown class"""
import numpy.testing as npt
from gpkit import Model, Variable


class Breakdown(Model):
    def __init__(self, input_dict):
        """
        model to make graphical breakdowns of weights, etc.
        takes input as a dict, variable arguments provided in a list
        """
        #create the list of variables to make constraints out of
        self.constr = []
        self.varlist = []
        #call recursive function to create gp constraints
        total = self.recurse(input_dict)

        self.sol = self.solve_method()

    def getSolution(self):
        return self.sol

    def solve_method(self):
        m = Model(self.make_objective(), self.make_constraints())
        return m.solve(verbosity=0)

    def recurse(self, input_dict):
        order = input_dict.keys()
        i = 0
        hold = []
        while i < len(order):
            if isinstance(input_dict[order[i]], dict):
                #create the variable
                var = Variable(order[i], None)
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
                if len(input_dict[order[i]])==1:
                    var = Variable(order[i],input_dict[order[i]][0])
                elif len(input_dict[order[i]])==2:
                    var = Variable(order[i],input_dict[order[i]][0],input_dict[order[i]][1])
                elif len(input_dict[order[i]])==3:
                    var = Variable(order[i],input_dict[order[i]][0],input_dict[order[i]][1],input_dict[order[i]][1])
                elif len(input_dict[order[i]])==4:
                    var = Variable(order[i],input_dict[order[i]][1],input_dict[order[i]][2],input_dict[order[i]][3],input_dict[order[i]][4])

                self.varlist.append(var)
                hold.append(var)
            else:
                #create a var
                var = Variable(order[i], input_dict[order[i]])
                self.varlist.append(var)
                hold.append(var)
            i = i+1
        return hold

    #method to generate the gp constraints
    def make_constraints(self):
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
        #return the first variable that is created, this is what should be minimized
        return self.varlist[0]

    def test(self):
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
    TEST = {'w': {'w1': {'w11':[3], 'w12':{'w121':[2], 'w122':[6]}},
                  'w2': {'w21':[1], 'w22':[2]}, 'w3':[1]}}
    BD = Breakdown(TEST)
    BD.test()
