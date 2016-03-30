"""Define the Breakdown class"""
import numpy.testing as npt
from gpkit import Model, Variable
from gpkit import small_scripts
import svgwrite
from svgwrite import cm

class Breakdown(object):
    """
    model to make graphical breakdowns of weights, etc.
    takes input as a dict, variable arguments provided in a list. Also takes the
    chosen units in a string as a constructor argument. To run without units input
    None for varstring. Default diagram width is 12cm, height is 20cm. Use set_diagramHeight and set_diagramWidth
    to adjust these.

    To generate the diagrams you must have the package svgwrite installed. If you don't have,
    it can be installed in the terminal with the command pip install svgwrite
    """
    def __init__(self, input_dict, varstring):
        """
        class constructor - initialize all global variables, generate gpkit Vars
        """
        #define a variable to count number of "levels" in the dict, used for drawing
        self.levels = 0
        #create the list of variables to make constraints out of
        self.constr = []
        self.varlist = []
        #initialize varstring
        self.varstring = varstring
        #call recursive function to create gp constraints
        self.recurse(input_dict)
        #initialize sol for stability purposes
        self.sol = None
        self.dwg = None
        self.total = None
        self.elementlength = None
        #initialize input_dict
        self.input_dict = input_dict
        #set defualt parameters for the drawing
        self.sidelength = 12
        self.height = 20

    def solve_method(self):
        """
        method constructs gp constraints and objective, solves the model to
        determine the value of each weight
        """
        model = Model(self.make_objective(), self.make_constraints())
        self.sol = model.solve(verbosity=0)
        return self.sol

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

    def make_diagram(self):
        """
        method called to make the diagram - calls importsvgwrite from interactive to import
        svgwrite, calls all necessary follow on methods and sets importantvariables
        """
        #extract the total breakdown value for scaling purposes
        self.total = self.sol(self.input_dict.keys()[0])
        #depth of each breakdown level
        self.elementlength = (self.sidelength/self.levels)
        self.dwg = svgwrite.Drawing(filename="breakdown.svg", debug=True)
        self.dwgrecurse(self.input_dict, (2, 2), -1)
        #save the drawing at the conlusion of the recursive call
        self.dwg.save()

    def dwgrecurse(self, input_dict, initcoord, currentlevel):
        """
        #recursive function to divide widnow into seperate units to be drawn and calls the draw
        #function
        """
        order = input_dict.keys()
        i = 0
        totalheight = 0
        currentlevel = currentlevel+1
        while i < len(order):
            print input_dict
            print currentlevel
            height = ((self.height/self.total)*self.sol(order[i]))
            name = order[i]
            currentcoord = (initcoord[0], initcoord[1]+totalheight)
            self.drawsegment(name, height, currentcoord)
            totalheight = totalheight+height
            if isinstance(input_dict[order[i]], dict):
                #compute new initcoord
                newinitcoord = (initcoord[0]+self.elementlength,
                                initcoord[1]+totalheight-height)
                #recurse again
                self.dwgrecurse(input_dict[order[i]], newinitcoord,
                                currentlevel)
            #make sure all lines end at the same place
            elif currentlevel != self.levels:
                boundarylines = self.dwg.add(self.dwg.g(id='boundarylines',
                                                        stroke='black'))
                #top boudnary line
                boundarylines.add(self.dwg.line(
                    start=(currentcoord[0]*cm, currentcoord[1]*cm),
                    end=((currentcoord[0] +
                          (self.levels-currentlevel)*self.elementlength)*cm,
                         currentcoord[1]*cm)))
                #bottom boundary line
                boundarylines.add(self.dwg.line(
                    start=((currentcoord[0]+self.elementlength)*cm,
                           (currentcoord[1]+height)*cm),
                    end=((currentcoord[0] +
                          (self.levels-currentlevel)*self.elementlength)*cm,
                         (currentcoord[1]+height)*cm)))
            i = i+1

    def drawsegment(self, input_name, height, initcoord):
        """
        #function to draw each poriton of the diagram
        """
        lines = self.dwg.add(self.dwg.g(id='lines', stroke='black'))
        #draw the top horizontal line
        lines.add(self.dwg.line(start=(initcoord[0]*cm, initcoord[1]*cm), end=((initcoord[0]+
                        self.elementlength)*cm, initcoord[1]*cm)))
        #draw the bottom horizontaal line
        lines.add(self.dwg.line(start=(initcoord[0]*cm, (initcoord[1]+height)*cm),
                                end=((initcoord[0]+self.elementlength)*cm,
                                     (initcoord[1]+height)*cm)))
        #draw the vertical line
        lines.add(self.dwg.line(start=((initcoord[0])*cm, initcoord[1]*cm),
                                end=(initcoord[0]*cm, (initcoord[1]+height)*cm)))
        #adding in the breakdown namee
        writing = self.dwg.add(self.dwg.g(id='writing', stroke='black'))
        writing.add(svgwrite.text.Text(input_name, insert=None, x=[(.5+initcoord[0])*cm],
                                       y=[(height/2+initcoord[1])*cm], dx=None,
                                        dy=None, rotate=None))

    def set_diagram_width(self, sidelength):
        """
        method to set the width of the diagram, units are cm
        """
        self.sidelength = sidelength*cm

    def set_diagram_height(self, height):
        """
        method to set the total height of the diagram, units are cm
        """
        self.height = height*cm

    def test(self):
        """
        test method
        """
        npt.assert_almost_equal(small_scripts.mag(self.sol('w')), 15, decimal=5)
        npt.assert_almost_equal(small_scripts.mag(self.sol('w1')), 11, decimal=5)
        npt.assert_almost_equal(small_scripts.mag(self.sol('w2')), 3, decimal=5)
        npt.assert_almost_equal(small_scripts.mag(self.sol('w3')), 1, decimal=5)
        npt.assert_almost_equal(small_scripts.mag(self.sol('w11')), 3, decimal=5)
        npt.assert_almost_equal(small_scripts.mag(self.sol('w12')), 8, decimal=5)
        npt.assert_almost_equal(small_scripts.mag(self.sol('w121')), 2, decimal=5)
        npt.assert_almost_equal(small_scripts.mag(self.sol('w122')), 6, decimal=5)
        npt.assert_almost_equal(small_scripts.mag(self.sol('w21')), 1, decimal=5)
        npt.assert_almost_equal(small_scripts.mag(self.sol('w22')), 2, decimal=5)

        self.make_diagram()

if __name__ == "__main__":
    TEST = {'w': {'w1': {'w11':[3, "N"], 'w12':{'w121':[2, "N"], 'w122':[6, "N"]}},
                    'w2': {'w21':[1, "N"], 'w22':[2, "N"]}, 'w3':[1, "N"]}}
    BD = Breakdown(TEST, "N")
    BD.solve_method()
    BD.test()
