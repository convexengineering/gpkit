"""
Contains all svgwrite dependent methods
"""
import svgwrite
from svgwrite import cm

def BDmake_diagram(sol, depth, filename, sidelength, height, input_dict, units):
    """
    method called to make the diagram - calls importsvgwrite from interactive
    to import svgwrite, calls all necessary follow on methods and sets
    important variables

    Arguments
    ---------
    sol - the solution dict generated after solving the breakdown GP problem

    input_dict - a dict of values a breakdown is being generated for, sub levels are indicated
    by nesting in the dict. For exapmle, if w11 and w12 are subweights of w1 the input
    dict would look like {w1: {w11: 1, w12: 1}}

    depth - the total number of sub weights. The depth of the example above is 2. The
    depth of the breakdown with intpu sol {w1: {w11: {w111: 1, w112: 1}, w12: 1}} is 3.

    filename - the name of the svg file created, must end in .svg

    sidelength - the width of the svg file created in cm

    height - the height of the svg created in cm
    """
    #extract the total breakdown value for scaling purposes
    total = sol(input_dict.keys()[0])
    #depth of each breakdown level
    elementlength = (sidelength/depth)
    #generate a drawing objet
    dwg = svgwrite.Drawing(filename, debug=True)
    #call the recursive drawing function
    BDdwgrecurse(input_dict, (2, 2), -1, sol, elementlength, height, total,
                 depth, dwg, units)
    #save the drawing at the conlusion of the recursive call
    dwg.save()

    return dwg

def BDdwgrecurse(input_dict, initcoord, currentlevel, sol, elementlength,
                 sheight, total, depth, dwg, units):
    """
    recursive function to divide widnow into seperate units to be drawn and
    calls the draw function
    """
    totalheight = 0
    currentlevel = currentlevel+1
    for key, value in sorted(input_dict.items()):
        if units != '-':
            #make sure all quantities are in the default unit system
            height = (sheight/total)*sol(key).to(units)
        else:
            height = (sheight/total)*sol(key)
        #extract just the numeric value from the quantity
        height = height.item()
        #set the current coordinate
        currentcoord = (initcoord[0], initcoord[1]+totalheight)
        #draw the three lines for each element of the breakdown
        BDdrawsegment(key, height, currentcoord, dwg, elementlength)
        totalheight += height
        if isinstance(value, dict):
            #compute new initcoord
            newinitcoord = (initcoord[0]+elementlength,
                            initcoord[1]+totalheight-height)
            #recurse again
            BDdwgrecurse(value, newinitcoord, currentlevel, sol, elementlength,
                         sheight, total, depth, dwg, units)
        #make sure all lines end at the same place
        elif currentlevel != depth:
            boundarylines = dwg.add(dwg.g(id='boundarylines',
                                          stroke='black'))
            #top boudnary line
            boundarylines.add(dwg.line(
                start=(currentcoord[0]*cm, currentcoord[1]*cm),
                end=((currentcoord[0] +
                      (depth + 1 - currentlevel)*elementlength)*cm,
                     currentcoord[1]*cm)))
            #bottom boundary line
            boundarylines.add(dwg.line(
                start=((currentcoord[0] + elementlength)*cm,
                       (currentcoord[1] + height)*cm),
                end=((currentcoord[0] +
                      (depth + 1 - currentlevel)*elementlength)*cm,
                     (currentcoord[1]+height)*cm)))

def BDdrawsegment(input_name, height, initcoord, dwg, elementlength):
    """
    function to draw each poriton of the diagram
    """
    lines = dwg.add(dwg.g(id='lines', stroke='black'))
    #draw the top horizontal line
    lines.add(dwg.line(start=(initcoord[0]*cm, initcoord[1]*cm),
                       end=((initcoord[0]+elementlength)*cm, initcoord[1]*cm)))
    #draw the bottom horizontaal line
    lines.add(dwg.line(start=(initcoord[0]*cm, (initcoord[1]+height)*cm),
                       end=((initcoord[0]+elementlength)*cm,
                            (initcoord[1]+height)*cm)))
    #draw the vertical line
    lines.add(dwg.line(start=((initcoord[0])*cm, initcoord[1]*cm),
                       end=(initcoord[0]*cm, (initcoord[1]+height)*cm)))
    #adding in the breakdown namee
    writing = dwg.add(dwg.g(id='writing', stroke='black'))
    writing.add(svgwrite.text.Text(input_name,
                                   insert=None, x=[(.5+initcoord[0])*cm],
                                   y=[(float(height)/2+initcoord[1]+.125)*cm],
                                   dx=None, dy=None, rotate=None))
