import svgwrite
from svgwrite import cm
"""
Contains all svgwrite dependent methods
"""

# TODO: remove all calls to bd attributes


def make_diagram(bd, sol):
    """
    method called to make the diagram - calls importsvgwrite from interactive
    to import svgwrite, calls all necessary follow on methods and sets
    important variables
    """
    #extract the total breakdown value for scaling purposes
    bd.total = sol(bd.input_dict.keys()[0])
    #depth of each breakdown level
    bd.elementlength = (bd.sidelength/bd.depth)
    bd.dwg = svgwrite.Drawing(filename="breakdown.svg", debug=True)
    dwgrecurse(bd, bd.input_dict, (2, 2), -1, sol)
    #save the drawing at the conlusion of the recursive call
    bd.dwg.save()


def dwgrecurse(bd, input_dict, initcoord, currentlevel, sol):
    """
    recursive function to divide widnow into seperate units to be drawn and
    calls the draw function
    """
    order = sorted(input_dict.keys())
    i = 0
    totalheight = 0
    currentlevel = currentlevel+1
    while i < len(order):
        height = int(round((((bd.height/bd.total)*sol(order[i])))))
        name = order[i]
        currentcoord = (initcoord[0], initcoord[1]+totalheight)
        drawsegment(bd, name, height, currentcoord)
        totalheight = totalheight+height
        if isinstance(input_dict[order[i]], dict):
            #compute new initcoord
            newinitcoord = (initcoord[0]+bd.elementlength,
                            initcoord[1]+totalheight-height)
            #print initcoord[1]+height
            #recurse again
            dwgrecurse(bd, input_dict[order[i]], newinitcoord,
                       currentlevel, sol)
        #make sure all lines end at the same place
        elif currentlevel != bd.depth:
            boundarylines = bd.dwg.add(bd.dwg.g(id='boundarylines',
                                                stroke='black'))
            #top boudnary line
            boundarylines.add(bd.dwg.line(
                start=(currentcoord[0]*cm, currentcoord[1]*cm),
                end=((currentcoord[0] +
                      (bd.depth-currentlevel)*bd.elementlength)*cm,
                     currentcoord[1]*cm)))
            #bottom boundary line
            boundarylines.add(bd.dwg.line(
                start=((currentcoord[0]+bd.elementlength)*cm,
                       (currentcoord[1]+height)*cm),
                end=((currentcoord[0] +
                      (bd.depth-currentlevel)*bd.elementlength)*cm,
                     (currentcoord[1]+height)*cm)))
        i = i+1


def drawsegment(bd, input_name, height, initcoord):
    """
    #function to draw each poriton of the diagram
    """
    lines = bd.dwg.add(bd.dwg.g(id='lines', stroke='black'))
    #draw the top horizontal line
    lines.add(bd.dwg.line(start=(initcoord[0]*cm, initcoord[1]*cm),
                          end=((initcoord[0]+bd.elementlength)*cm, initcoord[1]*cm)))
    #draw the bottom horizontaal line
    lines.add(bd.dwg.line(start=(initcoord[0]*cm, (initcoord[1]+height)*cm),
                          end=((initcoord[0]+bd.elementlength)*cm,
                               (initcoord[1]+height)*cm)))
    #draw the vertical line
    lines.add(bd.dwg.line(start=((initcoord[0])*cm, initcoord[1]*cm),
                          end=(initcoord[0]*cm, (initcoord[1]+height)*cm)))
    #adding in the breakdown namee
    writing = bd.dwg.add(bd.dwg.g(id='writing', stroke='black'))
    writing.add(svgwrite.text.Text(input_name,
                                   insert=None, x=[(.5+initcoord[0])*cm],
                                   y=[(float(height)/2+initcoord[1]+.125)*cm],
                                   dx=None, dy=None, rotate=None))
