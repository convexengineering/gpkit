"""Allows for importing og svgwrite to make svg diagrams"""

import svgwrite

def importsvgwrite():
    """returns previously imported svgwrite"""
    return svgwrite

#THIS IS WHAT IT LOOKS LIKE IF I MOVE EVERYTHING WHICH TOUCHES THE DWG OBJECT OVER

"""def breakdown_drawing(name,total,elemlength,inputdict
        from svgwrite import cm
        dwg = svgwrite.Drawing(filename=name, debug=True)
                      
        def dwgrecurse(input_dict, initcoord, currentlevel):
           
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
                drawsegment(name, height, currentcoord)
                totalheight = totalheight+height
                if isinstance(input_dict[order[i]], dict):
                    #compute new initcoord
                    newinitcoord = (initcoord[0]+self.elementlength, initcoord[1]+totalheight-height)
                    #recurse again
                    dwgrecurse(input_dict[order[i]], newinitcoord, currentlevel)
                #make sure all lines end at the same place
                elif currentlevel != self.levels:
                    boundarylines = self.dwg.add(self.dwg.g(id='boundarylines', stroke='black'))
                    #top boudnary line
                    boundarylines.add(self.dwg.line(start=(currentcoord[0]*cm, currentcoord[1]*cm),
                                    end=((currentcoord[0]+(self.levels-currentlevel)*self.elementlength)
                                    *cm, currentcoord[1]*cm)))
                    #bottom boundary line
                    boundarylines.add(self.dwg.line(start=((currentcoord[0]+self.elementlength)*cm,
                                    (currentcoord[1]+height)*cm), end=((currentcoord[0]+(self.levels-
                                    currentlevel)*self.elementlength)*cm, (currentcoord[1]+height)*cm)))
                i = i+1

        def drawsegment(self, input_name, height, initcoord):
      
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


                      
        dwgrecurse(inputdict, (2, 2), -1)
        #save the drawing at the conlusion of the recursive call
        dwg.save()"""
