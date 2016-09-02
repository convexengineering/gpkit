class M2new(Model):
    def __init__(self, **kwargs):
        
        #Make the necessary Variables
        y = Variable(‘y2’)
        z = Variable(‘z’)
        
        #make the constraints
        constraints = [
                       z >= 2,
                       y*z >= 1,
                       ]
            
                       #declare the objective
                       objective = z*y
                       
                       #construct the model
        Model.__init__(self, objective, constraints, **kwargs)