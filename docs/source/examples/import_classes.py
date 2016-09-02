from gpkit import Variable, Model

class M1(Model):
    def __init__(self, **kwargs):
        
        #Make the necessary Variables
        x = Variable("x")
        y = Variable("y")
                
        #make the constraints
        constraints = [
                       x >= 1,
                       x*y >= 0.5,
                       ]
            
        
        #declare the objective
        objective = x*y
        
        #construct the model
        Model.__init__(self, objective, constraints, **kwargs)


class M2(Model):
    def __init__(self, **kwargs):
        
        #Make the necessary Variables
        y = Variable("y")
        z = Variable("z")
                
        #make the constraints
        constraints = [
                       z >= 2,
                       y*z >= 1,
                       ]
            
        #declare the objective
        objective = z*y
            
        #construct the model
        Model.__init__(self, objective, constraints, **kwargs)

class M2new(Model):
    def __init__(self, **kwargs):
        
        #Make the necessary Variables
        y = Variable("y2")
        z = Variable("z")
        
        #make the constraints
        constraints = [
                       z >= 2,
                       y*z >= 1,
                       ]
            
        #declare the objective
        objective = z*y
       
        #construct the model
        Model.__init__(self, objective, constraints, **kwargs)
