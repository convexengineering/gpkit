from gpkit import ConstraintSet


class External_Constraint(ConstraintSet):
    # Overloading the __init__ function here permits the constraint class to be 
    # called more cleanly at the top level GP.
    def __init__(self, x):

        # Calls the ConstriantSet __init__ function
        super(External_Constraint,self).__init__([])

        # We need a GPkit variable defined to return in our constraint.  The 
        # easiest way to do this is to read in the parameters of interest in
        # the initiation of the class and store them here.
        self.x = x
    
    # Prevents the External_Constraint class from solving in a GP, thus forcing 
    # iteration
    def as_posyslt1(self):
        raise TypeError("External Constraint Model is not allowed to solve as a GP.")
        
    # Returns the External_Constraint class as a GP compatible constraint when 
    # requested by the GPkit solver
    def as_gpconstr(self, x0):

        # Unpacking the GPkit variables
        x = self.x
        
        # Creating a default constraint for the first solve
        if not x0:
            return (y >= x)
        
        # Returns constraint updated with new call to the external code
        else:
            # Unpack Design Variables at the current point
            x_star = x0["x"]

            # Call external code
            res = external_code(x_star)
            
        # Return linearized constraint
        return (y >= res*x/x_star)
