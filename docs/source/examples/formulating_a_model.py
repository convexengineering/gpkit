objective = 1/(x*y*z)
constraints = [2*x*y + 2*x*z + 2*y*z <= S,
                x >= 2*y]
m = Model(objective, constraints)
