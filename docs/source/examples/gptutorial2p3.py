from gpkit import Variable, Model

# Declare decision variables
x = Variable('x')
y = Variable('y')
z = Variable('z')

# formulate the Model directly
prob = Model(y/x,      # objective
          [2 <= x,  # constraints
           x <= 3,
           x**2 + 3*y/z <= y**0.5,
           x/y == z**2])

# Solve the Model -- note, it's infeasible!
sol = prob.solve()

# Print results table
print sol.table()
