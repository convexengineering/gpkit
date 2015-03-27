from gpkit import Variable, GP

# Declare decision variables
x = Variable('x')
y = Variable('y')
z = Variable('z')

# formulate the GP directly
prob = GP(y/x,      # objective
          [2 <= x,  # constraints
           x <= 3,
           x**2 + 3*y/z <= y**0.5,
           x/y == z**2])

# Solve the GP -- note, it's infeasible!
sol = prob.solve()

# Print results table
print sol.table()
