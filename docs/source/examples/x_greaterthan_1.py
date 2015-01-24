from gpkit import Variable, GP

# Decision variable
x = Variable("x", "m", "A really useful variable called x with units of meters")

# Constraint
constraint = [1/x <= 1]

# Objective (to minimize)
objective = x

# Formulate the GP
gp = GP(objective, constraint)

# Solve the GP
sol = gp.solve()

# Print results table
print sol.table()