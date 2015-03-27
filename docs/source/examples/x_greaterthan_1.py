from gpkit import Variable, GP

# Decision variable
x = Variable('x')

# Constraint
constraints = [x >= 1]

# Objective (to minimize)
objective = x

# Formulate the GP
gp = GP(objective, constraints)

# Solve the GP
sol = gp.solve(printing=False)

# print selected results
print "Optimal cost:  %s" % sol['cost']
print "Optimal x val: %s" % sol(x)
