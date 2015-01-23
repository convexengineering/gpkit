import gpkit

# A convenient shorthand to make code more readable
var = gpkit.Variable

# Parameters
alpha = var("alpha", 2, "-", "lower limit, wall aspect ratio")
beta = var("beta", 10, "-", "upper limit, wall aspect ratio")
gamma = var("gamma", 2, "-", "lower limit, floor aspect ratio")
delta = var("delta", 10, "-", "upper limit, floor aspect ratio")
A_wall = var("A_{wall}", 200, "m^2", "upper limit, wall area")
A_floor = var("A_{floor}", 50, "m^2", "upper limit, floor area")

# Decision variables
h = var("h", "m", "height")
w = var("w", "m", "width")
d = var("d", "m", "depth")

#Constraints
constraints = [A_wall >= 2*h*w + 2*h*d,
               A_floor >= w*d,
               h/w >= alpha,
               h/w <= beta,
               d/w >= gamma,
               d/w <= delta]

#Objective function
V = h*w*d
objective = 1/V #Standard form is to minimize the objective function

# Formulate the GP
gp = gpkit.GP(objective, constraints)

# Solve the GP
sol = gp.solve()

# Print results table
print sol.table()