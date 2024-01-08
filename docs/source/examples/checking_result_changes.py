"Example code for solution saving and differencing."
import pickle
from gpkit import Model, Variable

# build model (dummy)
# decision variable
x = Variable("x")
y = Variable("y")

# objective and constraints
objective = 0.23 + x/y # minimize x and y
constraints = [x + y <= 5, x >= 1, y >= 2]

# create model
m = Model(objective, constraints)

# solve the model
# verbosity is 0 for testing's sake, no need to do that in your code!
sol = m.solve(verbosity=0)

# save the current state of the model
sol.save("last_verified.sol")

# uncomment the line below to verify a new model
with open("last_verified.sol", mode="rb") as f:
    last_verified_sol = pickle.load(f)
if not sol.almost_equal(last_verified_sol, reltol=1e-3):
    print(last_verified_sol.diff(sol))

# Note you can replace the last three lines above with
# print(sol.diff("last_verified.sol"))
# if you don't mind doing the diff in that direction.
