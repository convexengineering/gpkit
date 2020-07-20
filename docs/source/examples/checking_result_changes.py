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
sol = m.solve()
# uncomment the line below to verify a new model
# sol.save("last_verified.sol")
last_verified_sol = pickle.load(open("last_verified.sol"))
last_verified_sol = pickle.load(open("last_verified.sol", mode="rb"))
if not sol.almost_equal(last_verified_sol, reltol=1e-3):
    print(last_verified_sol.diff(sol))

# Note you can replace the last three lines above with
print(sol.diff("last_verified.sol"))
# if you don't mind doing the diff in that direction.