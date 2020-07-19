import pickle
...  # build the model
sol = m.solve()
# uncomment the line below to verify a new model
# sol.save("last_verified.sol")
last_verified_sol = pickle.load(open("last_verified.sol"))
if not sol.almost_equal(last_verified_sol, reltol=1e-3):
    print (last_verified_sol.diff(sol))

# Note you can replace the last three lines above with
print (sol.diff("last_verified.sol"))
# if you don't mind doing the diff in that direction.