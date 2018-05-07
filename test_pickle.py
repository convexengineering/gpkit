from gpkit import *

x = Variable("x", "m")
m = Model(x, [x >= units("ft")])
sol = m.solve()

import cPickle as pickle

with open("tm.p", "w") as f:
    pickle.dump(x.key, f)
with open("tm.p") as f:
    y = pickle.load(f)
assert y == x.key

with open("tm.p", "w") as f:
    pickle.dump(x, f)
with open("tm.p") as f:
    y = pickle.load(f)
assert y == x

with open("tm.p", "w") as f:
    pickle.dump(x**2 + units("acre"), f)
with open("tm.p") as f:
    y = pickle.load(f)
assert y == x**2 + units("acre")

with open("tm.p", "w") as f:
    pickle.dump(sol["variables"], f)
with open("tm.p") as f:
    y = pickle.load(f)
assert y == sol["variables"]

with open("tm.p", "w") as f:
    pickle.dump(sol, f)
with open("tm.p") as f:
    y = pickle.load(f)
assert y["cost"] == sol["cost"]
assert y["freevariables"] == sol["freevariables"]
assert y["variables"] == sol["variables"]
assert y["soltime"] == sol["soltime"]
assert y["constants"] == sol["constants"]
assert y["sensitivities"]["constants"] == sol["sensitivities"]["constants"]
assert all(y["sensitivities"]["la"] == sol["sensitivities"]["la"])
assert all(y["sensitivities"]["nu"] == sol["sensitivities"]["nu"])

with open("tm.p", "w") as f:
    pickle.dump(m, f)
with open("tm.p") as f:
    y = pickle.load(f)
y.solve()
