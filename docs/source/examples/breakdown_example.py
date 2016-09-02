from gpkit.constraints.breakdown import Breakdown
from gpkit import Model, units

inputdict = {"w": {"w1": 2, "w2": {"w21": 2, "w22": [1, "lbf"]}}}

#arguments to Breakdown are the input dict and default units
#in the breakdown
bd = Breakdown(inputdict, "N")

#bd.root is the highest level variable in a breakdown
#which is w in this case
m = Model(bd.root, bd)

sol = m.solve()

