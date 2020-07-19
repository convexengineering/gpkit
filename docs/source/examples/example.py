from solar import *
Vehicle = Aircraft(Npod=1, sp = False)
M = Mission(Vehicle, latitude=[20])
M.cost = M[M.aircraft.Wtotal]
sol = M.solve()

from gpkit.interactive.sankey import Sankey
Sankey(M).diagram(M.aircraft.Wtotal)