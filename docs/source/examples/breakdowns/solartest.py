from solar.solar import *
Vehicle = Aircraft(Npod=3, sp=True)
M = Mission(Vehicle, latitude=[20])
M.cost = M[M.aircraft.Wtotal]

M.localsolve().save("solar.p")
