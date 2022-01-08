"An example to show off Breakdowns"
import os
import pickle
from gpkit.breakdowns import Breakdowns

# the code to create solar.p is in ./breakdowns/solartest.py
path = os.path.dirname(os.path.realpath(__file__))
sol = pickle.load(open(path + os.sep + "solar.p", "rb"))
bds = Breakdowns(sol)

print("Cost breakdown (you may be familiar with this from solution tables)")
print("==============")
bds.plot("cost")

print("Variable breakdowns (note the two methods of access)")
print("===================")
varkey, = sol["variables"].keymap[("Mission.FlightSegment.AircraftPerf"
                                   ".AircraftDrag.Poper")]
bds.plot(varkey)
bds.plot("AircraftPerf.AircraftDrag.MotorPerf.Q")

print("Combining the two above by increasing maxwidth")
print("----------------------------------------------")
bds.plot("AircraftPerf.AircraftDrag.Poper", maxwidth=105)

print("Model sensitivity breakdowns (note the two methods of access)")
print("============================")
bds.plot("model sensitivities")
bds.plot("Aircraft")

print("Exhaustive variable breakdown traces (and configuration arguments)")
print("====================================")
# often useful as a reference point when reading traces
bds.plot("AircraftPerf.AircraftDrag.Poper", height=12)
# includes factors, can be useful for reading traces as well
bds.plot("AircraftPerf.AircraftDrag.Poper", showlegend=True)
print("\nPermissivity = 2 (the default)")
print("----------------")
bds.trace("AircraftPerf.AircraftDrag.Poper")
print("\nPermissivity = 1 (stops at Pelec = v·i)")
print("----------------")
bds.trace("AircraftPerf.AircraftDrag.Poper", permissivity=1)

# you can also produce Plotly treemaps/icicle plots of your breakdowns
fig = bds.treemap("model sensitivities", returnfig=True)
fig = bds.icicle("cost", returnfig=True)
# uncommenting any of the below makes and shows the plot directly
# bds.icicle("model sensitivities")
# bds.treemap("cost")
