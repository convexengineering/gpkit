"""Modular aircraft concept"""
import numpy as np
from gpkit import Model, Variable, Vectorize


class Aircraft(Model):
    """The vehicle model

    Upper Unbounded
    ---------------
    W
    """
    def setup(self):
        self.fuse = Fuselage()
        self.wing = Wing()
        self.components = [self.fuse, self.wing]

        W = self.W = Variable("W", "lbf", "weight")

        return self.components, [
            W >= sum(c.topvar("W") for c in self.components)
            ]

    def dynamic(self, state):
        "This component's performance model for a given state."
        return AircraftP(self, state)


class AircraftP(Model):
    """Aircraft flight physics: weight <= lift, fuel burn

    Submodels
    ---------
    aircraft, state, wing_aero

    Upper Unbounded
    ---------------  # c will be bounded by Aircraft
    Wburn, c

    Lower Unbounded
    ---------------  # W will be bounded by Aircraft
    Wfuel, W
    """
    def setup(self, aircraft, state):
        self.aircraft = aircraft
        self.wing_aero = aircraft.wing.dynamic(state)
        self.perf_models = [self.wing_aero]
        Wfuel = self.Wfuel = Variable("W_{fuel}", "lbf", "fuel weight")
        Wburn = self.Wburn = Variable("W_{burn}", "lbf", "segment fuel burn")

        D = self.wing_aero["D"]
        Re = self.wing_aero["Re"]
        self.c = aircraft.wing.c
        W = self.W = aircraft.topvar("W")
        rho = state["\\rho"]
        V = state["V"]
        CL = self.CL = self.wing_aero["C_L"]
        S = aircraft.wing["S"]

        return self.perf_models, [
            W + Wfuel <= 0.5*rho*CL*S*V**2,
            Wburn >= 0.1*D
            ]

from gpkit.constraints.set import ConstraintSet

class FlightState(Model):
    "Context for evaluating flight physics"
    def setup(self):
        Variable("V", 40, "knots", "true airspeed")
        Variable("\\mu", 1.628e-5, "N*s/m^2", "dynamic viscosity")
        Variable("\\rho", 0.74, "kg/m^3", "air density")


class FlightSegment(Model):
    """Combines a context (flight state) and a component (the aircraft)

    Upper Unbounded
    ---------------  # c will be bounded by Aircraft
    c, Wburn

    Lower Unbounded
    ---------------  # W will be bounded by Aircraft
    W, Wfuel

    """
    def setup(self, aircraft):
        self.flightstate = FlightState()
        self.aircraftp = aircraft.dynamic(self.flightstate)

        self.c = aircraft.wing.c
        self.W = self.aircraftp.W
        self.Wburn = self.aircraftp.Wburn
        self.Wfuel = self.aircraftp.Wfuel

        return self.flightstate, self.aircraftp


class Mission(Model):
    """A sequence of flight segments

    Upper Unbounded
    ---------------  # c will be bounded by Aircraft
    c

    Lower Unbounded
    ---------------  # W will be bounded by Aircraft
    W
    """
    def setup(self, aircraft):
        with Vectorize(4):  # four flight segments
            self.fs = FlightSegment(aircraft)

        Wburn = self.fs.aircraftp["W_{burn}"]
        Wfuel = self.fs.aircraftp["W_{fuel}"]
        self.takeoff_fuel = Wfuel[0]
        self.W = aircraft.W
        self.c = aircraft.wing.c

        return self.fs, [Wfuel[:-1] >= Wfuel[1:] + Wburn[:-1],
                         Wfuel[-1] >= Wburn[-1]]


class Wing(Model):
    """Aircraft wing model

    Upper Unbounded
    ---------------
    W
    """
    def dynamic(self, state):
        "Returns this component's performance model for a given state."
        return WingAero(self, state)

    def setup(self):
        W = self.W = Variable("W", "lbf", "weight")
        S = Variable("S", 190, "ft^2", "surface area")
        rho = Variable("\\rho", 1, "lbf/ft^2", "areal density")
        A = Variable("A", 27, "-", "aspect ratio")
        c = self.c = Variable("c", "ft", "mean chord")

        return [W >= S*rho,
                c == (S/A)**0.5]


class WingAero(Model):
    """Wing aerodynamics


    Upper Unbounded
    ---------------  # c will be bounded by Aircraft
    D, c

    Lower Unbounded
    ---------------
    CL
    """
    def setup(self, wing, state):
        CD = Variable("C_D", "-", "drag coefficient")
        CL = self.CL = Variable("C_L", "-", "lift coefficient")
        e = Variable("e", 0.9, "-", "Oswald efficiency")
        Re = self.Re = Variable("Re", "-", "Reynold's number")
        D = self.D = Variable("D", "lbf", "drag force")
        self.c = wing.c

        return [
            CD >= (0.074/Re**0.2 + CL**2/np.pi/wing["A"]/e),
            Re == state["\\rho"]*state["V"]*wing["c"]/state["\\mu"],
            D >= 0.5*state["\\rho"]*state["V"]**2*CD*wing["S"],
            ]


class Fuselage(Model):
    "The thing that carries the fuel, engine, and payload"
    def setup(self):
        # fuselage needs an external dynamic drag model,
        # left as an exercise for the reader
        # V = Variable("V", 16, "gal", "volume")
        # d = Variable("d", 12, "in", "diameter")
        # S = Variable("S", "ft^2", "wetted area")
        # cd = Variable("c_d", .0047, "-", "drag coefficient")
        # CDA = Variable("CDA", "ft^2", "drag area")
        Variable("W", 100, "lbf", "weight")

AC = Aircraft()
MISSION = Mission(AC)
M = Model(MISSION.takeoff_fuel, [MISSION, AC])
sol = M.solve(verbosity=0)

vars_of_interest = set(AC.varkeys)
vars_of_interest.update(MISSION.fs.aircraftp.unique_varkeys)
vars_of_interest.add("D")
print sol.summary(vars_of_interest)
