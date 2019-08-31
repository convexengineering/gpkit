"""Modular aircraft concept"""
import pickle
import numpy as np
from gpkit import Model, Vectorize, parse_variables


class AircraftP(Model):
    """Aircraft flight physics: weight <= lift, fuel burn

    Variables
    ---------
    Wfuel  [lbf]  fuel weight
    Wburn  [lbf]  segment fuel burn

    Upper Unbounded
    ---------------
    Wburn, aircraft.wing.c, aircraft.wing.A

    Lower Unbounded
    ---------------
    Wfuel, aircraft.W, state.mu

    """
    @parse_variables(__doc__, globals())
    def setup(self, aircraft, state):
        self.aircraft = aircraft
        self.state = state

        self.wing_aero = aircraft.wing.dynamic(aircraft.wing, state)
        self.perf_models = [self.wing_aero]

        W = aircraft.W
        S = aircraft.wing.S

        V = state.V
        rho = state.rho

        D = self.wing_aero.D
        CL = self.wing_aero.CL

        return {
            "lift":
                W + Wfuel <= 0.5*rho*CL*S*V**2,
            "fuel burn rate":
                Wburn >= 0.1*D,
            "performance":
                self.perf_models}


class Aircraft(Model):
    """The vehicle model

    Variables
    ---------
    W  [lbf]  weight

    Upper Unbounded
    ---------------
    W

    Lower Unbounded
    ---------------
    wing.c, wing.S
    """
    @parse_variables(__doc__, globals())
    def setup(self):
        self.fuse = Fuselage()
        self.wing = Wing()
        self.components = [self.fuse, self.wing]

        return {
            "definition of W":
                W >= sum(c.W for c in self.components),
            "components":
                self.components}

    dynamic = AircraftP


class FlightState(Model):
    """Context for evaluating flight physics

    Variables
    ---------
    V     40       [knots]    true airspeed
    mu    1.628e-5 [N*s/m^2]  dynamic viscosity
    rho   0.74     [kg/m^3]   air density

    """
    @parse_variables(__doc__, globals())
    def setup(self):
        pass


class FlightSegment(Model):
    """Combines a context (flight state) and a component (the aircraft)

    Upper Unbounded
    ---------------
    Wburn, aircraft.wing.c, aircraft.wing.A

    Lower Unbounded
    ---------------
    Wfuel, aircraft.W

    """
    def setup(self, aircraft):
        self.aircraft = aircraft

        self.flightstate = FlightState()
        self.aircraftp = aircraft.dynamic(aircraft, self.flightstate)

        self.Wburn = self.aircraftp.Wburn
        self.Wfuel = self.aircraftp.Wfuel

        return {"flightstate": self.flightstate,
                "aircraft performance": self.aircraftp}


class Mission(Model):
    """A sequence of flight segments

    Upper Unbounded
    ---------------
    aircraft.wing.c, aircraft.wing.A

    Lower Unbounded
    ---------------
    aircraft.W
    """
    def setup(self, aircraft):
        self.aircraft = aircraft

        with Vectorize(4):  # four flight segments
            self.fs = FlightSegment(aircraft)

        Wburn = self.fs.aircraftp.Wburn
        Wfuel = self.fs.aircraftp.Wfuel
        self.takeoff_fuel = Wfuel[0]

        return {
            "definition of Wburn":
                Wfuel[:-1] >= Wfuel[1:] + Wburn[:-1],
            "require fuel for the last leg":
                Wfuel[-1] >= Wburn[-1],
            "flight segment":
                self.fs}


class WingAero(Model):
    """Wing aerodynamics

    Variables
    ---------
    CD      [-]    drag coefficient
    CL      [-]    lift coefficient
    e   0.9 [-]    Oswald efficiency
    Re      [-]    Reynold's number
    D       [lbf]  drag force

    Upper Unbounded
    ---------------
    D, Re, wing.A, state.mu

    Lower Unbounded
    ---------------
    CL, wing.S, state.mu, state.rho, state.V
    """
    @parse_variables(__doc__, globals())
    def setup(self, wing, state):
        self.wing = wing
        self.state = state

        c = wing.c
        A = wing.A
        S = wing.S
        rho = state.rho
        V = state.V
        mu = state.mu

        return {
            "drag model":
                CD >= 0.074/Re**0.2 + CL**2/np.pi/A/e,
            "definition of Re":
                Re == rho*V*c/mu,
            "definition of D":
                D >= 0.5*rho*V**2*CD*S}


class Wing(Model):
    """Aircraft wing model

    Variables
    ---------
    W        [lbf]       weight
    S        [ft^2]      surface area
    rho    1 [lbf/ft^2]  areal density
    A     27 [-]         aspect ratio
    c        [ft]        mean chord

    Upper Unbounded
    ---------------
    W

    Lower Unbounded
    ---------------
    c, S
    """
    @parse_variables(__doc__, globals())
    def setup(self):
        return {"parametrization of wing weight":
                    W >= S*rho,
                "definition of mean chord":
                    c == (S/A)**0.5}

    dynamic = WingAero


class Fuselage(Model):
    """The thing that carries the fuel, engine, and payload

    A full model is left as an exercise for the reader.

    Variables
    ---------
    W  100 [lbf]  weight

    """
    @parse_variables(__doc__, globals())
    def setup(self):
        pass

AC = Aircraft()
MISSION = Mission(AC)
M = Model(MISSION.takeoff_fuel, [MISSION, AC])
print(M)
sol = M.solve(verbosity=0)
# save solution to some files
sol.savemat()
sol.savecsv()
sol.savetxt()
sol.save("solution.pkl")
# retrieve solution from a file
sol_loaded = pickle.load(open("solution.pkl", "rb"))

vars_of_interest = set(AC.varkeys)
# note that there's two ways to access submodels
assert (MISSION["flight segment"]["aircraft performance"]
        is MISSION.fs.aircraftp)
vars_of_interest.update(MISSION.fs.aircraftp.unique_varkeys)
vars_of_interest.add(M["D"])
print(sol.summary(vars_of_interest))
print(sol.table(tables=["loose constraints"]))

MISSION["flight segment"]["aircraft performance"]["fuel burn rate"] = (
    MISSION.fs.aircraftp.Wburn >= 0.2*MISSION.fs.aircraftp.wing_aero.D)
sol = M.solve(verbosity=0)
print(sol.diff("solution.pkl", showvars=vars_of_interest, sortbymodel=False))
