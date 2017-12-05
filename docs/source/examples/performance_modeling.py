"""Modular aircraft concept"""
import numpy as np
from gpkit import Model, Vectorize, parse_variables


class AircraftP(Model):
    """Aircraft flight physics: weight <= lift, fuel burn

    Variables
    ---------
    Wfuel  [lbf]  fuel weight
    Wburn  [lbf]  segment fuel burn

    Upper Bounded by aircraft
    -------------------------
    c, A

    Lower Bounded by aircraft
    -------------------------
    W

    Upper Unbounded
    ---------------
    Wburn

    Lower Unbounded
    ---------------
    Wfuel

    """
    def setup(self, aircraft, state):
        exec parse_variables(AircraftP.__doc__)
        self.aircraft = aircraft
        self.wing_aero = aircraft.wing.dynamic(aircraft.wing, state)
        self.perf_models = [self.wing_aero]

        W = self.W = aircraft.W
        self.c = aircraft.wing.c
        self.A = aircraft.wing.A
        S = aircraft.wing.S

        V = state.V
        rho = state.rho

        D = self.wing_aero.D
        CL = self.wing_aero.CL

        return [W + Wfuel <= 0.5*rho*CL*S*V**2,
                Wburn >= 0.1*D], self.perf_models


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
    c, S
    """
    def setup(self):
        exec parse_variables(Aircraft.__doc__)
        self.fuse = Fuselage()
        self.wing = Wing()
        self.components = [self.fuse, self.wing]
        self.c = self.wing.c
        self.S = self.wing.S

        return self.components, W >= sum(c.W for c in self.components)

    dynamic = AircraftP


class FlightState(Model):
    """Context for evaluating flight physics

    Variables
    ---------
    V    40        [knots]    true airspeed
    mu    1.628e-5 [N*s/m^2]  dynamic viscosity
    rho   0.74     [kg/m^3]   air density

    """
    def setup(self):
        exec parse_variables(FlightState.__doc__)


class FlightSegment(Model):
    """Combines a context (flight state) and a component (the aircraft)

    Upper Bounded by aircraft
    -------------------------
    c, A

    Lower Bounded by aircraft
    -------------------------
    W

    Upper Unbounded
    ---------------
    Wburn

    Lower Unbounded
    ---------------
    Wfuel

    """
    def setup(self, aircraft):
        self.flightstate = FlightState()
        self.aircraftp = aircraft.dynamic(aircraft, self.flightstate)

        self.aircraft = aircraft
        self.Wburn = self.aircraftp.Wburn
        self.Wfuel = self.aircraftp.Wfuel

        return self.flightstate, self.aircraftp


class Mission(Model):
    """A sequence of flight segments

    Upper Bounded by aircraft
    -------------------------
    c, A

    Lower Bounded by aircraft
    -------------------------
    W
    """
    def setup(self, aircraft):
        with Vectorize(4):  # four flight segments
            self.fs = FlightSegment(aircraft)

        self.aircraft = aircraft

        Wburn = self.fs.aircraftp.Wburn
        Wfuel = self.fs.aircraftp.Wfuel
        self.takeoff_fuel = Wfuel[0]

        return self.fs, [Wfuel[:-1] >= Wfuel[1:] + Wburn[:-1],
                         Wfuel[-1] >= Wburn[-1]]


class WingAero(Model):
    """Wing aerodynamics

    Variables
    ---------
    CD      [-]    drag coefficient
    CL      [-]    lift coefficient
    e   0.9 [-]    Oswald efficiency
    Re      [-]    Reynold's number
    D       [lbf]  drag force

    Upper Bounded by wing
    ---------------------
    A, c

    Upper Unbounded
    ---------------
    D

    Lower Unbounded
    ---------------
    CL, S
    """
    def setup(self, wing, state):
        exec parse_variables(WingAero.__doc__)
        self.wing = wing
        c = wing.c
        A = wing.A
        S = self.S = wing.S
        rho = state.rho
        V = state.V
        mu = state.mu
        return [
            CD >= 0.074/Re**0.2 + CL**2/np.pi/A/e,
            Re == rho*V*c/mu,
            D >= 0.5*rho*V**2*CD*S]


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
    def setup(self):
        exec parse_variables(Wing.__doc__)
        return [W >= S*rho, c == (S/A)**0.5]

    dynamic = WingAero


class Fuselage(Model):
    """The thing that carries the fuel, engine, and payload

    A full model is left as an exercise for the reader.

    Variables
    ---------
    W  100 [lbf]  weight

    """
    def setup(self):
        exec parse_variables(Fuselage.__doc__)

AC = Aircraft()
MISSION = Mission(AC)
M = Model(MISSION.takeoff_fuel, [MISSION, AC])
sol = M.solve(verbosity=0)

vars_of_interest = set(AC.varkeys)
vars_of_interest.update(MISSION.fs.aircraftp.unique_varkeys)
vars_of_interest.add("D")
print sol.summary(vars_of_interest)
