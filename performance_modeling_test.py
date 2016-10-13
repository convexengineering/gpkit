"""Jungle Hawk Owl Concept"""
import numpy as np
from gpkit import Model, Variable, vectorize

class Aircraft(Model):
    "the JHO vehicle"

    def dynamic(self, state):
        """Creates an instance of this component's performance model,
        given a state"""
        return AircraftP(self, state)

    def __init__(self, **kwargs):
        self.fuse = Fuselage()
        self.wing = Wing()

        self.components = [self.fuse, self.wing]

        W = Variable("W", "lbf", "weight")
        self.weight = W
        constraints = [W >= sum(c["W"] for c in self.components)]

        Model.__init__(self, W, self.components + constraints, **kwargs)


class AircraftP(Model):
    """Aircraft Performance Model (i.e., aircraft flight physics)
    Currently implemented as lift = weight
    """
    def __init__(self, aircraft, state, **kwargs):
        self.aircraft = aircraft
        self.wing_aero = aircraft.wing.dynamic(state)
        self.perf_models = [self.wing_aero]
        constraints = [
            aircraft.weight <= (0.5*state["\\rho"]*state["V"]**2
                                * self.wing_aero["C_L"]
                                * aircraft.wing["S"])
            ]
        Model.__init__(self, None, self.perf_models + constraints, **kwargs)


class FlightState(Model):
    "Instantaneous context for evaluating flight physics"
    def __init__(self, **kwargs):
        V = Variable("V", 40, "knots", "true airspeed")
        mu = Variable("\\mu", 1.628e-5, "N*s/m^2", "dynamic viscosity")
        rho = Variable("\\rho", 0.74, "kg/m^3", "air density")
        constraints = [
            V == V,
            mu == mu,
            rho == rho
            ]
        Model.__init__(self, None, constraints, **kwargs)


class FlightSegment(Model):
    "Combines a context (flight state) and a component (the aircraft)"
    def __init__(self, aircraft, **kwargs):
        self.flightstate = FlightState()
        self.aircraftp = aircraft.dynamic(self.flightstate)
        Model.__init__(self, None,
                       [self.flightstate, self.aircraftp], **kwargs)


class Wing(Model):
    "The thing that creates the lift"
    def dynamic(self, state):
        """Creates an instance of this component's performance model,
        given a state"""
        return WingAero(self, state)

    def __init__(self, **kwargs):
        W = Variable("W", "lbf", "weight")
        S = Variable("S", 190, "ft^2", "surface area")
        rho = Variable("\\rho", 1, "lbf/ft^2", "areal density")
        A = Variable("A", 27, "-", "aspect ratio")
        c = Variable("c", "ft", "mean chord")

        constraints = [
            W >= S*rho,
            c == (S/A)**0.5
            ]
        super(Wing, self).__init__(None, constraints, **kwargs)


class WingAero(Model):
    "Wing drag model"
    def __init__(self, wing, state, **kwargs):
        CD = Variable("C_D", "-", "drag coefficient")
        CL = Variable("C_L", "-", "lift coefficient")
        e = Variable("e", 0.9, "-", "Oswald efficiency")
        Re = Variable("Re", "-", "Reynold's number")
        constraints = [
            CD >= (0.074/Re**0.2 + CL**2/np.pi/wing["A"]/e),
            Re == state["\\rho"]*state["V"]*wing["c"]/state["\\mu"],
            ]
        Model.__init__(self, None, constraints, **kwargs)


class Fuselage(Model):
    "The thing that carries the fuel, engine, and payload"
    def __init__(self, **kwargs):
        # fuselage needs an external dynamic drag model,
        # left as an exercise for the reader
        # V = Variable("V", 16, "gal", "volume")
        # d = Variable("d", 12, "in", "diameter")
        # S = Variable("S", "ft^2", "wetted area")
        # cd = Variable("c_d", .0047, "-", "drag coefficient")
        # CDA = Variable("CDA", "ft^2", "drag area")
        W = Variable("W", 100, "lbf", "weight")

        constraints = [  # CDA >= cd*4*V/d,
            W == W,  # todo replace with model
            ]

        super(Fuselage, self).__init__(None, constraints, **kwargs)


if __name__ == "__main__":
    JHO = Aircraft()
    N = 4
    with vectorize(N):
        FS = FlightSegment(JHO)
    COST = FS.aircraftp.wing_aero["C_D"].prod()**(1./N)
    M = Model(COST, [FS, JHO], {"V": np.linspace(20, 40, N)})
    SOL = M.solve("mosek")
    print SOL.table()
