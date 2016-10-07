"""Jungle Hawk Owl Concept"""
import numpy as np
from gpkit import Model, Variable, vectorize

# pylint: disable=invalid-name


class Aircraft(Model):
    "the JHO vehicle"
    def __init__(self, **kwargs):
        self.dynamic_model = AircraftP
        self.fuse = Fuselage()
        self.wing = Wing()

        self.components = [self.fuse, self.wing]

        W = Variable("W_aircraft", "lbf", "weight")
        csr = [W >= sum(c["W"] for c in self.components)]

        super(Aircraft, self).__init__(W, self.components + csr, **kwargs)


class AircraftP(Model):
    def __init__(self, static, state, **kwargs):
        self.wing = static.wing.dynamic_model(static.wing, state)
        Model.__init__(self, None, [self.wing], **kwargs)


class FlightState(Model):
    "One chunk of a mission"
    def __init__(self, **kwargs):
        V = Variable("V", 40, "knots", "true airspeed")
        mu = Variable("\\mu", 1.628e-5, "N*s/m^2", "dynamic viscosity")
        rho = Variable("\\rho", 0.74, "kg/m^3", "air density")
        csr = [V == V,
               mu == mu,
               rho == rho]
        super(FlightState, self).__init__(None, csr, **kwargs)


class FlightSegment(Model):
    def __init__(self, N, aircraft, **kwargs):
        with vectorize(N):
            fs = FlightState()
            aircraftP = aircraft.dynamic_model(aircraft, fs)
            csr = [aircraft["W_aircraft"] <= (0.5*fs["\\rho"]*fs["V"]**2
                                              * aircraftP.wing["C_L"]
                                              * aircraft.wing["S"])]
        Model.__init__(self, aircraftP.wing["C_D"].prod()**(1./N),
                       [fs, aircraft, aircraftP, csr], **kwargs)


class Wing(Model):
    "The thing that creates the lift"
    def __init__(self, **kwargs):
        W = Variable("W", "lbf", "weight")
        S = Variable("S", 190, "ft^2", "surface area")
        rho = Variable("\\rho", 1, "lbf/ft^2", "areal density")
        A = Variable("A", 27, "-", "aspect ratio")
        c = Variable("c", "ft", "mean chord")
        self.dynamic_model = WingP

        csr = [W >= S*rho,
               c == (S/A)**0.5]
        super(Wing, self).__init__(None, csr, **kwargs)


class WingP(Model):
    def __init__(self, static, state, **kwargs):
        CD = Variable("C_D", "-", "drag coefficient")
        CL = Variable("C_L", "-", "lift coefficient")
        e = Variable("e", 0.9, "-", "Oswald efficiency")
        Re = Variable("Re", "-", "Reynold's number")
        csr = [CD >= (0.074/Re**0.2 + CL**2/np.pi/static["A"]/e),
               Re == state["\\rho"]*state["V"]*static["c"]/state["\\mu"],
               ]
        Model.__init__(self, None, csr, **kwargs)


class Fuselage(Model):
    "The thing that carries the fuel, engine, and payload"
    def __init__(self, **kwargs):
        V = Variable("V", 16, "gal", "volume")
        d = Variable("d", 12, "in", "diameter")
        # S = Variable("S", "ft^2", "wetted area")
        cd = Variable("c_d", .0047, "-", "drag coefficient")
        CDA = Variable("CDA", "ft^2", "drag area")
        W = Variable("W", 100, "lbf", "weight")

        csr = [  # CDA >= cd*4*V/d,
            W == W,  # todo replace with model
            ]

        super(Fuselage, self).__init__(None, csr, **kwargs)


if __name__ == "__main__":
    JHO = Aircraft()
    N = 4
    JHO = FlightSegment(N, JHO)
    JHO.substitutions["V"] = np.linspace(20, 40, N)
    # JHO.debug(solver="mosek")
    SOL = JHO.solve("mosek")
    print SOL.table()
