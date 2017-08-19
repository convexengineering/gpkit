from gpkit import Model, parse_nomenclature, Variable
import numpy as np


class Fuselage(Model):
    """The thing that carries the fuel, engine, and payload

    Nomenclature
    ------------
    f                [-]             Fineness
    g          9.81  [m/s^2]         Standard gravity
    k                [-]             Form factor
    l                [ft]            Length
    mfac       2.0   [-]             Weight margin factor
    R                [ft]            Radius
    rhocfrp    1.6   [g/cm^3]        Density of CFRP
    rhofuel    6.01  [lbf/gallon]    Density of 100LL fuel
    S                [ft^2]          Wetted area
    t          0.024 [in]            Minimum skin thickness
    Vol              [ft^3]          Volume
    W                [lbf]           Weight
    """

    def setup(self, Wfueltot):
        exec parse_nomenclature(self.__doc__)
        return [
            f == l/R/2,
            k >= 1 + 60/f**3 + f/400,
            3*(S/np.pi)**1.6075 >= 2*(l*R*2)**1.6075 + (2*R)**(2*1.6075),
            Vol <= 4*np.pi/3*(l/2)*R**2,
            Vol >= Wfueltot/rhofuel,
            W/mfac >= S*rhocfrp*t*g,
        ]

Fuselage(Variable("Wfueltot", 5, "lbf"))
