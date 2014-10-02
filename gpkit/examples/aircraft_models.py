from math import pi
from gpkit import Monomial


# kind of awkward with this being a funciton...
# we could define the model outside a function, but then anyone importing it
# could mutate the (global!) model...
def steady_level_flight():
    lift = Monomial({'rho': 1, 'V': 2, 'C_L': 1, 'S': 1}, 0.5)
    drag = Monomial({'rho': 1, 'V': 2, 'C_D': 1, 'S': 1}, 0.5)
    Re_lim = Monomial({'rho': 1, 'mu': -1, 'V': 1, 'S': 0.5, 'A': -0.5})
    # probably want to define a ConstraintSet or GPModel class at some point
    # lists seem to work for now
    return [lift >= Monomial('W'),
            Monomial('D') >= drag,
            Monomial('Re') <= Re_lim]


def martins_wing_weight():
    surf = Monomial('S', 45.24)
    strc = Monomial({'N_ult': 1,
                     'A': 1.5,
                     'W': 0.5,
                     'W_0': 0.5,
                     'S': 0.5,
                     'tau': -1},
                    8.71e-5)
    return [Monomial('W_w') >= surf + strc]


def wing_drag():
    CDi = Monomial({'C_L': 2, 'e': -1, 'A': -1}, 1/pi)
    CDp = Monomial({'k': 1, 'S_wet_ratio': 1, 'Re': -0.2}, 0.074)
    return [Monomial('C_D_wing') >= CDi + CDp]


def fuselage_drag():
    return [Monomial('C_D_fuse') >= Monomial({'CDA0': 1, 'S': -1})]
