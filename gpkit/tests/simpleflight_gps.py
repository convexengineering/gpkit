import numpy as np

import gpkit

pi = gpkit.Monomial("\\pi", value=np.pi, units="", label="half of the circle constant")
CDA0 = gpkit.Monomial("(CDA0)", value=310.0, units="cm^2", label="fuselage drag area")
rho = gpkit.Monomial("\\rho", value=1.23, units="kg/m^3", label="density of air")
mu = gpkit.Monomial("\\mu", value=1.78e-5, units="kg/m/s", label="viscosity of air")
S_wetratio = gpkit.Monomial("(\\frac{S}{S_{wet}})", value=2.05, units="", label="wetted area ratio")
k = gpkit.Monomial("k", value=1.2, units="", label="form factor")
e = gpkit.Monomial("e", value=0.95, units="", label="Oswald efficiency factor")
W_0 = gpkit.Monomial("W_0", value=4.94, units="kN", label="aircraft weight excluding wing")
N_ult = gpkit.Monomial("N_{ult}", value=3.8, units="", label="ultimate load factor")
tau = gpkit.Monomial("\\tau", value=0.12, units="", label="airfoil thickness to chord ratio")
C_Lmax = gpkit.Monomial("C_{L,max}", value=1.5, units="", label="max CL with flaps down")
V_min = gpkit.Monomial("V_{min}", units="m/s", label="takeoff speed")

D = gpkit.Monomial("D", units="N", label="total drag force")
A = gpkit.Monomial("A", units="", label="aspect ratio")
S = gpkit.Monomial("S", units="m^2", label="total wing area")
C_D = gpkit.Monomial("C_D", units="", label="Drag coefficient of wing")
C_L = gpkit.Monomial("C_L", units="", label="Lift coefficent of wing")
C_f = gpkit.Monomial("C_f", units="", label="skin friction coefficient")
Re = gpkit.Monomial("Re", units="", label="Reynold's number")
W = gpkit.Monomial("W", units="N", label="total aircraft weight")
W_w = gpkit.Monomial("W_w", units="N", label="wing weight")
V = gpkit.Monomial("V", units="m/s", label="cruising speed")

substitutions = {}
equations = []

C_D_fuse = CDA0/S
C_D_wpar = k*C_f*S_wetratio
C_D_ind = C_L**2/(pi*A*e)
equations += [C_D >= C_D_fuse + C_D_wpar + C_D_ind]


W_w_strc = 8.71e-5*(N_ult*A**1.5*(W_0*W*S)**0.5)/tau / gpkit.units.m
W_w_surf = 45.24*S * gpkit.units.parse_expression('N/m^2')
equations += [W_w >= W_w_surf + W_w_strc]

equations += [D >= 0.5*rho*S*C_D*V**2,
              Re <= (rho/mu)*V*(S/A)**0.5,
              C_f >= 0.074/Re**0.2,
              W <= 0.5*rho*S*C_L*V**2,
              W <= 0.5*rho*S*C_Lmax*V_min**2,
              W >= W_0 + W_w]


def single():
    substitutions.update({V_min: 22})
    return gpkit.GP(D, equations, substitutions)


def sweep():
    substitutions.update({
        V_min: ("sweep", np.linspace(20, 25, 10)),
        V: ("sweep", np.linspace(45, 55, 10)),
    })
    return gpkit.GP(D, equations, substitutions)
