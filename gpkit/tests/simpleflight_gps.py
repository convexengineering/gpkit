import numpy as np

import gpkit

mon = gpkit.mon

pi = mon("\\pi", np.pi, "-", "half of the circle constant")
CDA0 = mon("(CDA0)", 310.0, "cm^2", "fuselage drag area")
rho = mon("\\rho", 1.23, "kg/m^3", "density of air")
mu = mon("\\mu", 1.78e-5, "kg/m/s", "viscosity of air")
S_wetratio = mon("(\\frac{S}{S_{wet}})", 2.05, "-", "wetted area ratio")
k = mon("k", 1.2, "-", "form factor")
e = mon("e", 0.95, "-", "Oswald efficiency factor")
W_0 = mon("W_0", 4.94, "kN", "aircraft weight excluding wing")
N_ult = mon("N_{ult}", 3.8, "-", "ultimate load factor")
tau = mon("\\tau", 0.12, "-", "airfoil thickness to chord ratio")
C_Lmax = mon("C_{L,max}", 1.5, "-", "max CL with flaps down")
V_min = mon("V_{min}", 22, "m/s", "takeoff speed")

D = mon("D", "N", "total drag force")
A = mon("A", "-", "aspect ratio")
S = mon("S", "m^2", "total wing area")
C_D = mon("C_D", "-", "Drag coefficient of wing")
C_L = mon("C_L", "-", "Lift coefficent of wing")
C_f = mon("C_f", "-", "skin friction coefficient")
Re = mon("Re", "-", "Reynold's number")
W = mon("W", "N", "total aircraft weight")
W_w = mon("W_w", "N", "wing weight")
V = mon("V", "m/s", "cruising speed")

equations = []

C_D_fuse = CDA0/S
C_D_wpar = k*C_f*S_wetratio
C_D_ind = C_L**2/(pi*A*e)
equations += [C_D >= C_D_fuse + C_D_wpar + C_D_ind]


W_w_strc = 8.71e-5*(N_ult*A**1.5*(W_0*W*S)**0.5)/tau / gpkit.units.m
W_w_surf = (45.24*gpkit.units.Pa) * S
equations += [W_w >= W_w_surf + W_w_strc]

equations += [D >= 0.5*rho*S*C_D*V**2,
              Re <= (rho/mu)*V*(S/A)**0.5,
              C_f >= 0.074/Re**0.2,
              W <= 0.5*rho*S*C_L*V**2,
              W <= 0.5*rho*S*C_Lmax*V_min**2,
              W >= W_0 + W_w]


def single():
    return gpkit.GP(D, equations)


def sweep():
    substitutions = {V_min: ("sweep", np.linspace(20, 25, 10)),
                     V: ("sweep", np.linspace(45, 55, 10)), }
    return gpkit.GP(D, equations, substitutions)
