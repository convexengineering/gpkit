"""
SIMPLE GP FOR AIRCRAFT DESIGN

The "ipynb" folder has the same example as an iPython Notebook.
"""

from __future__ import print_function

import cProfile
import pstats

# Profilin"
profile = cProfile.Profile()
profile.enable()

import numpy as np

import gpkit
mon = gpkit.mon

pi = mon("\\pi", np.pi, "-", "half of the circle constant")
rho = mon("\\rho", 1.23, "kg/m^3", "density of air")
mu = mon("\\mu", 1.78e-5, "kg/m/s", "viscosity of air")
S_wetratio = mon("(\\frac{S}{S_{wet}})", 2.05, "-", "wetted area ratio")
k = mon("k", 1.2, "-", "form factor")
e = mon("e", 0.95, "-", "Oswald efficiency factor")
N_ult = mon("N_{ult}", 3.8, "-", "ultimate load factor")
tau = mon("\\tau", 0.12, "-", "airfoil thickness to chord ratio")
C_Lmax = mon("C_{L,max}", 1.5, "-", "max CL with flaps down")

if gpkit.units:
    CDA0 = mon("(CDA0)", 310.0, "cm^2", "fuselage drag area")
    W_0 = mon("W_0", 4.94, "kN", "aircraft weight excluding wing")
else:
    CDA0 = mon("(CDA0)", 0.031, "m^2", "fuselage drag area")
    W_0 = mon("W_0", 4940.0, "N", "aircraft weight excluding wing")

V_min = mon("V_{min}", np.linspace(20, 25, 3), "m/s", "takeoff speed")
V = mon("V", np.linspace(45, 55, 3), "m/s", "cruising speed")

A = mon("A", "-", "aspect ratio")
S = mon("S", "m^2", "total wing area")
C_D = mon("C_D", "-", "Drag coefficient of wing")
C_L = mon("C_L", "-", "Lift coefficent of wing")
C_f = mon("C_f", "-", "skin friction coefficient")
Re = mon("Re", "", "Reynold's number")
W = mon("W", "N", "total aircraft weight")
W_w = mon("W_w", "kN" if gpkit.units else "N", "wing weight")

# drag modeling #
C_D_fuse = CDA0/S             # fuselage viscous drag
C_D_wpar = k*C_f*S_wetratio  # wing parasitic drag
C_D_ind = C_L**2/(pi*A*e)     # induced drag

# wing-weight modeling #
if gpkit.units:
    W_w_strc = 8.71e-5*(N_ult*A**1.5*(W_0*W*S)**0.5)/tau / gpkit.units.m
    W_w_surf = (45.24*gpkit.units.Pa) * S
else:
    W_w_strc = 8.71e-5*(N_ult*A**1.5*(W_0*W*S)**0.5)/tau
    W_w_surf = 45.24 * S

gp = gpkit.GP(  # minimize
                0.5*rho*S*C_D*V**2,
                [   # subject to
                    Re <= (rho/mu)*V*(S/A)**0.5,
                    C_f >= 0.074/Re**0.2,
                    W <= 0.5*rho*S*C_L*V**2,
                    W <= 0.5*rho*S*C_Lmax*V_min**2,
                    W >= W_0 + W_w,
                    W_w >= W_w_surf + W_w_strc,
                    C_D >= C_D_fuse + C_D_wpar + C_D_ind, ])

sol = gp.solve()

# Results
profile.disable()
ps = pstats.Stats(profile)
ps.strip_dirs()
ps.sort_stats("time")
ps.print_stats(10)

print(sol.table())
