"""
SIMPLE GP FOR AIRCRAFT DESIGN

The 'ipynb' folder has the same example as an iPython Notebook.
"""

from __future__ import print_function

import cProfile
import pstats

# Profilin'
profile = cProfile.Profile()
profile.enable()

import numpy as np

import gpkit

pi = gpkit.Monomial("\\pi", label="half of the circle constant")
CDA0 = gpkit.Monomial("(CDA0)", units="m^2", label="fuselage drag area")
rho = gpkit.Monomial("\\rho", units="kg/m^3", label="density of air")
mu = gpkit.Monomial("\\mu", units="kg*s/m", label="viscosity of air")
S_wetratio = gpkit.Monomial("(\\frac{S}{S_{wet}})", label="wetted area ratio")
k = gpkit.Monomial("k", label="form factor")
e = gpkit.Monomial("e", label="Oswald efficiency factor")
W_0 = gpkit.Monomial("W_0", units="N", label="aircraft weight excluding wing")
N_ult = gpkit.Monomial("N_{ult}", label="ultimate load factor")
tau = gpkit.Monomial("\\tau", label="airfoil thickness to chord ratio")
C_Lmax = gpkit.Monomial("C_{L,max}", label="max CL with flaps down")
V_min = gpkit.Monomial("V_{min}", units="m/s", label="takeoff speed")

substitutions = {
    pi: np.pi,
    CDA0: 0.031,
    rho: 1.23,
    mu: 1.78e-5,
    S_wetratio: 2.05,
    k: 1.2,
    e: 0.95,
    W_0: 4940,
    N_ult: 3.8,
    tau: 0.12,
    C_Lmax: 1.5,
    V_min: 22,
}

A = gpkit.Monomial("A", label="aspect ratio")
S = gpkit.Monomial("S", units="m^2", label="total wing area")
C_D = gpkit.Monomial("C_D", label="Drag coefficient of wing")
C_L = gpkit.Monomial("C_L", label="Lift coefficent of wing")
C_f = gpkit.Monomial("C_f", label="skin friction coefficient")
Re = gpkit.Monomial("Re", label="Reynold's number")
W = gpkit.Monomial("W", units="N", label="total aircraft weight")
W_w = gpkit.Monomial("W_w", units="N", label="wing weight")
V = gpkit.Monomial("V", units="m/s", label="cruising speed")

# drag modeling #
C_D_fuse = CDA0/S             # fuselage viscous drag
C_D_wpar = k*C_f*S_wetratio  # wing parasitic drag
C_D_ind = C_L**2/(pi*A*e)     # induced drag

# wing-weight modeling #
W_w_surf = 45.24*S                                    # surface weight
W_w_strc = 8.71e-5*(N_ult*A**1.5*(W_0*W*S)**0.5)/tau  # structural weight

gp = gpkit.GP(  # minimize
                0.5*rho*S*C_D*V**2,
                [   # subject to
                    Re <= (rho/mu)*V*(S/A)**0.5,
                    C_f >= 0.074/Re**0.2,
                    W <= 0.5*rho*S*C_L*V**2,
                    W <= 0.5*rho*S*C_Lmax*V_min**2,
                    W >= W_0 + W_w,
                    W_w >= W_w_surf + W_w_strc,
                    C_D >= C_D_fuse + C_D_wpar + C_D_ind
                ], substitutions)

sol = gp.solve()

# Results
profile.disable()
ps = pstats.Stats(profile)
ps.strip_dirs()
ps.sort_stats('time')
ps.print_stats(10)

sol.print_table()
