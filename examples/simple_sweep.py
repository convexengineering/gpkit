"""
SIMPLE GP FOR AIRCRAFT DESIGN

The "ipynb" folder has the same example as an iPython Notebook.
"""

import cProfile
import pstats

# Profilin"
profile = cProfile.Profile()
profile.enable()

import numpy as np

import gpkit

pi = gpkit.Variable("\\pi", "half of the circle constant")
CDA0 = gpkit.Variable("(CDA0)", "m^2", "fuselage drag area")
rho = gpkit.Variable("\\rho", "kg/m^3", "density of air")
mu = gpkit.Variable("\\mu", "kg*s/m", "viscosity of air")
S_wetratio = gpkit.Variable("(\\frac{S}{S_{wet}})", "wetted area ratio")
k = gpkit.Variable("k", "form factor")
e = gpkit.Variable("e", "Oswald efficiency factor")
W_0 = gpkit.Variable("W_0", "N", "aircraft weight excluding wing")
N_ult = gpkit.Variable("N_{ult}", "ultimate load factor")
tau = gpkit.Variable("\\tau", "airfoil thickness to chord ratio")
C_Lmax = gpkit.Variable("C_{L,max}", "max CL with flaps down")
V_min = gpkit.Variable("V_{min}", "m/s", "takeoff speed")

substitutions = {
    "\\pi": np.pi,
    "(CDA0)": 0.031,
    "\\rho": 1.23,
    "\\mu": 1.78e-5,
    "(\\frac{S}{S_{wet}})": 2.05,
    "k": 1.2,
    "e": 0.95,
    "W_0": 4940,
    "N_{ult}": 3.8,
    "\\tau": 0.12,
    "C_{L,max}": 1.5,
    "V_{min}": ("sweep", np.linspace(20, 25, 10)),
    "V": ("sweep", np.linspace(45, 55, 10)),
}

A = gpkit.Variable("A", "aspect ratio")
S = gpkit.Variable("S", "m^2", "total wing area")
C_D = gpkit.Variable("C_D", "Drag coefficient of wing")
C_L = gpkit.Variable("C_L", "Lift coefficent of wing")
C_f = gpkit.Variable("C_f", "skin friction coefficient")
Re = gpkit.Variable("Re", "Reynold's number")
W = gpkit.Variable("W", "N", "total aircraft weight")
W_w = gpkit.Variable("W_w", "N", "wing weight")
V = gpkit.Variable("V", "m/s", "cruising speed")

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

data = gp.solve()

# Results
profile.disable()
ps = pstats.Stats(profile)
ps.strip_dirs()
ps.sort_stats("time")
ps.print_stats(10)

print "                 | Averages"
for key, table in data.iteritems():
    try:
        val = table.mean()
    except AttributeError:
        val = table
    descr = gp.var_descrs[key]
    if descr:
        if descr[0] is None:
            descr = "[-] %s" % descr[1]
        else:
            descr = "[%s] %s" % (descr[0], descr[1])
    print "%16s" % key, ": %-8.3g" % val, descr
print "                 |"
