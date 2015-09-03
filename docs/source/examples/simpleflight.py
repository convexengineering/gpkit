from gpkit.shortcuts import *
import gpkit
import numpy as np


pi     = Var("\\pi", np.pi, "-", "half of the circle constant")
rho    = Var("\\rho", 1.23, "kg/m^3", "density of air")
mu     = Var("\\mu", 1.78e-5, "kg/m/s", "viscosity of air")
S_wetratio = Var("(\\frac{S}{S_{wet}})", 2.05, "-", "wetted area ratio")
k      = Var("k", 1.2, "-", "form factor")
e      = Var("e", 0.95, "-", "Oswald efficiency factor")
N_ult  = Var("N_{ult}", 3.8, "-", "ultimate load factor")
tau    = Var("\\tau", 0.12, "-", "airfoil thickness to chord ratio")
C_Lmax = Var("C_{L,max}", 1.5, "-", "max CL with flaps down")
V_min  = Var("V_{min}", 22, "m/s", "takeoff speed")

if gpkit.units:
    CDA0 = Var("(CDA0)", 310.0, "cm^2", "fuselage drag area")
    W_0  = Var("W_0", 4.94, "kN", "aircraft weight excluding wing")
else:
    CDA0 = Var("(CDA0)", 0.031, "m^2", "fuselage drag area")
    W_0  = Var("W_0", 4940.0, "N", "aircraft weight excluding wing")

D   = Var("D", "N", "total drag force")
A   = Var("A", "-", "aspect ratio")
S   = Var("S", "m^2", "total wing area")
C_D = Var("C_D", "-", "Drag coefficient of wing")
C_L = Var("C_L", "-", "Lift coefficent of wing")
C_f = Var("C_f", "-", "skin friction coefficient")
Re  = Var("Re", "-", "Reynold's number")
W   = Var("W", "N", "total aircraft weight")
W_w = Var("W_w", "N", "wing weight")
V   = Var("V", "m/s", "cruising speed")

constraints = []

C_D_fuse = CDA0/S
C_D_wpar = k*C_f*S_wetratio
C_D_ind  = C_L**2/(pi*A*e)
constraints += [C_D >= C_D_fuse + C_D_wpar + C_D_ind]

if gpkit.units:
    W_w_strc = 8.71e-5*(N_ult*A**1.5*(W_0*W*S)**0.5)/tau / gpkit.units.m
    W_w_surf = (45.24*gpkit.units.Pa) * S
else:
    W_w_strc = 8.71e-5*(N_ult*A**1.5*(W_0*W*S)**0.5)/tau
    W_w_surf = 45.24 * S
constraints += [W_w >= W_w_surf + W_w_strc]

constraints += [D >= 0.5*rho*S*C_D*V**2,
                Re <= (rho/mu)*V*(S/A)**0.5,
                C_f >= 0.074/Re**0.2,
                W <= 0.5*rho*S*C_L*V**2,
                W <= 0.5*rho*S*C_Lmax*V_min**2,
                W >= W_0 + W_w]


def model():
    return gpkit.Model(D, constraints)


def sweep(n):
    substitutions = {V_min: ("sweep", np.linspace(20, 25, n)),
                     V: ("sweep", np.linspace(45, 55, n)), }
    return gpkit.Model(D, constraints, substitutions)


if __name__ == "__main__":
    import cProfile
    import pstats

    # Profilin"
    profile = cProfile.Profile()
    profile.enable()

    m = sweep(3)
    sol = m.solve()

    # Results
    profile.disable()
    ps = pstats.Stats(profile)
    ps.strip_dirs()
    ps.sort_stats("time")
    ps.print_stats(10)

    print(sol.table())
