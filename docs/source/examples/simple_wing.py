from gpkit.shortcuts import *
import numpy as np

# Define the constants in the problem
CDA0   = Var('(CDA_0)',            0.0306,  descr='Fuselage drag area')
C_Lmax = Var('C_{L,max}',          2.0,     descr='Maximum C_L, flaps down')
e      = Var('e',                  0.96,    descr='Oswald efficiency factor')
k      = Var('k',                  1.2,     descr='Form factor')
mu     = Var('\\mu',               1.78E-5, descr='Viscosity of air')
N_lift = Var('N_{lift}',           2.5,     descr='Ultimate load factor')
rho    = Var('\\rho',              1.23  ,  descr='Density of air')
Sw_S   = Var('\\frac{S_{wet}}{S}', 2.05,    descr='Wetted area ratio')
tau    = Var('\\tau',              0.12,    descr='Airfoil thickness-to-chord ratio')
V_min  = Var('V_{min}',            22,      descr='Desired landing speed')
W_0    = Var('W_0',                4940,    descr='Aircraft weight excluding wing')

# Define decision variables
A   = Var('A',   descr='Aspect ratio')
C_D = Var('C_D', descr='Drag coefficient')
C_L = Var('C_L', descr='Lift coefficient')
C_f = Var('C_f', descr='Skin friction coefficient')
S   = Var('S',   descr='Wing planform area')
Re  = Var('Re',  descr='Reynolds number')
W   = Var('Re',  descr='Total aircraft weight')
W_w = Var('W_w', descr='Wing weight')
V   = Var('V',   descr='Cruise velocity')

# Define objective function
objective = 0.5 * rho * V**2 * C_D * S

# Define constraints
constraints = [C_f * Re**0.2 >= 0.074,
               C_D >= CDA0 / S + k * C_f * Sw_S + C_L**2 / (np.pi * A * e),
               0.5 * rho * V**2 * C_L * S >= W,
               W >= W_0 + W_w,
               W_w >= 45.42*S + 8.71E-5 * N_lift * A**1.5 * (W_0 * W * S)**0.5 / tau,
               0.5 * rho * V_min**2 * C_Lmax * S >= W,
               Re == rho * V * (S/A)**0.5 / mu]

# Formulate the GP
gp = GP(objective, constraints)

# Solve the GP
sol = gp.solve()

# Print the solution table
print sol.table()
