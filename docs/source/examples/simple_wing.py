from gpkit.shortcuts import *
import numpy as np

# Define the constants in the problem
CDA0 = Var('(CDA_0)', 0.0306, 'm^2', label='Fuselage drag area')
C_Lmax = Var('C_{L,max}', 2.0, label='Maximum C_L, flaps down')
e = Var('e', 0.96, label='Oswald efficiency factor')
k = Var('k', 1.2, label='Form factor')
mu = Var('\\mu', 1.78E-5, 'kg/m/s', label='Viscosity of air')
N_lift = Var('N_{lift}', 2.5, label='Ultimate load factor')
rho = Var('\\rho', 1.23, 'kg/m^3', label='Density of air')
Sw_S = Var('\\frac{S_{wet}}{S}', 2.05, label='Wetted area ratio')
tau = Var('\\tau', 0.12, label='Airfoil thickness-to-chord ratio')
V_min = Var('V_{min}', 22, 'm/s', label='Desired landing speed')
W_0 = Var('W_0', 4940, 'N', label='Aircraft weight excluding wing')
cww1 = Var('cww1', 45.42, 'N/m^2', 'Wing weight area factor')
cww2 = Var('cww2', 8.71E-5, '1/m', 'Wing weight bending factor')

# Define decision variables
A = Var('A', label='Aspect ratio')
C_D = Var('C_D', label='Drag coefficient')
C_L = Var('C_L', label='Lift coefficient')
C_f = Var('C_f', label='Skin friction coefficient')
S = Var('S', 'm^2', label='Wing planform area')
Re = Var('Re', label='Reynolds number')
W = Var('W', 'N', label='Total aircraft weight')
W_w = Var('W_w', 'N', label='Wing weight')
V = Var('V', 'm/s', label='Cruise velocity')

# Define objective function
objective = 0.5 * rho * V**2 * C_D * S

# Define constraints
constraints = [C_f * Re**0.2 >= 0.074,
               C_D >= CDA0 / S + k * C_f * Sw_S + C_L**2 / (np.pi * A * e),
               0.5 * rho * V**2 * C_L * S >= W,
               W >= W_0 + W_w,
               W_w >= (cww1*S +
                       cww2 * N_lift * A**1.5 * (W_0 * W * S)**0.5 / tau),
               0.5 * rho * V_min**2 * C_Lmax * S >= W,
               Re == rho * V * (S/A)**0.5 / mu]

# Formulate the Model
m = Model(objective, constraints)

# Solve the Model
sol = m.solve()
