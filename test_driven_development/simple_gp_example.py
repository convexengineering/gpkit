"""
SIMPLE GP FOR AIRCRAFT

This file is an example to test out the gpkit interface.

The file is heavily commented, not just because it's an example but
  also to show the potential for clear and self-explanatory modeling.

The file 'Simple GP for Aircraft.ipynb' in this folder has the same
  example but as an iPython Notebook (and with graphs).
"""

from numpy import linspace, zeros, pi
import gpkit


### INITALIZE ###
## constants ##
CDA0 = 0.03062702 # [m^2] fuselage drag area
rho = 1.23 # [kg/m^3] density of air
mu = 1.78e-5 # [kg/(m*s)] viscosity of air
S_wet_ratio = 2.05 # wetted area ratio
k = 1.2 # form factor
e = 0.96 # Oswald efficiency factor
W_0 = 4940 # [N] aircraft weight excluding wing
N_ult = 2.5 # ultimate load factor
tau = 0.12 # airfoil thickness to chord ratio
C_Lmax = 2.0 # max CL, flaps down


## free variables ##
varstr = 'A S  C_D  C_L  C_f  Re  W  W_w'
(A, # [-] aspect ratio
S, # [m^2] total wing area
C_D, # [-] Drag coefficient of wing
C_L, # [-] Lift coefficent of wing
C_f, # [-] skin friction coefficient
Re, # [-] Reynold's number
W, # [N] total aircraft weight
W_w # [N] wing weight
) = gpkit.monify(varstr)

# drag modeling
C_D_fuse = CDA0/S # fuselage viscous drag
C_D_wpar = k*C_f*S_wet_ratio # wing parasitic drag
C_D_ind = C_L**2/(pi*A*e) # induced drag

# wing-weight modeling
W_w_surf = 45.24*S # surface weight
W_w_strc = 8.71e-5*(N_ult*A**1.5*(W_0*W*S)**0.5)/tau # structural weight


## scan range and resolution ##
shape = (30,30)
V_range = linspace(45, 55, shape[0])
V_min_range = linspace(20, 25, shape[1])



### SOLVE ###
## ready the arrays ##
data = {}
for var in varstr.split():
  data[var] = zeros(shape)

## sweep through takeoff and cruising speeds ##
for i, V in enumerate(V_range):
  for j, V_min in enumerate(V_min_range):
    ## solve inside the loop ##
    sol = gpkit.minimize(
              0.5*rho*S*C_D*V**2, # [N] total drag force
            [ # subject to #
              Re <= (rho/mu)*V*(S/A)**0.5, # should be driven to equality
              C_f >= 0.074/Re**0.2, # fully turbulent boundary layer approx.
              W <= 0.5*rho*S*C_L*V**2, # cruising lift
              W <= 0.5*rho*S*C_Lmax*V_min**2, # takeoff lift
              W >= W_0 + W_w, # should be driven to equality
              W_w >= W_w_surf + W_w_strc, # see 'wing-weight modeling' above
              C_D >= C_D_fuse + C_D_wpar + C_D_ind # see 'drag modeling' above
            ],
              'cvxopt', {'show_progress': False})
    # save solution to array
    for var in sol:  data[var][i,j] = sol[var]