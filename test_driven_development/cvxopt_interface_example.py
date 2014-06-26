import numpy as np
import gpkit

# TODO(ned): check units? using that python package?

## INIT scan range and resolution
shape = (30,30)
V_range = np.linspace(45, 55, shape[0])
V_min_range = np.linspace(20, 25, shape[1])

## INIT constants
CDA0 = 0.03062702 # [m^2] fuselage drag area
rho = 1.23 # [kg/m^3] density of air
mu = 1.78e-5 # [kg/(m*s)] viscosity of air
S_wet = 2.05 # wetted area ratio
k = 1.2 # form factor
e = 0.96 # Oswald efficiency factor
W_0 = 4940 # [N] aircraft weight excluding wing
N_ult = 2.5 # ultimate load factor
tau = 0.12 # airfoil thickness to chord ratio
C_Lmax = 2.0 # max CL, flaps down
from numpy import pi

## INIT free variables 
varstr = 'A S  C_D  C_L  C_f  Re  W  W_w'
(
A, # [m^2]
S, 
C_D, 
C_L, 
C_f, 
Re, 
W, 
W_w
) = gpkit.monify(varstr)

data = {}
for var in varstr.split():
  data[var] = np.zeros(shape)

# SOLVE #############################################

for i, V in enumerate(V_range):
  for j, V_min in enumerate(V_min_range):
  	sol = gpkit.minimize(
              0.5*rho*C_D*S*V**2,
    	      [ # subject to
              Re <= rho*V/mu,
              C_f >= 0.074*Re**-0.2,
              C_D >= CDA0/S + k*C_f*S_wet/S + C_L**2/(pi*A*e),
              0.5*rho*V**2*C_L*S >= W,
              W >= W_0 + W_w,
              W_w >= 45.24*S + 8.71e-5*(N_ult*A**1.5*(W_0*W*S)**0.5)/tau,
              W <= 0.5*rho*C_Lmax*S*V_min**2
            ],
              'cvxopt', {'show_progress': False})

    for var in sol:
      data[var][i][j] = sol[var]

#####################################################

## PLOT INIT
%matplotlib inline
# Retina resolution, too!
import matplotlib.pyplot as plt

def contour_plot(title, Z, contour_format = '%2.1f', light='#80cdc1', dark='#01665e'):
  fig = plt.figure(figsize=(10,5))
  ax = fig.gca()
  
  contf = ax.contour(V_range, V_min_range, Z.T, 48,
                     linewidths = 0.5,
                     colors=light)
  cont = ax.contour(V_range, V_min_range, Z.T, 8,
                    linewidths = 1,
                    colors=dark)
  
  neutral = '#888888'
  ax.set_xlabel('cruise vel V (m/s)', color=neutral)
  ax.set_xlim(45, 55)
  ax.set_xticks(np.linspace(45,55,11),minor=True)
  ax.set_ylabel('landing vel V_min (m/s)', color=neutral)
  ax.set_ylim(20, 25)
  plt.clabel(cont, fmt = contour_format, colors=dark, fontsize=14)
  ax.set_title(title, color=dark, fontsize=14)
  ax.tick_params(colors=neutral)
  ax.set_frame_on(False)
  
  plt.show()

## PLOT GO
F_D = 0.5*rho*sol['S']*sol['C_D'] * (matrix(V_range)*np.ones((1,30)))**2
contour_plot('Total aircraft cruising drag, F_D [N]', F_D, '%2.0f')
contour_plot('Optimal wing area, S [m^2]', sol['S'])
contour_plot('Optimal aspect ratio, A [-]', sol['A'])
contour_plot('Optimal wing weight, W_w [N]', sol['W_w'], '%2.0f')