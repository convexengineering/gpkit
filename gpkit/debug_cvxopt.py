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

V = 50
V_min = 25

posynomials = [ 0.5*rho*C_D*S*V**2,
	      Re <= rho*V/mu,
	      C_f >= 0.074*Re**-0.2,
	      C_D >= CDA0/S + k*C_f*S_wet/S + C_L**2/(pi*A*e),
	      0.5*rho*V**2*C_L*S >= W,
	      W >= W_0 + W_w,
	      W_w >= 45.24*S + 8.71e-5*(N_ult*A**1.5*(W_0*W*S)**0.5)/tau,
	      W <= 0.5*rho*C_Lmax*S*V_min**2
	    ]

# the first posynomial will be the cost function

from cvxopt import matrix, log, exp, solvers
from itertools import chain
solvers.options.update(options)

freevars = set().union(*[p.vars for p in posynomials])
monomials = list(chain(*[p.monomials for p in posynomials]))

#  See http://cvxopt.org/userguide/solvers.html?highlight=gp#cvxopt.solvers.gp
#    for more details on the format of these matrixes

# K: number of monomials (columns of F) present in each constraint
K = [len(p.monomials) for p in posynomials]
# g: constant coefficients of the various monomials in F
g = log(matrix([m.c for m in monomials]))
# F: exponents of the various control variables for each of the needed monomials
F = matrix([[float(m.exps.get(v, 0)) for m in monomials] for v in freevars])
