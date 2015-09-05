"""
A simple beam example with fixed geometry. Solves the discretized
Euler-Bernoulli beam equations for a constant distributed load
"""
import numpy as np
from gpkit.shortcuts import *

def beam(N=10, L=5., EI=1E4, P=100):

    dx = Var("dx", L/(N-1), units="m")
    EI = Var("EI", EI, units="N*m^2")

    p = Vec(N, "p", units="N/m", label="Distributed load")
    p = p.sub(p, P*np.ones(N))

    V  = Vec(N, "V", units="N", label="Internal shear")
    M  = Vec(N, "M", units="N*m", label="Internal moment")
    th = Vec(N, "th", units="-", label="Slope")
    w  = Vec(N, "w", units="m", label="Displacement")

    eps = 1E-16 #an arbitrarily small positive number

    substitutions = {var: eps for var in [V[-1], M[-1], th[0], w[0]]}

    objective = w[-1]

    constraints = [V.left[1:N]     >= V[1:N]    + 0.5*dx*(p.left[1:N]    + p[1:N]),
                   M.left[1:N]     >= M[1:N]    + 0.5*dx*(V.left[1:N]    + V[1:N]),
                   th.right[0:N-1] >= th[0:N-1] + 0.5*dx*(M.right[0:N-1] + M[0:N-1])/EI,
                   w.right[0:N-1]  >= w[0:N-1]  + 0.5*dx*(th.right[0:N-1]+ th[0:N-1])
                  ]

    return Model(objective, constraints, substitutions)


N = 10 #  [-] grid size
L = 5. #   [m] beam length
EI = 1E4 # [N*m^2] elastic modulus * area moment of inertia
P = 100 #  [N/m] magnitude of distributed load

m = beam(N, L, EI, P)
sol = m.solve(verbosity=1)

x = np.linspace(0, L, N) # position along beam
w_gp = sol("w") # deflection along beam
w_exact =  P/(24.*EI)* x**2 * (x**2  - 4*L*x + 6*L**2) # analytical soln

assert max(abs(w_gp - w_exact)) <= 1e-2

PLOT = False
if PLOT:
    import matplotlib.pyplot as plt
    x_exact = np.linspace(0, L, 1000)
    w_exact =  P/(24.*EI)* x_exact**2 * (x_exact**2  - 4*L*x_exact + 6*L**2)
    plt.plot(x, w_gp, color='red', linestyle='solid', marker='^',
            markersize=8)
    plt.plot(x_exact, w_exact, color='blue', linestyle='dashed')
    plt.xlabel('x [m]')
    plt.ylabel('Deflection [m]')
    plt.axis('equal')
    plt.legend(['GP solution', 'Analytical solution'])
    plt.show()
