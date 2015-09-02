"""
A simple beam example with fixed geometry. Solves the discretized
Euler-Bernoulli beam equations for a constant distributed load
"""
import numpy as np
import matplotlib.pyplot as plt
from gpkit.shortcuts import *

def beam(N, L, EI, P):

    dx = L/(N-1)
    EI = Var("EI", EI)

    p = Vec(N, "p", label="Distributed load")
    p = p.sub(p, P*np.ones(N))

    V  = Vec(N, "V", label="Internal shear")
    M  = Vec(N, "M", label="Internal moment")
    th = Vec(N, "th", label="Slope")
    w  = Vec(N, "w", label="Displacement")

    eps = 1E-16 #an arbitrarily small positive number

    substitutions = {var: eps for var in [V[-1], M[-1], th[0], w[0]]}

    objective = w[-1]

    constraints = [V.left[1:N]     >= V[1:N]    + 0.5*dx*(p.left[1:N]    + p[1:N]),
                   M.left[1:N]     >= M[1:N]    + 0.5*dx*(V.left[1:N]    + V[1:N]),
                   th.right[0:N-1] >= th[0:N-1] + 0.5*dx*(M.right[0:N-1] + M[0:N-1])/EI,
                   w.right[0:N-1]  >= w[0:N-1]  + 0.5*dx*(th.right[0:N-1]+ th[0:N-1])
                  ]

    return Model(objective, constraints, substitutions)


if __name__ == "__main__":

    PLOT = True

    N = 10 #  [-] grid size
    L = 5. #   [m] beam length
    EI = 1E4 # [N*m^2] elastic modulus * area moment of inertia
    P = 100 #  [N/m] magnitude of distributed load 

    m = beam(N, L, EI, P)
    sol = m.solve(verbosity=1)

    x = np.linspace(0, L, N) # position along beam
    w_gp = sol("w") # deflection along beam
    w_exact =  P/(24.*EI)* x**2 * (x**2  - 4*L*x + 6*L**2) # analytical soln

    assert max(abs(w_gp - w_exact)) <= 1e-4

    if PLOT:
        plt.plot(x, w_gp, 'k', x, w_exact, 'b')
        plt.xlabel('x')
        plt.ylabel('Deflection')
        plt.axis('equal')
        plt.show()
