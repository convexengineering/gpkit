"""
A simple beam example with fixed geometry. Solves the discretized
Euler-Bernoulli beam equations for a constant distributed load
"""
import numpy as np
from gpkit.shortcuts import *


class Beam(Model):
    """Discretization of the Euler beam equations for a distributed load.

    Arguments
    ---------
    N : int
        Number of finite elements that compose the beam.
    L : float
        [m] Length of beam.
    EI : float
        [N m^2] Elastic modulus times cross-section's area moment of inertia.
    P : float
        [N/m] Loading density.
    """
    def setup(self, N=4, L=5, EI=1e4, P=100):
        EI = Var("EI", EI, "N*m^2")
        dx = Var("dx", L/float(N-1), "m", "Length of an element")
        p = Vec(N-1, "p", [P]*(N-1), "N/m", "Distributed load per element")
        V = Vec(N, "V", "N", "Internal shear")
        M = Vec(N, "M", "N*m", "Internal moment")
        th = Vec(N, "\\theta", "-", "Slope")
        w = Vec(N, "w", "m", "Displacement")
        # tip loading and moment are 0
        V[-1]["value"], M[-1]["value"] = 0, 0
        # shear sums up the loads from tip to base
        shear_eq = [V.left >= V + (dx*p).padleft]
        # moment increases from tip to base
        moment_eq = [M.left >= M + 0.5*dx*(V + V.left)]
        # base slope and displacement are 0
        th[0]["value"], w[0]["value"] = 0, 0
        # slope and displacement increase from base to tip
        theta_eq = [th.right >= th + 0.5*dx*(M + M.right)/EI]
        displ_eq = [w.right >= w + 0.5*dx*(th + th.right)]
        # minimize tip displacement (the last w)
        return w[-1], [shear_eq, moment_eq, theta_eq, displ_eq]


N, L, EI, P = 10, 5, 1e4, 100
m = Beam(N, L, EI, P)
sol = m.solve(verbosity=1)
x = np.linspace(0, L, N)  # position along beam
w_gp = sol("w")  # deflection along beam
w_exact = P/(24.*EI) * x**2 * (x**2 - 4*L*x + 6*L**2)  # analytical soln

assert max(abs(w_gp - w_exact)) <= 1e-2

PLOT = False
if PLOT:
    import matplotlib.pyplot as plt
    x_exact = np.linspace(0, L, 1000)
    w_exact = P/(24.*EI) * x_exact**2 * (x_exact**2 - 4*L*x_exact + 6*L**2)
    plt.plot(x, w_gp, color='red', linestyle='solid', marker='^',
             markersize=8)
    plt.plot(x_exact, w_exact, color='blue', linestyle='dashed')
    plt.xlabel('x [m]')
    plt.ylabel('Deflection [m]')
    plt.axis('equal')
    plt.legend(['GP solution', 'Analytical solution'])
    plt.show()
