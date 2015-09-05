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
    def setup(self, N=10, L=5, EI=1e4, P=100):
        dx = Var("dx", L/float(N-1), "m", "Length of an element")
        EI = Var("EI", EI, "N*m^2")
        p  = Vec(N, "p", P*np.ones(N), "N/m", "Distributed load")
        V  = Vec(N, "V", "N", "Internal shear")
        M  = Vec(N, "M", "N*m", "Internal moment")
        th = Vec(N, "\\theta", "-", "Slope")
        w  = Vec(N, "w", "m", "Displacement")
        # shear and moment increase from tip to base (left > right)
        shear_eq = [V.left >= V + 0.5*dx*(p.left + p)]
        moment_eq = [M.left >= M + 0.5*dx*(V.left + V)]
        # theta and displacement decrease from tip to base (right > left)
        theta_eq = [th.right >= th + 0.5*dx*(M.right + M)/EI]
        displ_eq = [w.right >= w + 0.5*dx*(th.right + th)]
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
