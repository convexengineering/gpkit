"""
A simple beam example with fixed geometry. Solves the discretized
Euler-Bernoulli beam equations for a constant distributed load
"""
import numpy as np
import gpkit
from gpkit.shortcuts import Var, Vec, Model


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
    q : float or N-vector of floats
        [N/m] Loading density: can be specified as constants or as an array.
    """
    def setup(self, N=4):
        EI = Var("EI", 1e4, "N*m^2")
        dx = Var("dx", "m", "Length of an element")
        L = Var("L", 5, "m", "Overall beam length")
        q = Vec(N, "q", 100*np.ones(N), "N/m", "Distributed load at each point")
        V = Vec(N, "V", "N", "Internal shear")
        V_tip = Var("V_{tip}", 0, "N", "Tip loading")
        M = Vec(N, "M", "N*m", "Internal moment")
        M_tip = Var("M_{tip}", 0, "N*m", "Tip moment")
        th = Vec(N, "\\theta", "-", "Slope")
        th_base = Var("\\theta_{base}", 0, "-", "Base angle")
        w = Vec(N, "w", "m", "Displacement")
        w_base = Var("w_{base}", 0, "m", "Base deflection")
        # below: trapezoidal integration to form a piecewise-linear
        #        approximation of loading, shear, and so on
        # shear and moment increase from tip to base (left > right)
        shear_eq = (V >= V.right + 0.5*dx*(q + q.right))
        shear_eq[-1] = (V[-1] >= V_tip)  # tip boundary condition
        moment_eq = (M >= M.right + 0.5*dx*(V + V.right))
        moment_eq[-1] = (M[-1] >= M_tip)
        # slope and displacement increase from base to tip (right > left)
        theta_eq = (th >= th.left + 0.5*dx*(M + M.left)/EI)
        theta_eq[0] = (th[0] >= th_base)  # base boundary condition
        displ_eq = (w >= w.left + 0.5*dx*(th + th.left))
        displ_eq[0] = (w[0] >= w_base)
        # minimize tip displacement (the last w)
        return w[-1], [shear_eq, moment_eq, theta_eq, displ_eq, L == (N-1)*dx]


b = Beam(N=10, substitutions={"L": 6, "EI": 1.1e4, "q": 110*np.ones(10)})
sol = b.solve(verbosity=1)
w_gp = sol("w")  # deflection along beam

L, EI, q = sol("L"), sol("EI"), sol("q")
x = np.linspace(0, L, len(q))*gpkit.units.m  # position along beam
q = q[0]  # assume uniform loading for the check below
w_exact = q/(24.*EI) * x**2 * (x**2 - 4*L*x + 6*L**2)  # analytic soln

assert max(abs(w_gp - w_exact)) <= 1e-2*gpkit.units.m

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
