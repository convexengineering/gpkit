"""
A simple beam example with fixed geometry. Solves the discretized
Euler-Bernoulli beam equations for a constant distributed load
"""
import numpy as np
from gpkit import Variable, VectorVariable, Model, ureg
from gpkit.small_scripts import mag

eps = 2e-4   # has to be quite small for consistent cvxopt printouts;
             # normally you'd set this to something more like 1e-20


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

    Upper Unbounded
    ---------------
    w_tip
    """
    def setup(self, N=4):
        EI = Variable("EI", 1e4, "N*m^2")
        dx = Variable("dx", "m", "Length of an element")
        L = Variable("L", 5, "m", "Overall beam length")
        q = VectorVariable(N, "q", 100*np.ones(N), "N/m",
                           "Distributed load at each point")
        V = VectorVariable(N, "V", "N", "Internal shear")
        V_tip = Variable("V_{tip}", eps, "N", "Tip loading")
        M = VectorVariable(N, "M", "N*m", "Internal moment")
        M_tip = Variable("M_{tip}", eps, "N*m", "Tip moment")
        th = VectorVariable(N, "\\theta", "-", "Slope")
        th_base = Variable("\\theta_{base}", eps, "-", "Base angle")
        w = VectorVariable(N, "w", "m", "Displacement")
        w_base = Variable("w_{base}", eps, "m", "Base deflection")
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
        self.cost = self.w_tip = w[-1]
        return [shear_eq, moment_eq, theta_eq, displ_eq,
                L == (N-1)*dx]


b = Beam(N=6, substitutions={"L": 6, "EI": 1.1e4, "q": 110*np.ones(6)})
sol = b.solve(verbosity=0)
print sol.summary(maxcolumns=6)
w_gp = sol("w")  # deflection along beam

L, EI, q = sol("L"), sol("EI"), sol("q")
x = np.linspace(0, mag(L), len(q))*ureg.m  # position along beam
q = q[0]  # assume uniform loading for the check below
w_exact = q/(24.*EI) * x**2 * (x**2 - 4*L*x + 6*L**2)  # analytic soln

assert max(abs(w_gp - w_exact)) <= 1.1*ureg.cm

PLOT = False
if PLOT:
    import matplotlib.pyplot as plt
    x_exact = np.linspace(0, L, 1000)
    w_exact = q/(24.*EI) * x_exact**2 * (x_exact**2 - 4*L*x_exact + 6*L**2)
    plt.plot(x, w_gp, color='red', linestyle='solid', marker='^',
             markersize=8)
    plt.plot(x_exact, w_exact, color='blue', linestyle='dashed')
    plt.xlabel('x [m]')
    plt.ylabel('Deflection [m]')
    plt.axis('equal')
    plt.legend(['GP solution', 'Analytical solution'])
    plt.show()
