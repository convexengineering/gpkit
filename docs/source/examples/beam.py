from gpkit.shortcuts import *
import numpy as np
import matplotlib.pyplot as plt

N = 100
L = 5.
dx = L/(N-1)
ei = 1E4
EI = Var("EI", ei)

P = 100.
p = Vec(N, "p", descr="Distributed load")
p = p.sub(p, P*np.ones(N))

V  = Vec(N, "V", label="Internal shear")
M  = Vec(N, "M", label="Internal moment")
th = Vec(N, "th", label="Slope")
w  = Vec(N, "w", label="Displacement")

eps = 1E-16 #something small

substitutions = {var: eps for var in [V[-1], M[-1], th[0], w[0]]}

objective = w[-1]

constraints = [V.left[1:N]     >= V[1:N]    + 0.5*dx*(p.left[1:N]    + p[1:N]),
               M.left[1:N]     >= M[1:N]    + 0.5*dx*(V.left[1:N]    + V[1:N]),
               th.right[0:N-1] >= th[0:N-1] + 0.5*dx*(M.right[0:N-1] + M[0:N-1])/EI,
               w.right[0:N-1]  >= w[0:N-1]  + 0.5*dx*(th.right[0:N-1]+ th[0:N-1])
              ]

m = Model(objective, constraints, substitutions)

sol = m.solve(verbosity=1)

x = np.linspace(0, L, N)
w_gp = sol("w")
w_exact =  P/(24.*ei)* x**2 * (x**2  - 4*L*x + 6*L**2)

assert max(abs(w_gp - w_exact)) <= 1e-4

plt.plot(x, w_gp, 'k', x, w_exact, 'b')
plt.xlabel('x')
plt.ylabel('Deflection')
plt.axis('equal')
plt.show()
