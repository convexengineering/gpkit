from gpkit import Variable, VectorVariable, GP
import numpy as np

N = 4

w_min = Variable("w_min", .25, "-", "Minimum Section Width")
w_max = Variable("w_max", 1, "-", "Maximum Section Width")
h_min = Variable("h_min", .25, "-", "Minimum Section Height")
h_max = Variable("h_max", 1, "-", "")
S_min = Variable("S_min", .1, "-", "Density of Air")
S_max = Variable("S_max", 3, "-", "Density of Air")
y_max = Variable("y_max", .05, "-", "Density of Air")

x_arr = np.arange(0, length, length/float(N))+1e-6

rho   = Variable("\\rho", 7750, "kg/m^3", "Density of Air")
Load  = Variable("Load", 100000, "N", "Tip Load")
L     = Variable("L", length, "m", "Beam Length")
x 	  = VectorVariable(N, "x", x_arr, "m", "Beam Location")
s_yld = Variable("s_yld", 250000000, "N/m^2", "Max Yield for Steel")

Volume 		= Variable("Volume", "m^3", "Beam Volume")
Mass 		= Variable("Mass", "kg", "Beam Mass")
RBM 		= VectorVariable("RBM", "N*m", "Root Bending Moment")
TBM 		= VectorVariable("TBM", "N*m", "Tip Bending Moment")

V 			= VectorVariable(N, "V", "N", "Shear")
M 			= VectorVariable(N, "M", "N*m", "Moment")
Sigma_max	= VectorVariable(N, "Sigma_max", "N/m^2", "Max Normal Stress")
Tau_max		= VectorVariable(N, "Tau_max", "N/m^2", "Max Shear Stress")
I 			= VectorVariable(N, "I", "m^4", "Moment of Inertia")
Q 			= VectorVariable(N, "Q", "m^3", "First Area Inertia")
d 			= VectorVariable(N, "d", "m", "Length of side of square cross section")

constraints = (V == Load,
			   [M[j-1] >= M[j] + V[j-1]*x[j-1] for j in range(1, N)],
			   M[N-1] == L/N*Load,
			   I == 1/12*d^4,
			   s_yld >= M*d/(2*I),
			   Q == 1/8*d^3,
			   s_yld/2 >= V*Q/(I*d),
			   Volume >= 1/2*d[0]^2*L + 1/2*d[N-1]^2*L,
			   Mass == rho*Volume
			  )

gp = GP(Mass, constraints)
sol = gp.solve(printing=False)
print sol(Mass)
print sol(d)