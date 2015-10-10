'''
Example 6, Xu 2013
Status: Finds correct solution
'''

from gpkit.shortcuts import *
from gpkit import SignomialsEnabled

x1 = Var('x1')
x2 = Var('x2')

with SignomialsEnabled():
    m = Model(x1,
             [0.25*x1 + 0.5*x2 - (1./16)*x1**2 - (1./16)*x2**2 <= 1,
              0.25*x1 + 0.5*x2 - (1./16)*x1**2 - (1./16)*x2**2 >= 1,
              (1./14)*x1**2 + (1./14)*x2**2 +1 >= (3./7)*x1 + (3./7)*x2, 
              (1./14)*x1**2 + (1./14)*x2**2 +1 <= (3./7)*x1 + (3./7)*x2,
              x1 >= 1,
              x2 >= 1,
              x1 <= 5.5,
              x2 <= 5.5])

sol = m.localsolve(rel_tol=1E-6)

