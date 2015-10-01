'''
Example 4, Xu 2013
Status: t no upper bound, exp overflow, UNKNOWN status
'''

from gpkit.shortcuts import *
from gpkit import SignomialsEnabled

x1 = Var('x1')
x2 = Var('x2')
t = Var('t')

with SignomialsEnabled():
    m = Model(t, # slightly different
             [t >= x1**2 + x2**2 + 5 - 4*x1 - 2*x2,
              0.25*x1**2 + x2**2 <= 1,
              2*x2 - x1 <= 1,
              2*x2 - x1 >= 1
              ])

m.localsolve(algorithm="Xu")

