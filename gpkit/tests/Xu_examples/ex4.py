'''
Example 4, Xu 2013
Status: Removing the <= part of == constraint yields the exact solution
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
              2*x2 - x1 >= 1
              ])

sol = m.localsolve()

