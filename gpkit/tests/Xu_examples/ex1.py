'''
Example 1, Xu 2013
Status: ValueError
'''

from gpkit.shortcuts import *
from gpkit import SignomialsEnabled

x1 = Var('x1')
x2 = Var('x2')
x3 = Var('x3')
t = Var('t')

with SignomialsEnabled():
    m = Model(t, # slightly different
             [t >= 0.05*x1*x2**-1 - x1 - 5*x2**-1,
              0.01*x2*x3**-1 + 0.01*x1 + 0.0005*x1*x3 <= 1,
              x1 >= 1,
              x2 >= 1,
              x3 >= 1,
              x1 <= 100,
              x2 <= 100,
              x3 <= 100])

m.localsolve()

