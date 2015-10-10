'''
Example 2, Xu 2013
Status: Approximately correct solution
'''

from gpkit.shortcuts import *
from gpkit import SignomialsEnabled

x1 = Var('x1')
x2 = Var('x2')
x3 = Var('x3')
x4 = Var('x4')
x5 = Var('x5')
x6 = Var('x6')
x7 = Var('x7')
x8 = Var('x8')
t = Var('t')

with SignomialsEnabled():
    m = Model(t, # slightly different
             [t >= 0.4*x1**0.67*x7**-0.67 + 0.4*x2**0.67*x8**-0.67 + 10 - x1 - x2, 
              0.0588*x5*x7 + 0.1*x1 <= 1,
              0.0588*x6*x8 + 0.1*x1 + 0.1*x2 <= 1,
              4*x3*x5**-1 + 2*x3**-0.71*x5**-1 + 0.0588*x3**-1.3*x7 <= 1,
              4*x4*x6**-1 + 2*x4**-0.71*x6**-1 + 0.0588*x4**-1.3*x8 <= 1,
              x1 >= 0.1,
              x2 >= 0.1,
              x3 >= 0.1,
              x4 >= 0.1,
              x5 >= 0.1,
              x6 >= 0.1,
              x7 >= 0.1,
              x8 >= 0.1,
              x1 <= 10,
              x2 <= 10,
              x3 <= 10,
              x4 <= 10,
              x5 <= 10,
              x6 <= 10,
              x7 <= 10,
              x8 <= 10])


sol = m.localsolve(rel_tol=1.E-6)

