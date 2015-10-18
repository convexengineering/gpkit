'''
Example 5, Xu 2013
Status: Currently generates error (AttributeError: 'int' object has no attribute 'exps')
NOTE: changing the objective to 1/x4 finds a solution
'''

from gpkit.shortcuts import *
from gpkit import SignomialsEnabled

x1 = Var('x1')
x2 = Var('x2')
x3 = Var('x3')
x4 = Var('x4')
x5 = Var('x5')
x6 = Var('x6')
t = Var('t')

k1 = 0.09755988
k2 = 0.99*k1
k3 = 0.0391908
k4 = 0.9*k3

with SignomialsEnabled():
    m = Model(t, # slightly different
             [t >= -x4,
              x1 + k1*x1*x5 <= 1,
              x1 + k1*x1*x5 >= 1,
              x2 + k2*x2*x6 >= x1,
              x2 + k2*x2*x6 <= x1,
              x3 + x1 + k3*x3*x5 >= 1,
              x3 + x1 + k3*x3*x5 <= 1,
              x4 + x2 + k4*x4*x6 <= x1 + x3,
              x4 + x2 + k4*x4*x6 >= x1 + x3,
              x5**0.5 + x6**0.5 <= 4,
              x5 >= 1E-5,
              x6 >= 1E-5,
              x1 <= 1,
              x2 <= 1,
              x3 <= 1,
              x4 <= 1,
              x5 <= 16,
              x6 <= 16
              ])

m.localsolve(algorithm="Xu")

