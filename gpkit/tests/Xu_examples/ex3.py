'''
Example 3, Xu 2013
Status: (very) approximately correct solution
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

with SignomialsEnabled():
    m = Model(x1 + x2 + x3, 
             [833.33252*x1**-1*x4*x6**-1 + 100*x6**-1 - 83333.333*x1**-1*x6**-1 <= 1,
              1250*x2**-1*x5*x7**-1 + x4*x7**-1 - 1250*x2**-1*x4*x7**-1 <= 1,
              1250000*x3**-1*x8**-1 + x5*x8**-1 - 2500*x3**-1*x5*x8**-1 <= 1,
              0.0025*x4 + 0.0025*x6 <= 1,
              -0.0025*x4 + 0.0025*x5 + 0.0025*x7 <= 1,
              0.01*x8 - 0.01*x5 <= 1,
              x1 >= 100,
              x1 <= 10000,
              x2 >= 1000,
              x3 >= 1000,
              x2 <= 10000,
              x3 <= 10000,
              x4 >= 10,
              x5 >= 10,
              x6 >= 10,
              x7 >= 10,
              x8 >= 10,
              x4 <= 1000,
              x5 <= 1000,
              x6 <= 1000,
              x7 <= 1000,
              x8 <= 1000])

m.localsolve(algorithm="Xu")

