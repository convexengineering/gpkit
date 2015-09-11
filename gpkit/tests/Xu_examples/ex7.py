'''
Example 7, Xu 2013
Status: t no upper bound, exp overflow, UNKNOWN status
'''

from gpkit.shortcuts import *
from gpkit import SignomialsEnabled

Hc = Var('Hc')
Ht = Var('Ht')
H = Var('H')
R = Var('R')
Rm = Var('Rm')
phi = Var('phi')
t = Var('t')

with SignomialsEnabled():
    m = Model(t,
             [t >= 720*Hc + 43200*phi + 14400*phi**3 + 5760*phi**5 + R**2*phi**3 + 0.4*R**2*phi**5 - 7198.2,
              252.154*H**-2 + 4500*R**-2 <= 1,
              0.0125*H + 0.00833*R*phi + 0.0000694*R*phi**5 - 0.001389*R*phi**3 <= 1,
              2238.432*Hc**-3 + 53720.208*Hc**-4*phi + 17906.736*Hc**-4*phi**3 + 7162.694*Hc**-4*phi**5 + 19.995*Hc**-1 - 8951.297*Hc**-4 - 120*Hc**-1*phi -40*Hc**-1*phi**3 - 16*Hc**-1*phi**5 <= 1,
              30.52132*Hc**-1 - 120*Hc**-1*phi - 40*Hc**-1*phi**3 - 16*Hc**-1*phi**5 <= 1,
              252.1543*Ht**-2 + 0.005837*Ht**-2*R**2*phi**4 + 4500*R**-2 - 0.0175*Ht**-2*R**2*phi**2 - 0.000778*Ht**-2*R**2*phi**6 <= 1,
              67.73085*H**-1.8*Rm**0.2*phi**0.2 + 146.53487*H**-0.8*Rm**-0.8*phi**0.2 + 393.09732*H**0.2*Rm**-1.8*phi**0.2 <= 1,
              H*Ht**-1 + 0.5*Ht**-1*R*phi**2 + 0.02777*Ht**-1*R*phi**3 - 0.0416667*Ht**-1*R*phi**4 - 0.16663*Ht**-1*R*phi - 0.001389*Ht**-1*R*phi**5 <= 1,
              H*Ht**-1 + 0.5*Ht**-1*R*phi**2 + 0.02777*Ht**-1*R*phi**3 - 0.0416667*Ht**-1*R*phi**4 - 0.16663*Ht**-1*R*phi - 0.001389*Ht**-1*R*phi**5 >= 1,
              2*H*R**-1*phi**-2 -2*Hc*R**-1*phi**-2 - 0.41667*phi**2 - 0.16944*phi**4 <= 1,
              2*H*R**-1*phi**-2 -2*Hc*R**-1*phi**-2 - 0.41667*phi**2 - 0.16944*phi**4 >= 1,
              R**-1*Rm - 0.5*H*R**-1 <= 1,
              R**-1*Rm - 0.5*H*R**-1 >= 1,
              ])

m.localsolve()

