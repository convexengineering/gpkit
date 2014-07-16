"""
SIMPLE GP FOR AIRCRAFT DESIGN

The 'ipynb' folder has the same example as an iPython Notebook.
"""

import cProfile
import pstats

# Profilin'
profile = cProfile.Profile()
profile.enable()

from numpy import linspace, pi

import gpkit

constants = {
    'CDA0': (0.03062702, "[m^2] fuselage drag area"),
    'rho': (1.23, "[kg/m^3] density of air"),
    'mu': (1.78e-5, "[kg/(m*s)] viscosity of air"),
    'S_wet_ratio': (2.05, "[-] wetted area ratio"),
    'k': (1.2, "[-] form factor"),
    'e': (0.96, "[-] Oswald efficiency factor"),
    'W_0': (4940, "[N] aircraft weight excluding wing"),
    'N_ult': (2.5, "[-] ultimate load factor"),
    'tau': (0.12, "[-] airfoil thickness to chord ratio"),
    'C_Lmax': (2.0, "[-] max CL, flaps down"),
    'V': ('sweep', linspace(45, 55, 10), "[m/s] cruising speed"),
    'V_min': ('sweep', linspace(20, 25, 10), "[m/s] takeoff speed"),
}
gpkit.monify_up(globals(), constants)

free_variables = {
    'A': "[-] aspect ratio",
    'S': "[m^2] total wing area",
    'C_D': "[-] Drag coefficient of wing",
    'C_L': "[-] Lift coefficent of wing",
    'C_f': "[-] skin friction coefficient",
    'Re': "[-] Reynold's number",
    'W': "[N] total aircraft weight",
    'W_w': "[N] wing weight",
}
gpkit.monify_up(globals(), free_variables)

# drag modeling #
C_D_fuse = CDA0/S             # fuselage viscous drag
C_D_wpar = k*C_f*S_wet_ratio  # wing parasitic drag
C_D_ind = C_L**2/(pi*A*e)     # induced drag

# wing-weight modeling #
W_w_surf = 45.24*S                                    # surface weight
W_w_strc = 8.71e-5*(N_ult*A**1.5*(W_0*W*S)**0.5)/tau  # structural weight

gp = gpkit.GP(  # minimize
                0.5*rho*S*C_D*V**2,
                [   # subject to
                    Re <= (rho/mu)*V*(S/A)**0.5,
                    C_f >= 0.074/Re**0.2,
                    W <= 0.5*rho*S*C_L*V**2,
                    W <= 0.5*rho*S*C_Lmax*V_min**2,
                    W >= W_0 + W_w,
                    W_w >= W_w_surf + W_w_strc,
                    C_D >= C_D_fuse + C_D_wpar + C_D_ind
                ], constants=constants, solver='mosek_cli')

data = gp.solve()

# Results
profile.disable()
ps = pstats.Stats(profile)
ps.strip_dirs()
ps.sort_stats('time')
ps.print_stats(10)

print "                 | Averages"
for key, table in data.iteritems():
    try:
        val = table.mean()
    except AttributeError:
        val = table
    if abs(val) > 1e-9:
        print "%16s" % key, ": %-8.3g" % val, gp.var_descrs[key]
print "                 |"
