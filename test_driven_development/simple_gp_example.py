"""
SIMPLE GP FOR AIRCRAFT

This file is an example to test out the gpkit interface.

The file is heavily commented, not just because it's an example but
  also to show the potential for clear and self-explanatory modeling.

The file 'Simple GP for Aircraft.ipynb' in this folder has the same
  example but as an iPython Notebook (and with graphs).
"""

from numpy import linspace, zeros, pi
import gpkit

from line_profiler import LineProfiler

try:
    from line_profiler import LineProfiler

    def do_profile(follow=[]):
        def inner(func):
            def profiled_func(*args, **kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)
                    for f in follow:
                        profiler.add_function(f)
                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats()
            return profiled_func
        return inner

except ImportError:
    def do_profile(follow=[]):
        "Helpful if you accidentally leave in production!"
        def inner(func):
            def nothing(*args, **kwargs):
                return func(*args, **kwargs)
            return nothing
        return inner


### INITALIZE ###
constants = {
  'CDA0': 0.03062702,  # [m^2] fuselage drag area
  'rho': 1.23,         # [kg/m^3] density of air
  'mu': 1.78e-5,       # [kg/(m*s)] viscosity of air
  'S_wet_ratio': 2.05, # wetted area ratio
  'k': 1.2,            # [-] form factor
  'e': 0.96,           # [-] Oswald efficiency factor
  'W_0': 4940,         # [N] aircraft weight excluding wing
  'N_ult': 2.5,        # [-] ultimate load factor
  'tau': 0.12,         # [-] airfoil thickness to chord ratio
  'C_Lmax': 2.0,       # [-] max CL, flaps down
  'V': None,           # [m/s] cruising speed, to be iterated over
  'V_min': None,       # [m/s] takeoff speed, to be iterated over 
}
gpkit.monify_up(globals(), constants)

## free variables ##
freevars = [
  'A', # [-] aspect ratio
  'S', # [m^2] total wing area
  'C_D', # [-] Drag coefficient of wing
  'C_L', # [-] Lift coefficent of wing
  'C_f', # [-] skin friction coefficient
  'Re', # [-] Reynold's number
  'W', # [N] total aircraft weight
  'W_w' # [N] wing weight
]
gpkit.monify_up(globals(), freevars)

# drag modeling
C_D_fuse = CDA0/S # fuselage viscous drag
C_D_wpar = k*C_f*S_wet_ratio # wing parasitic drag
C_D_ind = C_L**2/(pi*A*e) # induced drag

# wing-weight modeling
W_w_surf = 45.24*S # surface weight
W_w_strc = 8.71e-5*(N_ult*A**1.5*(W_0*W*S)**0.5)/tau # structural weight


from gpkit._mosek.expopt import imize as optimizer
gp = gpkit.GP( # minimize
               0.5*rho*S*C_D*V**2, 
             [ # subject to
              Re <= (rho/mu)*V*(S/A)**0.5,
              C_f >= 0.074/Re**0.2,
              W <= 0.5*rho*S*C_L*V**2,
              W <= 0.5*rho*S*C_Lmax*V_min**2,
              W >= W_0 + W_w,
              W_w >= W_w_surf + W_w_strc,
              C_D >= C_D_fuse + C_D_wpar + C_D_ind
             ], solver='attached',
                options={'solver': optimizer})


@do_profile(follow=[gp.solve, optimizer])
def solve_all(gp, shape, Vs, V_mins):
  data = {var: zeros(shape) for var in freevars}
  for i, V in enumerate(Vs):
    constants.update({'V': V})
    for j, V_min in enumerate(V_mins):
        constants.update({'V_min': V_min})
        gp.replace_constants(constants)
        sol = gp.solve()
        for var in sol:  data[var][i,j] = sol[var]
  return data

from numpy import linspace, zeros
shape = (30,30)
Vs = linspace(45, 55, shape[0])
V_mins = linspace(20, 25, shape[1])

data = solve_all(gp, shape, Vs, V_mins)

print "           | Averages"
for key, table in data.iteritems():
  val = table.mean().mean()
  if val < 100 and val > 0.1:
    valstr = ("%4.3f" % val)[:4]
  else:
    valstr = "%2.2e" % val
  print "%10s" % key, ":", valstr
print "           |"


