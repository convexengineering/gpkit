import numpy as np

import gpkit
from gpkit import Variable


class simpleflight_generator(object):

    def __init__(self, disableUnits=False):
        if disableUnits:
            gpkit.disable_units()
            if gpkit.units:
                raise RuntimeWarning

        pi = Variable("\\pi", np.pi, "-", "half of the circle constant")
        rho = Variable("\\rho", 1.23, "kg/m^3", "density of air")
        mu = Variable("\\mu", 1.78e-5, "kg/m/s", "viscosity of air")
        S_wetratio = Variable("(\\frac{S}{S_{wet}})", 2.05, "-",
                              "wetted area ratio")
        k = Variable("k", 1.2, "-", "form factor")
        e = Variable("e", 0.95, "-", "Oswald efficiency factor")
        N_ult = Variable("N_{ult}", 3.8, "-", "ultimate load factor")
        tau = Variable("\\tau", 0.12, "-", "airfoil thickness to chord ratio")
        C_Lmax = Variable("C_{L,max}", 1.5, "-", "max CL with flaps down")
        V_min = Variable("V_{min}", 22, "m/s", "takeoff speed")

        if gpkit.units:
            CDA0 = Variable("(CDA0)", 310.0, "cm^2", "fuselage drag area")
            W_0 = Variable("W_0", 4.94, "kN", "aircraft weight excluding wing")
        else:
            CDA0 = Variable("(CDA0)", 0.031, "m^2", "fuselage drag area")
            W_0 = Variable("W_0", 4940.0, "N",
                           "aircraft weight excluding wing")

        D = Variable("D", "N", "total drag force")
        A = Variable("A", "-", "aspect ratio")
        S = Variable("S", "m^2", "total wing area")
        C_D = Variable("C_D", "-", "Drag coefficient of wing")
        C_L = Variable("C_L", "-", "Lift coefficent of wing")
        C_f = Variable("C_f", "-", "skin friction coefficient")
        Re = Variable("Re", "-", "Reynold's number")
        W = Variable("W", "N", "total aircraft weight")
        W_w = Variable("W_w", "N", "wing weight")
        V = Variable("V", "m/s", "cruising speed")

        equations = []

        C_D_fuse = CDA0/S
        C_D_wpar = k*C_f*S_wetratio
        C_D_ind = C_L**2/(pi*A*e)
        equations += [C_D >= C_D_fuse + C_D_wpar + C_D_ind]

        W_w_strc = 8.71e-5*(N_ult*A**1.5*(W_0*W*S)**0.5)/tau / gpkit.units.m
        W_w_surf = (45.24*gpkit.units.Pa) * S
        equations += [W_w >= W_w_surf + W_w_strc]

        equations += [D >= 0.5*rho*S*C_D*V**2,
                      Re <= (rho/mu)*V*(S/A)**0.5,
                      C_f >= 0.074/Re**0.2,
                      W <= 0.5*rho*S*C_L*V**2,
                      W <= 0.5*rho*S*C_Lmax*V_min**2,
                      W >= W_0 + W_w]

        self.pi, self.rho, self.mu, self.S_wetratio, self.k, self.e, self.N_ult, self.tau, self.C_Lmax, self.V_min, self.CDA0, self.W_0, self.CDA0, self.W_0, self.D, self.A, self.S, self.C_D, self.C_L, self.C_f, self.Re, self.W, self.W_w, self.V, self.C_D_fuse, self.C_D_wpar, self.C_D_ind, self.W_w_strc, self.W_w_surf, self.W_w_strc, self.W_w_surf, self.equations = pi, rho, mu, S_wetratio, k, e, N_ult, tau, C_Lmax, V_min, CDA0, W_0, CDA0, W_0, D, A, S, C_D, C_L, C_f, Re, W, W_w, V, C_D_fuse, C_D_wpar, C_D_ind, W_w_strc, W_w_surf, W_w_strc, W_w_surf, equations

    def gp(self):
        return gpkit.GP(self.D, self.equations)

    def sweep(self, n):
        substitutions = {self.V_min: ("sweep", np.linspace(20, 25, n)),
                         self.V: ("sweep", np.linspace(45, 55, n)), }
        return gpkit.GP(self.D, self.equations, substitutions)


if __name__ == "__main__":
    import cProfile
    import pstats

    # Profilin'
    profile = cProfile.Profile()
    profile.enable()

    sf = simpleflight_generator()
    sol = sf.sweep(3).solve()

    # Results
    profile.disable()
    ps = pstats.Stats(profile)
    ps.strip_dirs()
    ps.sort_stats("time")
    ps.print_stats(10)

    print(sol.table())
