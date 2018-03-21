from gpkit import Model, Variable, SignomialsEnabled

class PowerBalance(Model):
    "simple non-dimensional aircraft power balance model with BLI"
    def setup(self):
        # Variables
        ujet = Variable("V_{\\rm jet}/V_\\infty - 1", "-",
                        "normalized jet velocity perturbation")
        PK = Variable("P_K/(V_\\infty D')", "-", "nomalized flow power")

        # Constants
        Dp = Variable("D_p'/D'", 0.662, "-", "profile drag fraction")
        fBLI = Variable("f_{\\rm BLI}", 0.4, "-",
                        "ingested boundary layer fraction")
        fsurf = Variable("f_{\\rm surf}", 0.836, "-",
                         "profile surface dissipation fraction")
        mdot = Variable("\\dot{m}V_\\infty/D'", 1/0.7376, "-",
                        "normalized propulsor mass flow")

        # Power balance equation (generally a signomial)
        with SignomialsEnabled():
            s1 = Variable("\\sigma_1", "-", "signomial term", sp_init=1)
            s2 = Variable("\\sigma_2", "-", "signomial term", sp_init=0)
            constraint_PB = [s1 == mdot*ujet,
                             s2 == fBLI*Dp,
                             # s1 + s2 >= 1,
                             mdot*ujet + fBLI*Dp >= 1,
                             ]

        # Power consumption
        self.cost = PK
        constraint_PK = [PK >= 0.5*mdot*ujet*(2 + ujet) + fBLI*fsurf*Dp]

        return [constraint_PB, constraint_PK,
                fBLI <= 1, fsurf <= 1, Dp <= 1]


print PowerBalance().solve(verbosity=0).table(tables=["sensitivities"])
