from gpkit import Model, Variable, SignomialsEnabled

class PowerBalance(Model):
    def setup(self):
        ujet = Variable("ujet")
        PK = Variable("PK")

        # Constants
        Dp = Variable("Dp", 0.662)
        fBLI = Variable("fBLI", 0.4)
        fsurf = Variable("fsurf", 0.836)
        mdot = Variable("mdot", 1/0.7376)

        self.cost = PK
        with SignomialsEnabled():
            return [
                    # mdot*ujet >= 1 - 0.2648,
                    # mdot*ujet +  0.2648 >= 1,
                    mdot*ujet + fBLI*Dp >= 1,
                    PK >= 0.5*mdot*ujet*(2 + ujet) + fBLI*fsurf*Dp
                    ]


print PowerBalance().solve(verbosity=0).table()
