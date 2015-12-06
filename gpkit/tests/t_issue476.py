"""Test issue476 rotor example"""
import unittest
import numpy as np
from gpkit import Variable, VectorVariable, units, Model


def rotor_test():
    N = 20
    Weight = 10660*9.8
    xi = np.ones(N)*Weight/float(N)

    rho = Variable("rho", 1.23, "kg/m^3", "Density of Air")
    W = Variable("W", Weight, "N", "Weight of Vehicle")
    xi  = VectorVariable(N, "xi", xi, "N", "Constant Thrust per Bin")

    A_ideal     = Variable("A_ideal", "m^2", "Disk Area")
    Omega_ideal = Variable("Omega_ideal", "rpm", "Rotor RPM")
    Omega_max = Variable("Omega_max", 280, "rpm", "Max Rotor RPM")
    CP_ideal    = Variable("CP_ideal", "-", "Coefficient of Profile Power")
    P_ideal     = Variable("P_ideal", "W", "Total Power")
    R_ideal     = Variable("R_ideal", 8, "m", "Rotor Radius")
    
    r_ideal    = VectorVariable(N, "r_ideal", "-", "Non-dimensional Radius")
    dr_ideal   = VectorVariable(N, "dr_ideal", "-", "Non-dimensional Radius Step")
    Vi_ideal   = VectorVariable(N, "Vi_ideal", "m/s", "Induced Velocities")
    dCT_ideal  = VectorVariable(N, "dC_T_ideal", "-", "Incremental Thrust Coefficient of Each Bin")
    dCP_ideal  = VectorVariable(N, "dC_P_ideal", "-", "Coefficient of Profile Power")
    dP_ideal   = VectorVariable(N, "dP_ideal", "W", "Incremental Profile Power of Each Bin")
    
    phys_constraints_ideal = [A_ideal == np.pi*R_ideal**2,
                              Omega_ideal <= Omega_max,
                              ]
    
    r_constraints_ideal = [dr_ideal[0]/2 == r_ideal[0],
                           [r_ideal[j] >= r_ideal[j-1] + .5*dr_ideal[j-1] + .5*dr_ideal[j] for j in range(1, N)],
                           r_ideal[-1] + dr_ideal[-1]/2 <= 1
                           ]
    
    Figure_of_Merit = [xi == rho*A_ideal*(Omega_ideal*R_ideal*r_ideal[-1])**2*dCT_ideal,
                       0.25 == Vi_ideal**2*r_ideal*dr_ideal/(dCT_ideal*(Omega_ideal*R_ideal*r_ideal[-1])**2),
                       0.25 == Vi_ideal**3*r_ideal*dr_ideal/(dCP_ideal*(Omega_ideal*R_ideal*r_ideal[-1])**3),
                       dP_ideal == rho*A_ideal*(Omega_ideal*R_ideal*r_ideal[-1])**3*dCP_ideal,
                       P_ideal >= dP_ideal.sum()
                      ]

    objective = P_ideal
    eqns = phys_constraints_ideal + r_constraints_ideal + Figure_of_Merit
    gp = Model(objective, eqns)
    Ideal_Rotor = gp.solve()


class TestIssue476(unittest.TestCase):
    """TestCase for Issue476"""

    def test_issue476(self):
        """Test issue476"""
        rotor_test()


TESTS = [TestIssue476]

if __name__ == '__main__':
    from gpkit.tests.helpers import run_tests
    run_tests(TESTS)
