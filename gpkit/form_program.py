from .nomials import Signomial
from .geometric_program import GeometricProgram
from .signomial_program import SignomialProgram
from .nomial_data import sort_and_simplify

from .substitution import substitution
from .small_scripts import mag

def form_program(programType, signomials, subs, verbosity=2):
    """Generates a program and solves it, sweeping as appropriate.

    Arguments
    ---------
    programType : "gp" or "sp"

    signomials : list of Signomials
        The first Signomial is the cost function.

    subs : dict
        Substitutions to do before solving.

    verbosity : int (optional)
        If greater than 0 prints runtime messages.
        Is decremented by one and then passed to program inits.

    Returns
    -------
    program : GP or SP
    mmaps : Map from initial monomials to substitued and simplified one.
            See small_scripts.sort_and_simplify for more details.

    Raises
    ------
    ValueError if programType and model constraints don't match.
    """
    signomials_, mmaps = [], []
    for s in signomials:
        _, exps, cs, _ = substitution(s, subs)
        if any((mag(c) != 0 for c in cs)):
            exps, cs, mmap = sort_and_simplify(exps, cs, return_map=True)
            signomials_.append(Signomial(exps, cs, units=s.units))
            mmaps.append(mmap)
        else:
            mmaps.append([None]*len(cs))

    cost = signomials_[0]
    constraints = signomials_[1:]

    if programType in ["gp", "GP"]:
        return GeometricProgram(cost, constraints, verbosity-1), mmaps
    elif programType in ["sp", "SP"]:
        return SignomialProgram(cost, constraints), mmaps
    else:
        raise ValueError("unknown program type %s." % programType)
