"""Defines form_program method"""
import numpy as np

from .nomials import Signomial
from .geometric_program import GeometricProgram
from .signomial_program import SignomialProgram
from .nomial_data import sort_and_simplify

from .substitution import substitution


def form_program(program_type, signomials, subs, verbosity=2):
    """Generates a program, applying substitutions

    Arguments
    ---------
    program_type : string
        "gp" or "sp"

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
    ValueError if program_type and model constraints don't match.
    """
    signomials_, mmaps = [], []
    for s in signomials:
        _, exps, cs, _ = substitution(s, subs)
        # remove any cs that are just nans and/or 0s
        notnan = ~np.isnan(cs)
        if np.any(notnan) and np.any(cs[notnan] != 0):
            exps, cs, mmap = sort_and_simplify(exps, cs, return_map=True)
            signomials_.append(Signomial(exps, cs, units=s.units))
            mmaps.append(mmap)
        else:
            mmaps.append([None]*len(cs))

    cost = signomials_[0]
    constraints = signomials_[1:]

    if program_type in ["gp", "GP"]:
        return GeometricProgram(cost, constraints, verbosity-1), mmaps
    elif program_type in ["sp", "SP"]:
        return SignomialProgram(cost, constraints), mmaps
    else:
        raise ValueError("unknown program type %s." % program_type)
