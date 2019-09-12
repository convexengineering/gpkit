"""Implement the SignomialProgram class"""
from __future__ import unicode_literals, print_function
from time import time
from collections import OrderedDict
import numpy as np
from gpkit.nomials import Variable, VectorVariable
from gpkit.nomials import SignomialInequality, PosynomialInequality
from gpkit.nomials import SingleSignomialEquality, MonomialEquality
from gpkit import SignomialsEnabled, NamedVariables
from gpkit.constraints.costed import CostedConstraintSet
from gpkit.small_scripts import mag

from robust.testing.models import simple_ac

"""
Deciphering this code...
Each maps consists of two layers of lists. 
First layer describes the number of constraints
Second layer contains the hmaps of all of the variables
that are involved in the constraints. 

"""

m = simple_ac()
constraints = [i for i in m.flat(constraintsets=False)]
n_constr = len(constraints)
varkeys = sorted(m.varkeys)
n_vks = len(varkeys)
maps = {}
signomial_indices = []
for i in range(n_constr):
    constraint = constraints[i]
    if isinstance(constraint, MonomialEquality):
        maps[i] = [[monomial.hmap] for monomial in constraint.as_posyslt1()]
    elif isinstance(constraint, PosynomialInequality):
        maps[i] = [[monomial.hmap for monomial in constraint.as_posyslt1()]]
    elif isinstance(constraint, SignomialInequality):
        signomial_indices.append(i)
        with SignomialsEnabled():
            if isinstance(constraint, SingleSignomialEquality):
                # Putting constraints in less-than-zero representation
                ltzero_rep = [constraint.right-constraint.left, constraint.left-constraint.right]
                maps[i] = [[monomial.hmap for monomial in ltzero_rep[0].chop()],
                           [monomial.hmap for monomial in ltzero_rep[1].chop()]]
            else:
                ltzero_rep = (constraint.right-constraint.left)*(-1.+2*(constraint.oper=='>='))
                maps[i] = [[monomial.hmap for monomial in ltzero_rep.chop()]]
