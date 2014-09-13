from collections import defaultdict
from collections import namedtuple
from collections import Iterable
from internal_utils import *

from gpkit.nomials import Posynomial

try:
    import numpy as np
except ImportError:
    print "Could not import numpy: will not be able to sweep variables"

try:
    from scipy.sparse import coo_matrix
except ImportError:
    print "Could not import scipy: will not be able to use splines"

CootMatrix = namedtuple('CootMatrix', ['row', 'col', 'data'])
PosyTuple = namedtuple('PosyTuple', ['exps', 'cs', 'var_locs', 'var_descrs',
                                     'substitutions'])


class CootMatrix(CootMatrix):
    "A very simple sparse matrix representation."
    shape = (None, None)

    def append(self, i, j, x):
        assert (i >= 0 and j >= 0), "Only positive indices allowed"
        self.row.append(i)
        self.col.append(j)
        self.data.append(x)

    def update_shape(self):
        self.shape = (max(self.row)+1, max(self.col)+1)

    def tocoo(self):
        "Converts to a Scipy sparse coo_matrix"
        return coo_matrix((self.data, (self.row, self.col)))

    def todense(self): return self.tocoo().todense()
    def tocsr(self): return self.tocoo().tocsr()
    def tocsc(self): return self.tocoo().tocsc()
    def todok(self): return self.tocoo().todok()
    def todia(self): return self.tocoo().todia()


class Model(object):

    def __repr__(self):
        return "\n".join(["gpkit.Model with",
                          "  Equations"] +
                         ["     %s <= 1" % p._string()
                          for p in self.posynomials]
                         )

    def add_eqns(self):
        self.posynomials = posynomials

        self.sweep = {}
        self._gen_unsubbed_vars()

        if constants:
            self.sub(constants, tobase='initialsub')

    def print_boundwarnings(self):
        for var, bound in self.missingbounds.iteritems():
            print "%s (%s) has no %s bound" % (
                  var, self.var_descrs[var], bound)

    def add_constraints(self, constraints):
        if isinstance(constraints, Posynomial):
            constraints = [constraints]
        self.constraints += tuple(constraints)
        self._gen_unsubbed_vars()

    def rm_constraints(self, constraints):
        if isinstance(constraints, Posynomial):
            constraints = [constraints]
        for p in constraints:
            self.constraints.remove(p)
        self._gen_unsubbed_vars()

    def _gen_unsubbed_vars(self):
        posynomials = self.posynomials

        var_descrs = defaultdict(str)
        for p in posynomials:
            var_descrs.update(p.var_descrs)

        exps = reduce(lambda x,y: x+y, map(lambda x: x.exps, posynomials))
        cs = reduce(lambda x,y: x+y, map(lambda x: x.cs, posynomials))
        var_locs = locate_vars(exps)

        self.unsubbed = PosyTuple(exps, cs, var_locs, var_descrs, {})
        self.load(self.unsubbed)

        # k [j]: number of monomials (columns of F) present in each constraint
        self.k = [len(p.cs) for p in posynomials]

        # p_idxs [i]: posynomial index of each monomial
        p_idx = 0
        self.p_idxs = []
        for p_len in self.k:
            self.p_idxs += [p_idx]*p_len
            p_idx += 1

    def sub(self, substitutions, val=None, frombase='last', tobase='subbed'):
        # look for sweep variables
        sweepdescrs = {}
        if isinstance(substitutions, dict):
            subs = dict(substitutions)
            for var, sub in substitutions.iteritems():
                try:
                    if sub[0] == 'sweep':
                        del subs[var]
                        if isinstance(sub[1], Iterable):
                            self.sweep.update({var: sub[1]})
                            if isinstance(sub[-1], str):
                                if isinstance(sub[-2], str):
                                    sweepdescrs.update({var: sub[-2:]})
                                else:
                                    sweepdescrs.update({var: [None, sub[-1]]})
                    else:
                        raise ValueError("sweep variables must be iterable.")
                except: pass
        else:
            subs = substitutions

        base = getattr(self, frombase)

        # perform substitution
        var_locs, exps, cs, newdescrs, subs = substitution(base.var_locs,
                                                           base.exps,
                                                           base.cs,
                                                           subs, val)
        if not subs:
            raise ValueError("could not find anything to substitute")

        var_descrs = base.var_descrs
        var_descrs.update(newdescrs)
        var_descrs.update(sweepdescrs)
        substitutions = base.substitutions
        substitutions.update(subs)

        newbase = PosyTuple(exps, cs, var_locs, var_descrs, substitutions)
        setattr(self, tobase, self.last)
        self.load(newbase)
        self.print_boundwarnings()

    def load(self, posytuple):
        self.last = posytuple
        for attr in ['exps', 'cs', 'var_locs', 'var_descrs', 'substitutions']:
            new = getattr(posytuple, attr)
            setattr(self, attr, new)

        # A: exponents of the various free variables for each monomial
        #    rows of A are variables, columns are monomials
        self.missingbounds = {}
        self.A = CootMatrix([], [], [])
        for j, var in enumerate(self.var_locs):
            varsign = None
            for i in self.var_locs[var]:
                exp = self.exps[i][var]
                self.A.append(j, i, exp)
                if varsign is None: varsign = np.sign(exp)
                elif varsign is "both": pass
                elif np.sign(exp) != varsign: varsign = "both"
            if varsign != "both" and var not in self.sweep:
                if varsign == 1: bound = "lower"
                elif varsign == -1: bound = "upper"
                self.missingbounds[var] = bound
        self.A.update_shape()