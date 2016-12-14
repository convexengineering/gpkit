"Implements SPData class"
import numpy as np
from ..nomials import NomialData, SignomialInequality, PosynomialInequality
from ..constraints.geometric_program import genA
from ..small_scripts import mag


class SPData(NomialData):
    """Generates matrices describing an SP.

    Usage
    -----
    >>> spdata = SPData(m)
    >>> spdata.save('example_sp.h5')
    """
    def __init__(self, model):
        # pylint:disable=super-init-not-called
        if not hasattr(model, "solution"):
            raise ValueError("You need to solve the model first.")

        self.signomials = [model.cost]
        for constraint in model.flat(constraintsets=False):
            if isinstance(constraint, (SignomialInequality,
                                       PosynomialInequality)):
                self.signomials.extend(constraint.unsubbed)
            else:
                raise ValueError("unknown constraint %s of type %s"
                                 % (constraint, type(constraint)))
        NomialData.init_from_nomials(self, self.signomials)

        # k [j]: number of monomials (columns of F) present in each constraint
        k = [len(p.cs) for p in self.signomials]
        # p_idxs [i]: posynomial index of each monomial
        p_idxs = []
        for i, p_len in enumerate(k):
            p_idxs += [i]*p_len
        self.p_idxs = np.array(p_idxs)
        # A [i, v]: sparse matrix of variable's powers in each monomial
        self.A, _ = genA(self.exps, self.varlocs)
        # NOTE: NomialData might be refactored to include the above

        self.varsols = np.array([mag(model.solution(var))
                                 for var in self.varlocs])
        self.varnames = np.array([str(var) for var in self.varlocs])

    def __repr__(self):
        return str(self.signomials)

    def save(self, filename):
        "Save spdata to an h5 file."
        try:
            import h5py  # pylint:disable=import-error
        except ImportError as ie:
            print "SPData.save requires the h5py library."
            raise ie
        h5f = h5py.File(filename, 'w')
        try:
            h5f.create_dataset('cs', data=self.cs)
            h5f.create_dataset('A', data=self.A.todense())
            h5f.create_dataset('p_idxs', data=self.p_idxs)
            h5f.create_dataset('varsols', data=self.varsols)
            h5f.create_dataset('varnames', data=self.varnames)
        finally:
            h5f.close()
