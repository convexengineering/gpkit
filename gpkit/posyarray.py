import numpy as np


class PosyArray(np.ndarray):
    "Numpy array subclass with elementwise inequalities and substitutions"
    def __ne__(self, m):
        return not isinstance(other, self.__class__) and self._eq(self, other).all()

    def _latex(self, unused=None):
        return "["+", ".join(el._latex() for el in self)+"]"

    # constraint generators
    _eq = np.vectorize(lambda a, b: a == b)
    def __eq__(self, x):
        if self.shape:
            return [e for e in self._eq(self, x)]
        else:
            self = self.flatten()
            return self._eq(self, x)
    _leq = np.vectorize(lambda a, b: a <= b)
    def __le__(self, x): return [e for e in self._leq(self, x)]
    _geq = np.vectorize(lambda a, b: a >= b)
    def __ge__(self, x): return [e for e in self._geq(self, x)]

    def outer(self, x): return PosyArray(np.outer(self, x))

    def __new__(cls, input_array, info=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'info', None)

    def sub(self, subs, val=None):
        if self.shape:
            return PosyArray([p.sub(subs, val) for p in self])
        else:
            # 0D array
            self = self.flatten()[0]
            return PosyArray(self.sub(subs, val))

    def __nonzero__(self):
        return self.all().__nonzero__()