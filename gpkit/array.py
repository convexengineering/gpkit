import numpy as np


class array(np.ndarray):
    _eq = np.vectorize(lambda a, b: a == b)
    def __eq__(self, other):
        return isinstance(other, self.__class__) and self._eq(self, other).all()
    def __ne__(self, m): return not self == m

    # constraint generators
    _leq = np.vectorize(lambda a, b: a <= b)
    def __lt__(self, x): return self <= x
    def __le__(self, x): return [e for e in self._leq(self, x)]
    _geq = np.vectorize(lambda a, b: a >= b)
    def __gt__(self, x): return self >= x
    def __ge__(self, x): return [e for e in self._geq(self, x)]

    def outer(self, x): return array(np.outer(self, x))

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
            return array([p.sub(subs, val) for p in self])
        else:
            # 0D array
            self = self.flatten()[0]
            return array(self.sub(subs, val))
