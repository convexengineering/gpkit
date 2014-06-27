import numpy as np

class matrix(np.matrix):
    # change printing
    __repr__ = lambda self: str(self)
    __eq__ = lambda self, m: (isinstance(m, self.__class__)
                              and str(self) == str(m))
    __ne__ = lambda self, m: not self == m

    # element-wise operators
    pow = lambda self, x: np.power(self, x)
    div = lambda self, x: np.divide(self, x)
    mul = lambda self, x: np.multiply(self, x)

    # constraint generators
    _leq = np.vectorize(lambda a, b: a <= b)
    __lt__ = lambda self, x: self <= x
    __le__ = lambda self, x: [e if len(e)>1
                                else e[0]
                              for e in self._leq(self, x).tolist()]