import numpy as np

class matrix(np.matrix):
    # change printing
    def __repr__(self): return str(self)
    def __eq__(self, m): (isinstance(m, self.__class__)
                                and str(self) == str(m))
    def __ne__(self, m): not self == m

    # element-wise operators
    def pow(self, x): return np.power(self, x)
    def div(self, x): return np.divide(self, x)
    def mul(self, x): return np.multiply(self, x)

    # constraint generators
    _leq = np.vectorize(lambda a, b: a <= b)
    def __lt__(self, x): return self <= x
    def __le__(self, x): return [e if len(e) > 1 else e[0]
                                 for e in self._leq(self, x).tolist()]