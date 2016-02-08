from .nomials import Variable
from .nomials import ArrayVariable
from .model import Model

Var = Variable
Vec = ArrayVariable
Arr = ArrayVariable
Model = Model


def GP(*args, **kwargs):
    print("'GP' has been replaced by 'Model', and will be removed in the next"
          " point release. Please update your code!")
    return Model(*args, **kwargs)


def SP(*args, **kwargs):
    print("'SP' has been replaced by 'Model', and will be removed in the next"
          " point release. Please update your code!")
    return Model(*args, **kwargs)
