from .variables import Variable
from .variables import VectorVariable
from .variables import ArrayVariable
from .model import Model

Var = Variable
Vec = VectorVariable
Arr = ArrayVariable
Model = Model

def GP(*args, **kwargs):
    print("'Model' has replaced 'GP'. Please update your code!")
    return Model(*args, **kwargs)

def SP(*args, **kwargs):
    print("'Model' has replaced 'GP'. Please update your code!")
    return Model(*args, **kwargs)
