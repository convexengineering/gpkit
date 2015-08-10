from .variables import Variable
from .variables import VectorVariable
from .variables import ArrayVariable

Var = Variable
Vec = VectorVariable
Arr = ArrayVariable


def GP(*args, **kwargs):
    raise Exception("'Model' has replaced 'GP'. Please update your code!")

def SP(*args, **kwargs):
    raise Exception("'Model' has replaced 'GP'. Please update your code!")
