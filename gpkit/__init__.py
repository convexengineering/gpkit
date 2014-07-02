"""
Lightweight GP Modeling Package

Needs mosek or cvxopt to be able to solve GPs

Uses numpy for ndarrays
     scipy for splines, sparse matrix operations
"""

from array import array
from nomials import Monomial
from nomials import Posynomial
from geometric_program import GP

from utils import *

from os import sep as os_sep
from os.path import dirname as os_path_dirname
settings_path = os_sep.join([os_path_dirname(__file__),
                             "env", "settings"])
with open(settings_path) as settingsfile:
    lines = [line.split() for line in settingsfile
             if len(line.split()) == 2]
    settings = {name: value for (name, value) in lines}
