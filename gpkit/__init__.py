# -*- coding: utf-8 -*-
"""Lightweight GP Modeling Package

    For examples please see ``../examples``

    Requirements
    ------------
    numpy
    MOSEK or CVXOPT
    sympy(optional): for latex printing in iPython Notebook

    Attributes
    ----------
    settings : dict
        Contains settings loaded from ``./env/settings``
"""

from .posyarray import PosyArray
from .nomials import Monomial
from .nomials import Posynomial
from .geometric_program import GP
from .variables import Variable
from .variables import VectorVariable

try:
    # requires sympy
    from .printing import init_printing as init_ipynb_printing
except ImportError:
    pass

# Load settings
from os import sep as os_sep
from os.path import dirname as os_path_dirname
settings_path = os_sep.join([os_path_dirname(__file__),
                             "env", "settings"])
with open(settings_path) as settingsfile:
    lines = [line[:-1].split(" : ") for line in settingsfile
             if len(line.split(" : ")) == 2]
    settings = {name: value.split(", ") for (name, value) in lines}
