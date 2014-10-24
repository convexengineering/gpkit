# -*- coding: utf-8 -*-
"""Lightweight GP Modeling Package

    For examples please see the examples folder.

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

from .nomials import Monomial
from .nomials import Posynomial
from .nomials import monovector
from .posyarray import PosyArray
from .geometric_program import GP

# Load settings
from os import sep as os_sep
from os.path import dirname as os_path_dirname
settings_path = os_sep.join([os_path_dirname(__file__), "env", "settings"])
with open(settings_path) as settingsfile:
    lines = [line[:-1].split(" : ") for line in settingsfile
             if len(line.split(" : ")) == 2]
    settings = {name: value.split(", ") for (name, value) in lines}
