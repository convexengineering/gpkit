# -*- coding: utf-8 -*-
"""Lightweight GP Modeling Package

    For examples please see the examples folder.

    Requirements
    ------------
    numpy
    MOSEK or CVXOPT
    scipy(optional): for complete sparse matrix support
    sympy(optional): for latex printing in iPython Notebook

    Attributes
    ----------
    settings : dict
        Contains settings loaded from ``./env/settings``
"""

__version__ = "0.4.0"
UNIT_REGISTRY = None
SIGNOMIALS_ENABLED = False

from os import sep as os_sep
from os.path import dirname as os_path_dirname
UNITDEF_PATH = os_sep.join([os_path_dirname(__file__), "pint", "units.txt"])
SETTINGS_PATH = os_sep.join([os_path_dirname(__file__), "env", "settings"])


def mdparse(filename, return_tex=False):
    with open(filename) as f:
        py_lines = []
        texmd_lines = []
        block_idx = 0
        in_replaced_block = False
        for line in f:
            line = line[:-1]  # remove newline
            texmd_content = line if not in_replaced_block else ""
            texmd_lines.append(texmd_content)
            py_content = ""
            if line == "```python":
                block_idx = 1
            elif block_idx and line == "```":
                block_idx = 0
                if in_replaced_block:
                    texmd_lines[-1] = ""  # remove the ``` line
                in_replaced_block = False
            elif block_idx:
                py_content = line
                block_idx += 1
                if block_idx == 2:
                    # parsing first line of code block
                    if line[:8] == "#inPDF: ":
                        texmd_lines[-2] = ""  # remove the ```python line
                        texmd_lines[-1] = ""  # remove the #inPDF line
                        in_replaced_block = True
                        if line[8:21] == "replace with ":
                            texmd_lines.append("\\input{%s}" % line[21:])
            elif line:
                py_content = "# " + line
            py_lines.append(py_content)
        if not return_tex:
            return "\n".join(py_lines)
        else:
            return "\n".join(py_lines), "\n".join(texmd_lines)


def mdmake(filename, make_tex=True):
    mdpy, texmd = mdparse(filename, return_tex=True)
    with open(filename+".py", "w") as f:
        f.write(mdpy)
    if make_tex:
        with open(filename+".tex.md", "w") as f:
            f.write(texmd)
    return open(filename+".py")


def enable_units(path=UNITDEF_PATH):
    """Enables units support in a particular instance of GPkit.

    Posynomials created after calling this are incompatible with those created
    before.

    If gpkit is imported multiple times, this needs to be run each time."""
    global units, DimensionalityError, UNIT_REGISTRY
    try:
        import pint
        if UNIT_REGISTRY is None:
            UNIT_REGISTRY = pint.UnitRegistry(path)
        units = UNIT_REGISTRY
        DimensionalityError = pint.DimensionalityError
    except ImportError:
        print("Optional Python units library (Pint) not installed;"
              " unit support disabled.")
        disable_units()


def disable_units():
    """Disables units support in a particular instance of GPkit.

    Posynomials created after calling this are incompatible with those created
    before.

    If gpkit is imported multiple times, this needs to be run each time.

    The correct way to call this is:
        import gpkit
        gpkit.disable_units()

    The following will *not* have the intended effect:
        from gpkit import disable_units
        disable_units()
    """
    global units, DimensionalityError

    class DummyUnits(object):
        "Dummy class to replace missing pint"
        class Quantity(object):
            pass

        def __nonzero__(self):
            return 0

        def __bool__(self):
            return False

        def __getattr__(self, attr):
            return 1

    units = DummyUnits()
    DimensionalityError = ValueError

enable_units()


class SignomialsEnabled(object):
    """Class to put up and tear down signomial support in an instance of GPkit.

    Example
    -------
    >>> import gpkit
    >>> x = gpkit.Variable("x")
    >>> y = gpkit.Variable("y", 0.1)
    >>> with SignomialsEnabled():
    >>>     constraints = [x >= 1-y]
    >>> gpkit.Model(x, constraints).localsolve()
    """

    def __enter__(self):
        global SIGNOMIALS_ENABLED
        SIGNOMIALS_ENABLED = True

    def __exit__(self, type_, val, traceback):
        global SIGNOMIALS_ENABLED
        SIGNOMIALS_ENABLED = False


from .nomials import *
from .constraints import *
from .constraints.base import ConstraintBase
from .constraints.link import LinkConstraint
from .geometric_program import GeometricProgram
from .signomial_program import SignomialProgram
from .varkey import VarKey
from .model import Model

if units:
    def _subvert_pint():
        """
        When gpkit objects appear in mathematical operations with pint
        Quantity objects, let the gpkit implementations determine what to do
        """
        def skip_if_gpkit_objects(fallback, objects=(NomialArray, Signomial)):
            def _newfn(self, other):
                if isinstance(other, objects):
                    return NotImplemented
                else:
                    return getattr(self, fallback)(other)
            return _newfn

        for op in "eq ge le add mul div truediv floordiv".split():
            dunder = "__%s__" % op
            trunder = "___%s___" % op
            original = getattr(units.Quantity, dunder)
            setattr(units.Quantity, trunder, original)
            newfn = skip_if_gpkit_objects(fallback=trunder)
            setattr(units.Quantity, dunder, newfn)

    _subvert_pint()


def load_settings(path=SETTINGS_PATH):
    try:
        with open(path) as settingsfile:
            lines = [line[:-1].split(" : ") for line in settingsfile
                     if len(line.split(" : ")) == 2]
            settings_ = {name: value.split(", ") for name, value in lines}
            for name, value in settings_.items():
                # hack to flatten 1-element lists,
                # unless they're the solver list
                if len(value) == 1 and name != "installed_solvers":
                    settings_[name] = value[0]
    except IOError:
        print("Could not load settings file.")
        settings_ = {"installed_solvers": [""]}
    return settings_
settings = load_settings()
settings["latex_modelname"] = True
