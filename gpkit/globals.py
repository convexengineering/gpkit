"global mutable variables"
from __future__ import unicode_literals, print_function
import os
import sys
from collections import defaultdict
from six import with_metaclass
from . import build


def load_settings(path=None, firstattempt=True):
    "Load the settings file at SETTINGS_PATH; return settings dict"
    if path is None:
        path = os.sep.join([os.path.dirname(__file__), "env", "settings"])
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
                if sys.version_info >= (3, 0) and name == "installed_solvers":
                    if "mosek" in value:
                        value.remove("mosek")
    except IOError:
        settings_ = {"installed_solvers": [""]}
    if (settings_["installed_solvers"] == [""]
            or ("mosek" in settings_["installed_solvers"]
                and "mosek_version" not in settings_)):
        if firstattempt:
            print("Found no installed solvers, beginning a build.")
            build()
            settings_ = load_settings(path, firstattempt=False)
            if settings_["installed_solvers"] != [""]:
                settings_["just built!"] = True
            else:
                print("""
=============
Build failed!  :(
=============
You may need to install a solver and then `import gpkit` again;
see https://gpkit.readthedocs.io/en/latest/installation.html
for troubleshooting details.

But before you go, please post the output above
(starting from "Found no installed solvers, beginning a build.")
to gpkit@mit.edu or https://github.com/convexengineering/gpkit/issues/new
so we can prevent others from having to see this message.

        Thanks!  :)
""")
    settings_["default_solver"] = settings_["installed_solvers"][0]
    return settings_


settings = load_settings()


class SignomialsEnabledMeta(type):
    "Metaclass to implement falsiness for SignomialsEnabled"

    def __nonzero__(cls):
        return 1 if cls._true else 0

    def __bool__(cls):
        return cls._true


class SignomialsEnabled(with_metaclass(SignomialsEnabledMeta)):  # pylint: disable=no-init
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
    _true = False  # the current signomial permissions

    def __enter__(self):
        SignomialsEnabled._true = True

    def __exit__(self, type_, val, traceback):
        SignomialsEnabled._true = False


class Vectorize(object):
    """Creates an environment in which all variables are
       exended in an additional dimension.
    """
    vectorization = ()  # the current vectorization shape

    def __init__(self, dimension_length):
        self.dimension_length = dimension_length

    def __enter__(self):
        "Enters a vectorized environment."
        Vectorize.vectorization = (self.dimension_length,) + self.vectorization

    def __exit__(self, type_, val, traceback):
        "Leaves a vectorized environment."
        Vectorize.vectorization = self.vectorization[1:]


class NamedVariables(object):
    """Creates an environment in which all variables have
       a model name and num appended to their varkeys.
    """
    lineage = ()  # the current model nesting
    modelnums = defaultdict(int)  # the number of models of each lineage
    namedvars = defaultdict(list)  # variables created in the current nesting

    @classmethod
    def reset_modelnumbers(cls):
        "Clear all model number counters"
        for key in list(cls.modelnums):
            del cls.modelnums[key]

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        "Enters a named environment."
        num = self.modelnums[(self.lineage, self.name)]
        self.modelnums[(self.lineage, self.name)] += 1
        NamedVariables.lineage += ((self.name, num),)  # NOTE: Class reference
        return self.lineage, self.namedvars[self.lineage]

    def __exit__(self, type_, val, traceback):
        "Leaves a named environment."
        del self.namedvars[self.lineage]
        NamedVariables.lineage = self.lineage[:-1]   # NOTE: Class reference
