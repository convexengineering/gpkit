"global mutable variables"
import os
from collections import defaultdict
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
    except IOError:
        settings_ = {"installed_solvers": [""]}
    if settings_["installed_solvers"] == [""]:
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
    settings_["latex_modelname"] = True
    return settings_


settings = load_settings()


SIGNOMIALS_ENABLED = set()  # the current signomial permissions


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
    # pylint: disable=global-statement
    def __enter__(self):
        SIGNOMIALS_ENABLED.add(True)

    def __exit__(self, type_, val, traceback):
        SIGNOMIALS_ENABLED.remove(True)


VECTORIZATION = []  # the current vectorization shape


class Vectorize(object):
    """Creates an environment in which all variables are
       exended in an additional dimension.
    """
    def __init__(self, dimension_length):
        self.dimension_length = dimension_length

    def __enter__(self):
        "Enters a vectorized environment."
        VECTORIZATION.insert(0, self.dimension_length)

    def __exit__(self, type_, val, traceback):
        "Leaves a vectorized environment."
        VECTORIZATION.pop(0)


MODELS = []     # the current model hierarchy
MODELNUMS = []  # modelnumbers corresponding to MODELS, above
# lookup table for the number of models of each name that have been made
MODELNUM_LOOKUP = defaultdict(int)
# the list of variables named in the current MODELS/MODELNUM environment
NAMEDVARS = defaultdict(list)


def reset_modelnumbers():
    "Zeroes all model number counters"
    for key in MODELNUM_LOOKUP:
        MODELNUM_LOOKUP[key] = 0


def begin_variable_naming(model):
    "Appends a model name and num to the environment."
    MODELS.append(model)
    num = MODELNUM_LOOKUP[model]
    MODELNUMS.append(num)
    MODELNUM_LOOKUP[model] += 1
    return num, (tuple(MODELS), tuple(MODELNUMS))


def end_variable_naming():
    "Pops a model name and num from the environment."
    NAMEDVARS.pop((tuple(MODELS), tuple(MODELNUMS)), None)
    MODELS.pop()
    MODELNUMS.pop()


class NamedVariables(object):
    """Creates an environment in which all variables have
       a model name and num appended to their varkeys.
    """
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        "Enters a named environment."
        begin_variable_naming(self.model)

    def __exit__(self, type_, val, traceback):
        "Leaves a named environment."
        end_variable_naming()
