from collections import defaultdict
from . import ConstraintSet
from ..small_classes import KeyDict
from ..nomials import Variable


def constraintset_iterables(obj):
    if hasattr(obj, "__iter__") and not isinstance(obj, ConstraintSet):
        return ConstraintSet(obj)
    else:
        return obj


def iter_subs(substitutions, constraintset):
    if substitutions:
        yield substitutions
    for constraint in constraintset.flat:
        if hasattr(constraint, "substitutions"):
            dictionary = constraint.substitutions
            constraint.substitutions = {}
            yield dictionary


class ConstraintBase(ConstraintSet):
    modelnums = defaultdict(int)

    def __init__(self, *args, **kwargs):
        name = kwargs.pop("name", self.__class__.__name__)
        num = ConstraintBase.modelnums[name]
        self.num = num
        ConstraintBase.modelnums[name] += 1
        substitutions = kwargs.pop("substitutions", None)
        list.__init__(self, self.setup(*args, **kwargs))
        self.recurse(constraintset_iterables)
        varkeys = KeySet()
        for constraint in self.flat:
            for k, v in dict(constraint.varkeys).items():
                models = k.descr.get("models", [])
                modelnums = k.descr.get("modelnums", [])
                if not models or models[-1] != model or modelnums[-1] != num:
                    k.descr["models"] = models + [model]
                    k.descr["modelnums"] = modelnums + [num]
                varkeys[k] = v
        subs_iter = iter_subs(substitutions, self)
        self.substitutions = KeyDict.with_keys(varkeys, subs_iter)

    def __getitem__(self, key):
        if isinstance(key, int):
            return list.__getitem__(self, key)
        else:
            variables = [Variable(**key.descr) for key in self.varkeys[key]]
            if len(variables) == 1:
                return variables[0]
            else:
                return variables
