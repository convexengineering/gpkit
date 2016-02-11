from collections import defaultdict
from . import ConstraintSet
from ..varkey import VarKey
from ..nomials import Variable


class ConstraintBase(ConstraintSet):
    modelnums = defaultdict(int)

    def __init__(self, *args, **kwargs):
        constraints = self.setup(*args, **kwargs)
        ConstraintSet.__init__(self, constraints)
        name = kwargs.pop("name", self.__class__.__name__)
        num = ConstraintBase.modelnums[name]
        self.num = num
        ConstraintBase.modelnums[name] += 1
        add_model_subs = {}
        for vk in self.varkeys:
            descr = dict(vk.descr)
            models = descr.pop("models", [])
            modelnums = descr.pop("modelnums", [])
            descr["models"] = models + [name]
            descr["modelnums"] = models + [name]
            newvk = VarKey(**descr)
            add_model_subs[vk] = newvk
            if vk in self.substitutions:
                self.substitutions[newvk] = self.substitutions.pop(vk)
        self.sub(add_model_subs)

    def __getitem__(self, key):
        if isinstance(key, int):
            return list.__getitem__(self, key)
        else:
            variables = [Variable(**key.descr) for key in self.varkeys[key]]
            if len(variables) == 1:
                return variables[0]
            else:
                return variables
