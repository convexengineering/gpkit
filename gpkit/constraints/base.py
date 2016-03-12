"Implements BaseConstraint"
from collections import defaultdict, Iterable
from .set import ConstraintSet
from ..varkey import VarKey
from ..nomials import Variable, Nomial
from .. import SignomialsEnabled


class ConstraintBase(ConstraintSet):
    "A ConstraintSet for making named models with setup methods"
    modelnums = defaultdict(int)

    def __init__(self, substitutions=None, *args, **kwargs):
        name = kwargs.pop("name", self.__class__.__name__)
        constraints = self.setup(*args, **kwargs)
        if hasattr(constraints, "cost"):
            self.cost = constraints.cost
        elif isinstance(constraints[0], Nomial):
            self.cost = constraints[0]
            constraints = constraints[1:]
        if isinstance(constraints[-1], dict):
            substitutions = constraints[-1]
            constraints = constraints[:-1]
        if len(constraints) == 1 and isinstance(constraints[0], Iterable):
            constraints = constraints[0]
        ConstraintSet.__init__(self, constraints, substitutions)
        self._add_models_tovars(name)

    def _add_models_tovars(self, name):
        num = ConstraintBase.modelnums[name]
        self.name, self.num = name, num
        ConstraintBase.modelnums[name] += 1
        add_model_subs = {}
        for vk in self.varkeys:
            descr = dict(vk.descr)
            descr["models"] = descr.pop("models", []) + [name]
            descr["modelnums"] = descr.pop("modelnums", []) + [num]
            newvk = VarKey(**descr)
            add_model_subs[vk] = newvk
            if vk in self.substitutions:
                self.substitutions[newvk] = self.substitutions[vk]
                del self.substitutions[vk]
        with SignomialsEnabled():  # since we're just substituting varkeys.
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

    def subconstr_str(self, excluded=None):
        "The collapsed appearance of a ConstraintBase"
        if self.name:
            return "%s_%s" % (self.name, self.num)

    def subconstr_tex(self, excluded=None):
        "The collapsed appearance of a ConstraintBase"
        if self.name:
            return "%s_{%s}" % (self.name, self.num)
