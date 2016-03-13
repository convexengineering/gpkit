"Implements Model"
from collections import defaultdict
from .costed import CostedConstraintSet
from ..varkey import VarKey
from ..nomials import Monomial
from .link import LinkConstraint
from .. import SignomialsEnabled


class Model(CostedConstraintSet):
    "A ConstraintSet for convenient solving and setup"
    _nums = defaultdict(int)
    name = None
    num = None

    def __init__(self, cost=None, constraints=None,
                 substitutions=None, name=None):
        if hasattr(self, "setup"):
            # temporarily fail gracefully for backwards compatibility
            raise RuntimeWarning(
                "setup methods are no longer used in GPkit. "
                "To initialize a model, rename your setup method as "
                "__init__(self, **kwargs) and have it call "
                "Model.__init__(self, cost, constraints, **kwargs) at the end.")
        cost = cost if cost is not None else Monomial(1)
        constraints = constraints if constraints is not None else []
        CostedConstraintSet.__init__(self, cost, constraints, substitutions)
        if self.__class__.__name__ != "Model" and not name:
            name = self.__class__.__name__
        if name:
            self.name = name
            self.num = Model._nums[name]
            Model._nums[name] += 1
            self._add_modelname_tovars(self.name, self.num)

    def link(self, other, include_only=None, exclude=None):
        "Connects this model with a set of constraints"
        lc = LinkConstraint([self, other], include_only, exclude)
        cost = self.cost.sub(lc.linked)
        return Model(cost, [lc], lc.substitutions)

    def _add_modelname_tovars(self, name, num):
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

    def subconstr_str(self, excluded=None):
        "The collapsed appearance of a ConstraintBase"
        if self.name:
            return "%s_%s" % (self.name, self.num)

    def subconstr_tex(self, excluded=None):
        "The collapsed appearance of a ConstraintBase"
        if self.name:
            return "%s_{%s}" % (self.name, self.num)
