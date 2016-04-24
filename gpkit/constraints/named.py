from collections import defaultdict
from .set import ConstraintSet
from ..varkey import VarKey
from ..keydict import KeyDict
from .. import SignomialsEnabled


class NamedConstraintSet(ConstraintSet):
    defaultnames = ["NamedConstraintSet"]
    _nums = defaultdict(int)
    name = None
    num = None

    def add_modelname(self, name=None):
        if not name and self.__class__.__name__ not in self.defaultnames:
            name = self.__class__.__name__
        if name:
            num = NamedConstraintSet._nums[name]
            NamedConstraintSet._nums[name] += 1
            self.name, self.num = name, num

            add_model_subs = KeyDict()
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
                self.subinplace(add_model_subs)

    def subconstr_str(self, excluded=None):
        "The collapsed appearance of a NamedConstraintSet"
        if self.name:
            return "%s_%s" % (self.name, self.num)

    def subconstr_latex(self, excluded=None):
        "The collapsed appearance of a NamedConstraintSet"
        if self.name:
            return "%s_{%s}" % (self.name, self.num)
